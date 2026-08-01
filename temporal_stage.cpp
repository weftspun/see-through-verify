// Temporal stage compare: check each stage of cross_frame_block against the
// oracle2 taps to localize the divergence.
// Usage: temporal_stage <unet.safetensors> <temporal2.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *data_start) {
    FILE *f = fopen(path, "rb");
    uint8_t buf[8]; fread(buf, 1, 8, f);
    uint64_t hdr_len = 0;
    for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)buf[j] << (j * 8);
    std::string json((size_t)hdr_len, 0);
    fread(&json[0], 1, hdr_len, f); *data_start = 8 + hdr_len; fclose(f);
    std::vector<SfTensor> tensors; size_t i = 0;
    while (i < json.size() && json[i] != '}') {
        while (i < json.size() && (json[i]==' '||json[i]==10||json[i]==9||json[i]==13||json[i]==',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }
        size_t ns = ++i; while (i < json.size() && json[i] != '"') i++;
        std::string name = json.substr(ns, i - ns); i++;
        if (name == "__metadata__") { while (i < json.size() && json[i] != '}') i++; i++; continue; }
        while (i < json.size() && json[i] != '{') i++; i++;
        SfTensor t; t.name = name; t.offset = 0; t.size = 0;
        while (i < json.size() && json[i] != '}') {
            while (i < json.size() && (json[i]==' '||json[i]==10||json[i]==9||json[i]==13||json[i]==',')) i++;
            if (json[i] == '}') break; if (json[i] != '"') { i++; continue; }
            size_t fs = ++i; while (i < json.size() && json[i] != '"') i++;
            std::string field = json.substr(fs, i - fs); i++;
            while (i < json.size() && json[i] != ':') i++; i++;
            if (field == "dtype") {
                if (json[i] == '"') { size_t vs = ++i; while (i < json.size() && json[i] != '"') i++; t.dtype = json.substr(vs, i - vs); i++; }
            } else if (field == "shape") {
                if (json[i] == '[') { i++; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; t.shape.push_back(strtol(&json[i], &end, 10)); i = end - &json[0]; } } i++; }
            } else if (field == "data_offsets") {
                if (json[i] == '[') { i++; int idx=0; uint64_t vals[2]={0,0}; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; vals[idx++]=strtoull(&json[i], &end, 10); i=end-&json[0]; } } i++; t.offset=vals[0]; t.size=vals[1]-vals[0]; }
            }
        }
        i++; tensors.push_back(t);
    }
    return tensors;
}
static std::vector<uint8_t> load_tensor_data(const char *sf, size_t ds, size_t off, size_t sz) {
    FILE *f = fopen(sf, "rb"); fseeko(f, (off_t)(ds+off), SEEK_SET);
    std::vector<uint8_t> data(sz); fread(data.data(), 1, sz, f); fclose(f); return data;
}
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) { uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16; memcpy(&out[i], &u32, 4); }
    return out;
}
static void token_linear(const float *x, const float *W, const float *b, float *y,
                         int T, int Cin, int Cout) {
    for (int t = 0; t < T; t++)
        for (int o = 0; o < Cout; o++) {
            float s = b ? b[o] : 0.f;
            for (int k = 0; k < Cin; k++) s += x[t*Cin+k] * W[o*Cin+k];
            y[t*Cout+o] = s;
        }
}
static void layer_norm_tokens(const float *x, float *y, const float *g, const float *bb,
                              int T, int D, float eps=1e-5f) {
    for (int t = 0; t < T; t++) {
        const float *xr = x + t*D; float *yr = y + t*D;
        float mn=0; for (int d=0;d<D;d++) mn+=xr[d]; mn/=D;
        float va=0; for (int d=0;d<D;d++){float q=xr[d]-mn; va+=q*q;} va/=D;
        float inv=1.f/sqrtf(va+eps);
        for (int d=0;d<D;d++) yr[d]=(xr[d]-mn)*inv*g[d]+bb[d];
    }
}
static float gelu_erf(float x) { return 0.5f*x*(1.0f+erff(x/1.41421356f)); }
static void mha_tokens(const float *Q, const float *K, const float *V, float *O,
                       int Tq, int Tk, int C, int H) {
    const int hd = C / H;
    std::vector<float> sc(Tq*Tk);
    for (int h=0;h<H;h++) {
        for (int i=0;i<Tq;i++)
          for (int j=0;j<Tk;j++) {
            float s=0;
            for (int d=0;d<hd;d++) s += Q[i*C+h*hd+d]*K[j*C+h*hd+d];
            sc[i*Tk+j] = s / sqrtf((float)hd);
          }
        for (int i=0;i<Tq;i++) {
            float mx=-1e30f;
            for (int j=0;j<Tk;j++) if (sc[i*Tk+j]>mx) mx=sc[i*Tk+j];
            float sm=0; for (int j=0;j<Tk;j++){ sc[i*Tk+j]=expf(sc[i*Tk+j]-mx); sm+=sc[i*Tk+j]; }
            for (int j=0;j<Tk;j++) sc[i*Tk+j]/=sm;
        }
        for (int i=0;i<Tq;i++)
          for (int d=0;d<hd;d++) {
            float s=0;
            for (int j=0;j<Tk;j++) s += sc[i*Tk+j]*V[j*C+h*hd+d];
            O[i*C+h*hd+d] = s;
          }
    }
}
static void cmp(const char *nm, const std::vector<float>&a, const std::vector<float>&b){
    double m=0,s=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; s+=d;}
    double c=0,na=0,nb=0; for(size_t i=0;i<n;i++){c+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} c/=sqrt(na*nb);
    printf("  %s: max=%.5f mean=%.5f cos=%.5f\n",nm,m,s/n,c);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors temporal2.bin\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);
    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name == suffix) return &t;
        return nullptr;
    };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        if (!t) { fprintf(stderr, "missing %s\n", suffix.c_str()); exit(1); }
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(sf_path, ds, t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };

    FILE *f = fopen(argv[2], "rb");
    int32_t hdr[5]; fread(hdr,4,5,f);
    int C=hdr[0], S=hdr[1], F=hdr[2], n_head=hdr[3], N=hdr[4];
    size_t n=(size_t)C*S*F;
    std::vector<float> xd(n);
    fread(xd.data(),4,n,f);
    std::vector<std::vector<float>> taps(N, std::vector<float>(n));
    for (int i=0;i<N;i++) fread(taps[i].data(),4,n,f);
    fclose(f);
    const int heads = C/64;
    // x is (C,S,F) ne0=C. xp (oracle tap0) is (C,F,S) ne0=C: xp[c+C*f+C*F*s]=x[c+C*s+C*S*f]
    // We validate via token-major per spatial s: x2[s][f*C+c]
    // Build oracle taps as x2taps[s][f*C+c] from taps[i] (which are (C,F,S) ne0=C):
    auto as_tok = [&](int ti, int s2, int f2, int c2)->float{
        // ov is (C,F,S) ne0=C: index=c2 + C*f2 + C*F*s2
        return taps[ti][(size_t)c2 + (size_t)C*f2 + (size_t)(C*F)*s2];
    };

    const std::string pre = "mid_block.attentions.0.temporal_transformer_blocks.0";
    auto tn_in_g=load_w(pre+".norm_in.weight"), tn_in_b=load_w(pre+".norm_in.bias");
    auto tff1w=load_w(pre+".ff_in.net.0.proj.weight"), tff1b=load_w(pre+".ff_in.net.0.proj.bias");
    auto tff2w=load_w(pre+".ff_in.net.2.weight"), tff2b=load_w(pre+".ff_in.net.2.bias");
    auto tn1g=load_w(pre+".norm1.weight"), tn1b=load_w(pre+".norm1.bias");
    auto tq1=load_w(pre+".attn1.to_q.weight"), tk1=load_w(pre+".attn1.to_k.weight"), tv1=load_w(pre+".attn1.to_v.weight");
    auto to1=load_w(pre+".attn1.to_out.0.weight"), tob1=load_w(pre+".attn1.to_out.0.bias");
    auto tn3g=load_w(pre+".norm3.weight"), tn3b=load_w(pre+".norm3.bias");
    auto tff1=load_w(pre+".ff.net.0.proj.weight"), tff1b2=load_w(pre+".ff.net.0.proj.bias");
    auto tff2=load_w(pre+".ff.net.2.weight"), tff2b2=load_w(pre+".ff.net.2.bias");
    int tg=(int)tff1w.size()/(2*C);

    // Build x2[s][f*C+c] from original xd (C,S,F) ne0=C: xd[c + C*s + C*S*f]
    std::vector<std::vector<float>> x2(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++) for (int f2=0;f2<F;f2++) for (int c2=0;c2<C;c2++)
        x2[s2][(size_t)f2*C+c2] = xd[(size_t)c2 + (size_t)C*s2 + (size_t)C*S*f2];

    // 1. LN(norm_in) -> compare tap1
    std::vector<std::vector<float>> hn(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++) layer_norm_tokens(x2[s2].data(), hn[s2].data(), tn_in_g.data(), tn_in_b.data(), F, C);
    std::vector<float> mine1(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++) mine1[(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2]=hn[s2][(size_t)f2*C+c2];
    std::vector<float> ref1(n); for(std::size_t i=0;i<n;i++) ref1[i]=taps[1][i];
    cmp("norm_in", mine1, ref1);

    // 2. GEGLU ff_in -> tap2
    std::vector<std::vector<float>> hff(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++){
        std::vector<float> p((size_t)F*2*tg), gl((size_t)F*tg), ff((size_t)F*C);
        token_linear(hn[s2].data(), tff1w.data(), tff1b.data(), p.data(), F, C, 2*tg);
        for(int t=0;t<F;t++) for(int c=0;c<tg;c++){float va=p[(size_t)t*2*tg+c], ga=p[(size_t)t*2*tg+tg+c]; gl[(size_t)t*tg+c]=va*gelu_erf(ga);}
        token_linear(gl.data(), tff2w.data(), tff2b.data(), ff.data(), F, tg, C);
        hff[s2]=ff;
    }
    std::vector<float> mine2(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++) mine2[(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2]=hff[s2][(size_t)f2*C+c2];
    std::vector<float> ref2(n); for(std::size_t i=0;i<n;i++) ref2[i]=taps[2][i];
    cmp("ff_in", mine2, ref2);

    // 3. +res (ff_in + xp) -> tap3
    std::vector<float> mine3(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++){
        size_t idx=(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2;
        mine3[idx]=hff[s2][(size_t)f2*C+c2]+x2[s2][(size_t)f2*C+c2];
    }
    std::vector<float> ref3(n); for(std::size_t i=0;i<n;i++) ref3[i]=taps[3][i];
    cmp("ff_in+res", mine3, ref3);

    // 4. LN(norm1) -> tap4: n1 = LN(norm1)(hr) where hr = ff_in+res
    std::vector<std::vector<float>> hr_s(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++) for(size_t i=0;i<hr_s[s2].size();i++) hr_s[s2][i]=hff[s2][i]+x2[s2][i];
    std::vector<std::vector<float>> hn1(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++) layer_norm_tokens(hr_s[s2].data(), hn1[s2].data(), tn1g.data(), tn1b.data(), F, C);
    std::vector<float> mine4(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++) mine4[(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2]=hn1[s2][(size_t)f2*C+c2];
    std::vector<float> ref4(n); for(std::size_t i=0;i<n;i++) ref4[i]=taps[4][i];
    cmp("norm1 (LN of ff_in+res)", mine4, ref4);

    // 5. self-attn over F (attn1) -> tap5. input n1 = LN(hr).
    std::vector<std::vector<float>> hattn(S, std::vector<float>((size_t)F*C));
    for (int s2=0;s2<S;s2++){
        std::vector<float> Q((size_t)F*C),K((size_t)F*C),V((size_t)F*C),Ao((size_t)F*C);
        token_linear(hn1[s2].data(), tq1.data(), nullptr, Q.data(), F, C, C);
        token_linear(hn1[s2].data(), tk1.data(), nullptr, K.data(), F, C, C);
        token_linear(hn1[s2].data(), tv1.data(), nullptr, V.data(), F, C, C);
        mha_tokens(Q.data(), K.data(), V.data(), Ao.data(), F, F, C, heads);
        std::vector<float> aout((size_t)F*C);
        token_linear(Ao.data(), to1.data(), tob1.data(), aout.data(), F, C, C);
        hattn[s2]=aout;
    }
    std::vector<float> mine5(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++) mine5[(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2]=hattn[s2][(size_t)f2*C+c2];
    std::vector<float> ref5(n); for(std::size_t i=0;i<n;i++) ref5[i]=taps[5][i];
    cmp("attn1 (self over F)", mine5, ref5);

    // 6. +attn1 -> tap6 (hr + attn)
    std::vector<float> mine6(n);
    for (int s2=0;s2<S;s2++) for(int f2=0;f2<F;f2++) for(int c2=0;c2<C;c2++){
        size_t idx=(size_t)c2+(size_t)C*f2+(size_t)(C*F)*s2;
        mine6[idx]=hr_s[s2][(size_t)f2*C+c2]+hattn[s2][(size_t)f2*C+c2];
    }
    std::vector<float> ref6(n); for(std::size_t i=0;i<n;i++) ref6[i]=taps[6][i];
    cmp("+attn1", mine6, ref6);
    return 0;
}
