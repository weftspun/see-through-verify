// Temporal cross-frame block compare: validate our direct implement against
// the ggml oracle. x is (C,S,F), cross-frame permutes to (C,F,S) so attention
// runs across F (frames) as tokens, batched over S (spatial).
// Usage: temporal_compare <unet.safetensors> <temporal.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// shared fast loader + Accelerate-BLAS token_linear (mmap once, numpy's C BLAS)
#include "verify_common.h"
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) { uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16; memcpy(&out[i], &u32, 4); }
    return out;
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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors temporal.bin\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);
    if (map_safetensors(sf_path, ds) != 0) return 1;
    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name == suffix) return &t;
        return nullptr;
    };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        if (!t) { fprintf(stderr, "missing %s\n", suffix.c_str()); exit(1); }
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(ds + t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };

    FILE *f = fopen(argv[2], "rb");
    int32_t hdr[4]; fread(hdr,4,4,f);
    int C=hdr[0], S=hdr[1], F=hdr[2], n_head=hdr[3];
    size_t n=(size_t)C*S*F;
    std::vector<float> ref(n), xd(n);
    fread(ref.data(),4,n,f); fread(xd.data(),4,n,f);
    fclose(f);
    const int heads = C/64;
    printf("temporal: C=%d S=%d F=%d heads=%d\n", C,S,F,heads);

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

    // input x in (C,S,F): ne0=C fastest. h[f][s*C+c] = xd[c + C*s + C*S*f]
    // cross_frame_block: x (C,S,F) -> permute(0,2,1,3) -> (C,F,S): x2[c][f][s] = x[c][s][f]
    // attention runs over F tokens, batched over S (Tke=F, batch=S).
    // Build x2mat[s][f*C+c] (token=f, dim=C) from xd:
    std::vector<std::vector<float>> x2(S, std::vector<float>((size_t)F*C));
    for (int f=0;f<F;f++) for (int s=0;s<S;s++) for (int c=0;c<C;c++)
        x2[s][(size_t)f*C+c] = xd[(size_t)c + (size_t)C*s + (size_t)C*S*f];

    // reference: res=x2; h=LN(norm_in)(x2); h=GEGLU(ff_in)(h); h+=res;
    //   n1=LN(norm1)(h); h+=attn1(self over F); h=GEGLU(ff)(LN(norm3)(h))
    // return permute back
    // 1. LN(norm_in) + ff_in + res
    for (int s=0;s<S;s++){
        std::vector<float> ln((size_t)F*C), p((size_t)F*2*tg), glu_v((size_t)F*tg), ff((size_t)F*C);
        layer_norm_tokens(x2[s].data(), ln.data(), tn_in_g.data(), tn_in_b.data(), F, C);
        token_linear(ln.data(), tff1w.data(), tff1b.data(), p.data(), F, C, 2*tg);
        for (int t=0;t<F;t++) for (int c=0;c<tg;c++){
            float val=p[(size_t)t*2*tg+c], ga=p[(size_t)t*2*tg+tg+c];
            glu_v[(size_t)t*tg+c]=val*gelu_erf(ga);
        }
        token_linear(glu_v.data(), tff2w.data(), tff2b.data(), ff.data(), F, tg, C);
        for (size_t i=0;i<x2[s].size();i++) x2[s][i]+=ff[i];
    }
    // 2. LN(norm1) + self-attn over F
    for (int s=0;s<S;s++){
        std::vector<float> ln((size_t)F*C), Q((size_t)F*C),K((size_t)F*C),V((size_t)F*C),Ao((size_t)F*C),Ao2((size_t)F*C);
        layer_norm_tokens(x2[s].data(), ln.data(), tn1g.data(), tn1b.data(), F, C);
        token_linear(ln.data(), tq1.data(), nullptr, Q.data(), F, C, C);
        token_linear(ln.data(), tk1.data(), nullptr, K.data(), F, C, C);
        token_linear(ln.data(), tv1.data(), nullptr, V.data(), F, C, C);
        mha_tokens(Q.data(), K.data(), V.data(), Ao.data(), F, F, C, heads);
        token_linear(Ao.data(), to1.data(), tob1.data(), Ao2.data(), F, C, C);
        for (size_t i=0;i<x2[s].size();i++) x2[s][i]+=Ao2[i];
    }
    // 3. final FF (no residual)
    for (int s=0;s<S;s++){
        std::vector<float> ln((size_t)F*C), p((size_t)F*2*tg), glu_v((size_t)F*tg), ff((size_t)F*C);
        layer_norm_tokens(x2[s].data(), ln.data(), tn3g.data(), tn3b.data(), F, C);
        token_linear(ln.data(), tff1w.data(), tff1b2.data(), p.data(), F, C, 2*tg);
        for (int t=0;t<F;t++) for (int c=0;c<tg;c++){
            float val=p[(size_t)t*2*tg+c], ga=p[(size_t)t*2*tg+tg+c];
            glu_v[(size_t)t*tg+c]=val*gelu_erf(ga);
        }
        token_linear(glu_v.data(), tff2w.data(), tff2b2.data(), ff.data(), F, tg, C);
        x2[s]=ff;
    }
    // permute back: out (C,S,F): out[c+C*s+C*S*f] = x2[s][f*C+c]
    std::vector<float> mine(n);
    for (int f=0;f<F;f++) for (int s=0;s<S;s++) for (int c=0;c<C;c++)
        mine[(size_t)c + (size_t)C*s + (size_t)C*S*f] = x2[s][(size_t)f*C+c];

    double max_abs=0, sum_abs=0; size_t cnt=n;
    for (size_t i=0;i<cnt;i++){ double dd=fabs((double)mine[i]-(double)ref[i]); if(dd>max_abs)max_abs=dd; sum_abs+=dd; }
    double cos=0,na=0,nb=0;
    for (size_t i=0;i<cnt;i++){cos+=(double)mine[i]*ref[i]; na+=(double)mine[i]*mine[i]; nb+=(double)ref[i]*ref[i];}
    cos/=sqrt(na*nb);
    printf("mine[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", mine[0],mine[1],mine[2],mine[3],mine[4],mine[5],mine[6],mine[7]);
    printf("ora [0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
    printf("temporal cross-frame: max_abs=%.6f mean=%.6f cosine=%.6f\n", max_abs, sum_abs/cnt, cos);
    printf("%s\n", max_abs < 0.1 && cos > 0.999 ? "VALIDATION PASS — temporal cross-frame matches ggml oracle"
                   : "VALIDATION FAIL");
    return (max_abs < 0.1 && cos > 0.999) ? 0 : 1;
}
