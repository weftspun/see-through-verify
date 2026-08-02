// BasicTransformerBlock compare: our direct f32 of one mid transformer_blocks.0
// (LN+self-attn+LN+cross-attn+LN+GEGLU) vs the ggml basic_transformer_block
// oracle. Isolates the UNet attention wiring from the transformer3d plumbing.
// Usage: btblock_compare <unet.safetensors> <btblock.bin>
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
    // Q,K,V token-major (T, C) with channel c = h*hd + d (head-major)
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
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors btblock.bin\n", argv[0]); return 1; }
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
    int32_t hdr[5]; fread(hdr,4,5,f);
    int C=hdr[0], S=hdr[1], F=hdr[2], Tk=hdr[3], EHD=hdr[4];
    size_t nout=(size_t)C*S*F;
    std::vector<float> ref(nout), xd((size_t)C*S*F), ed((size_t)EHD*Tk*F);
    fread(ref.data(),4,nout,f);
    fread(xd.data(),4,xd.size(),f);
    fread(ed.data(),4,ed.size(),f);
    fclose(f);
    const int heads = C/64;
    printf("btblock: C=%d S=%d F=%d Tk=%d EHD=%d heads=%d\n", C,S,F,Tk,EHD,heads);

    const std::string pre = "mid_block.attentions.0.transformer_blocks.0";
    auto n1g=load_w(pre+".norm1.weight"), n1b=load_w(pre+".norm1.bias");
    auto q1=load_w(pre+".attn1.to_q.weight"), k1=load_w(pre+".attn1.to_k.weight"), v1=load_w(pre+".attn1.to_v.weight");
    auto o1=load_w(pre+".attn1.to_out.0.weight"), ob1=load_w(pre+".attn1.to_out.0.bias");
    auto n2g=load_w(pre+".norm2.weight"), n2b=load_w(pre+".norm2.bias");
    auto q2=load_w(pre+".attn2.to_q.weight"), k2=load_w(pre+".attn2.to_k.weight"), v2=load_w(pre+".attn2.to_v.weight");
    auto o2=load_w(pre+".attn2.to_out.0.weight"), ob2=load_w(pre+".attn2.to_out.0.bias");
    auto n3g=load_w(pre+".norm3.weight"), n3b=load_w(pre+".norm3.bias");
    auto f1w=load_w(pre+".ff.net.0.proj.weight"), f1b=load_w(pre+".ff.net.0.proj.bias");
    auto f2w=load_w(pre+".ff.net.2.weight"), f2b=load_w(pre+".ff.net.2.bias");
    int g=(int)f1w.size()/(2*C);

    // x is token-major (S, C) [F=1]
    std::vector<float> h = xd;  // (S*C)
    // LN1
    std::vector<float> ln1((size_t)S*C); layer_norm_tokens(h.data(), ln1.data(), n1g.data(), n1b.data(), S, C);
    // self-attn attn1
    std::vector<float> Q((size_t)S*C),K((size_t)S*C),V((size_t)S*C),A((size_t)S*C),Ao((size_t)S*C);
    token_linear(ln1.data(), q1.data(), nullptr, Q.data(), S, C, C);
    token_linear(ln1.data(), k1.data(), nullptr, K.data(), S, C, C);
    token_linear(ln1.data(), v1.data(), nullptr, V.data(), S, C, C);
    mha_tokens(Q.data(), K.data(), V.data(), A.data(), S, S, C, heads);
    token_linear(A.data(), o1.data(), ob1.data(), Ao.data(), S, C, C);
    for (size_t i=0;i<h.size();i++) h[i]+=Ao[i];
    // LN2
    std::vector<float> ln2((size_t)S*C); layer_norm_tokens(h.data(), ln2.data(), n2g.data(), n2b.data(), S, C);
    // cross-attn attn2 on ehs
    std::vector<float> Q2((size_t)S*C),K2((size_t)Tk*C),V2((size_t)Tk*C),A2((size_t)S*C),Ao2((size_t)S*C);
    token_linear(ln2.data(), q2.data(), nullptr, Q2.data(), S, C, C);
    token_linear(ed.data(), k2.data(), nullptr, K2.data(), Tk, EHD, C);
    token_linear(ed.data(), v2.data(), nullptr, V2.data(), Tk, EHD, C);
    mha_tokens(Q2.data(), K2.data(), V2.data(), A2.data(), S, Tk, C, heads);
    token_linear(A2.data(), o2.data(), ob2.data(), Ao2.data(), S, C, C);
    for (size_t i=0;i<h.size();i++) h[i]+=Ao2[i];
    // LN3 + GEGLU
    std::vector<float> ln3((size_t)S*C); layer_norm_tokens(h.data(), ln3.data(), n3g.data(), n3b.data(), S, C);
    std::vector<float> p1((size_t)S*2*g);
    token_linear(ln3.data(), f1w.data(), f1b.data(), p1.data(), S, C, 2*g);
    std::vector<float> glu((size_t)S*g);
    for (int s=0;s<S;s++) for (int c=0;c<g;c++) {
        float val=p1[(size_t)s*2*g+c], gate=p1[(size_t)s*2*g+g+c];
        glu[(size_t)s*g+c] = val * gelu_erf(gate);
    }
    std::vector<float> ff((size_t)S*C);
    token_linear(glu.data(), f2w.data(), f2b.data(), ff.data(), S, g, C);
    for (size_t i=0;i<h.size();i++) h[i]+=ff[i];

    double max_abs=0, sum_abs=0; size_t cnt=h.size();
    for (size_t i=0;i<cnt;i++){ double dd=fabs((double)h[i]-(double)ref[i]); if(dd>max_abs)max_abs=dd; sum_abs+=dd; }
    double cos=0,na=0,nb=0;
    for (size_t i=0;i<cnt;i++){cos+=(double)h[i]*ref[i]; na+=(double)h[i]*h[i]; nb+=(double)ref[i]*ref[i];}
    cos/=sqrt(na*nb);
    printf("mine[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
    printf("ora [0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
    printf("btblock self+cross+GEGLU: max_abs=%.6f mean=%.6f cosine=%.6f\n", max_abs, sum_abs/cnt, cos);
    printf("%s\n", max_abs < 0.1 && cos > 0.999 ? "VALIDATION PASS — basic_transformer_block matches ggml oracle"
                   : "VALIDATION FAIL");
    return (max_abs < 0.1 && cos > 0.999) ? 0 : 1;
}
