// btblock stage compare: check LN1 then each subsequent stage against the
// ggml oracle v2 taps, to localize exactly where our direct math diverges.
// Usage: btblock_stage_compare <unet.safetensors> <btblock2.bin>
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
static void compare(const char *name, const std::vector<float>&a, const std::vector<float>&b){
    double max=0, sum=0; size_t n=std::min(a.size(),b.size());
    for(size_t i=0;i<n;i++){double dd=fabs((double)a[i]-(double)b[i]); if(dd>max)max=dd; sum+=dd;}
    double cos=0,na=0,nb=0; for(size_t i=0;i<n;i++){cos+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i];} cos/=sqrt(na*nb);
    printf("  %s: max_abs=%.6f mean=%.6f cos=%.6f\n", name, max, sum/n, cos);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors btblock2.bin\n", argv[0]); return 1; }
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
    int32_t hdr[6]; fread(hdr,4,6,f);
    int C=hdr[0], S=hdr[1], F=hdr[2], Tk=hdr[3], EHD=hdr[4], N=hdr[5];
    size_t NEL=(size_t)C*S*F;
    std::vector<float> xd((size_t)C*S*F), ed((size_t)EHD*Tk*F);
    fread(xd.data(),4,xd.size(),f);
    fread(ed.data(),4,ed.size(),f);
    std::vector<std::vector<float>> taps((size_t)N, std::vector<float>(NEL));
    for (int i=0;i<N;i++) fread(taps[i].data(),4,NEL,f);
    fclose(f);
    const int heads=C/64;
    const std::string pre = "mid_block.attentions.0.transformer_blocks.0";
    auto n1g=load_w(pre+".norm1.weight");
    std::vector<float> n1b=load_w(pre+".norm1.bias");
    auto q1=load_w(pre+".attn1.to_q.weight");
    auto o1=load_w(pre+".attn1.to_out.0.weight");
    std::vector<float> ob1=load_w(pre+".attn1.to_out.0.bias");

    // LN1 check
    std::vector<float> ln1((size_t)S*C);
    layer_norm_tokens(xd.data(), ln1.data(), n1g.data(), n1b.data(), S, C);
    compare("ln1", ln1, taps[0]);

    // self-attn attn1
    auto k1=load_w(pre+".attn1.to_k.weight"), v1=load_w(pre+".attn1.to_v.weight");
    std::vector<float> Q((size_t)S*C),K((size_t)S*C),V((size_t)S*C),A((size_t)S*C),Ao((size_t)S*C);
    token_linear(ln1.data(), q1.data(), nullptr, Q.data(), S, C, C);
    token_linear(ln1.data(), k1.data(), nullptr, K.data(), S, C, C);
    token_linear(ln1.data(), v1.data(), nullptr, V.data(), S, C, C);
    mha_tokens(Q.data(), K.data(), V.data(), A.data(), S, S, C, heads);
    token_linear(A.data(), o1.data(), ob1.data(), Ao.data(), S, C, C);
    compare("attn1 raw mha (kqv)", A, taps[7]);   // a1 = raw kqv
    compare("attn1 after to_out", Ao, taps[1]);
    return 0;
}
