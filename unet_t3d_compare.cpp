// Mid-block transformer3d compare: full direct f32 reimplementation of
// diffusers Transformer3DModel (transformer3d in unet_frame.cpp) against
// the ggml oracle. Validates the ENTIRE remaining UNet attention stack:
// GroupNorm->proj_in -> [10 x (LN+self-attn+LN+cross-attn+LN+GEGLU) with
// temporal cross-frame after every 2nd] -> proj_out -> residual.
// All F frames are processed together because temporal blocks mix frames.
// Inputs (from unet_oracle): mid_resnet0.bin (x, [W,H,C,F] WHCN),
//   ehs2.bin (conditioning [2048,77,F]), target mid_attn.bin.
// Usage: unet_t3d_compare <unet.safetensors> <oracle_dir> [F]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <Accelerate/Accelerate.h>

// token-major linear y[t,o] = sum_k x[t,k]*W[o,k] + b[o], where W is stored
// [Cout, Cin] (row = output). This is the GEMM  C = X·W^T + b. Routed through
// Apple's Accelerate BLAS (the same C BLAS numpy uses) so the harness runs in
// seconds, not minutes, while staying pure C++.
static void token_linear(const float *x, const float *W, const float *b, float *y,
                         int T, int Cin, int Cout) {
    if (T <= 0 || Cin <= 0 || Cout <= 0) return;
    for (int t = 0; t < T; t++)                // init with bias
        for (int o = 0; o < Cout; o++) y[t*Cout+o] = b ? b[o] : 0.f;
    // C[T,Cout] = A[T,Cin] * W^T[Cin,Cout] + C   (W stored row-major [Cout,Cin])
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, Cout, Cin, 1.f,
                x, Cin,          // lda
                W, Cin,          // ldb: leading dim of W is Cin
                1.f,             // accumulate onto bias-initialized C
                y, Cout);
}
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
// ---- fast safetensors loader: mmap the whole file once, slice tensors ----
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
struct MappedFile {
    const uint8_t * data = nullptr;
    size_t size = 0;
    int fd = -1;
    ~MappedFile() { if (fd >= 0) { munmap((void*)data, size); close(fd); } }
};
static MappedFile g_map;  // whole-file mapping, populated once in main

static std::vector<uint8_t> load_tensor_data(size_t off, size_t sz) {
    // slice from the already-mapped whole file; no per-tensor fopen/fread
    std::vector<uint8_t> data(sz);
    if (off + sz <= g_map.size) memcpy(data.data(), g_map.data + off, sz);
    return data;
}
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) { uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16; memcpy(&out[i], &u32, 4); }
    return out;
}

// ---------- helpers ----------
// (token_linear defined above, backed by Accelerate BLAS)
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
static void silu_f(float *x, size_t n) { for (size_t i=0;i<n;i++) x[i] = x[i]/(1.f+expf(-x[i])); }
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

// NCHW-style [C, T, B] tensor laid out as native[T*B*C + b*C + ... see below]
// We keep the transformer tensor as H[c][b][t] ("tensor") with index c + b*(C*B?) -- actually
// use plain token-per-frame: per frame the activations are token-major (S*C).
// For temporal blocks we need all frames: we store spatial activations as
// hmap[f][s*C+c].
using Tensor3 = std::vector<float>; // [F][S*C], token-major per frame

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors oracle_dir [F]\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);

    // mmap the whole file so tensor loading slices memory instead of
    // reopening the (multi-GB) file per tensor.
    g_map.fd = open(sf_path, O_RDONLY);
    if (g_map.fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(g_map.fd, &st);
    g_map.size = (size_t)st.st_size;
    g_map.data = (const uint8_t*)mmap(nullptr, g_map.size, PROT_READ, MAP_PRIVATE, g_map.fd, 0);
    if (g_map.data == MAP_FAILED) { perror("mmap"); return 1; }

    int F = 13;
    if (argc >= 4) F = atoi(argv[3]);

    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name == suffix) return &t;
        return nullptr;
    };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        if (!t) { fprintf(stderr, "missing %s\n", suffix.c_str()); exit(1); }
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(ds + t->offset, t->size);   // data_offsets are relative to data_start
        return bf16_to_f32(raw, (size_t)n);
    };
    auto read_oracle = [&](const char *name, int &oW,int &oH,int &oC,int &oF) -> std::vector<float> {
        FILE *f = fopen((std::string(argv[2]) + "/" + name).c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s\n", name); exit(1); }
        int32_t hdr[5]; fread(hdr, 4, 5, f);
        oW=hdr[0]; oH=hdr[1]; oC=hdr[2]; oF=hdr[3];
        size_t n = (size_t)oW*oH*oC*oF;
        std::vector<float> buf(n); fread(buf.data(), 4, n, f); fclose(f);
        return buf;
    };

    int xw,xh,xc,xf;
    auto x_whcn = read_oracle("mid_resnet0.bin", xw,xh,xc,xf);
    int W=xw, H=xh, C=xc, S=W*H;  // C=1280, S=4
    const int heads = C/64;
    const std::string pre = "mid_block.attentions.0";
    printf("x [%dx%dx%dx%d] C=%d heads=%d S=%d F=%d\n", W,H,C,xf, C, heads, S, xf);

    int ew,eh,ec,ef;
    auto ehs_whcn = read_oracle("ehs2.bin", ew,eh,ec,ef);
    const int Tk=eh, EHD=ew;
    printf("ehs2 [%d,%d,%d] Tk=%d EHD=%d\n", ew,eh,ec,Tk,EHD);
    // oracle F comes from inputs; use actual xf
    if (xf != F) { printf("note: oracle F=%d, requested %d -- using %d\n", xf, F, xf); F = xf; }

    // ---------- load conditioning per frame (token-major Tk*EHD) ----------
    std::vector<std::vector<float>> ehs(F, std::vector<float>((size_t)Tk*EHD));
    for (int n=0;n<F;n++)
      for (int t=0;t<Tk;t++)
        for (int d=0;d<EHD;d++)
            ehs[n][(size_t)t*EHD+d] = ehs_whcn[(size_t)n*Tk*EHD + (size_t)t*EHD + d];

    // ---------- input per frame token-major ----------
    std::vector<std::vector<float>> h(F, std::vector<float>((size_t)S*C));
    for (int n=0;n<F;n++)
      for (int s=0;s<S;s++){
        int w=s%W, row=s/W;
        for (int c=0;c<C;c++)
            h[n][(size_t)s*C+c] = x_whcn[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)row*W + w];
      }

    // ---------- transformer3d ----------
    auto gn_g=load_w(pre+".norm.weight"), gn_b=load_w(pre+".norm.bias");
    auto pin_w=load_w(pre+".proj_in.weight"), pin_b=load_w(pre+".proj_in.bias");
    auto pout_w=load_w(pre+".proj_out.weight"), pout_b=load_w(pre+".proj_out.bias");
    int n_layers=0; while (find(pre+".transformer_blocks."+std::to_string(n_layers)+".norm1.weight")) n_layers++;
    int n_temp=0;   while (find(pre+".temporal_transformer_blocks."+std::to_string(n_temp)+".norm_in.weight")) n_temp++;
    printf("layers=%d temporal=%d\n", n_layers, n_temp);

    // keep original x for final residual
    std::vector<std::vector<float>> x_in = h;

    // 1. group norm (32 groups over C, eps 1e-6) + affine, per frame
    {
        int n_groups=32, cpg=(C+n_groups-1)/n_groups;
        for (int n=0;n<F;n++){
            std::vector<float> gn((size_t)S*C);
            for (int gr=0;gr<n_groups;gr++){
                int st=gr*cpg, en=std::min(st+cpg,C), step=en-st;
                double sum=0; for (int c=st;c<en;c++) for (int s=0;s<S;s++) sum+=h[n][(size_t)s*C+c];
                double mean=sum/((double)S*step);
                double s2=0; for (int c=st;c<en;c++) for (int s=0;s<S;s++){double dd=h[n][(size_t)s*C+c]-mean; s2+=dd*dd;}
                double var=s2/((double)S*step); double sc2=1.0/sqrt(var+1e-6);
                for (int c=st;c<en;c++) for (int s=0;s<S;s++){size_t idx=(size_t)s*C+c; gn[idx]=(float)((h[n][idx]-mean)*sc2)*gn_g[c]+gn_b[c];}
            }
            h[n]=gn;
        }
    }
    // 2. proj_in (per frame, S tokens C->C)
    for (int n=0;n<F;n++){ std::vector<float> p((size_t)S*C); token_linear(h[n].data(), pin_w.data(), pin_b.data(), p.data(), S, C, C); h[n]=p; }

    // 3. layers
    int stride = n_layers >= 3 ? 2 : 1;
    int t_idx = 0;
    for (int l = 0; l < n_layers; l++) {
        std::string bp = pre+".transformer_blocks."+std::to_string(l);
        auto n1g=load_w(bp+".norm1.weight"), n1b=load_w(bp+".norm1.bias");
        auto q1=load_w(bp+".attn1.to_q.weight"), k1=load_w(bp+".attn1.to_k.weight"), v1=load_w(bp+".attn1.to_v.weight");
        auto o1=load_w(bp+".attn1.to_out.0.weight"), ob1=load_w(bp+".attn1.to_out.0.bias");
        auto n2g=load_w(bp+".norm2.weight"), n2b=load_w(bp+".norm2.bias");
        auto q2=load_w(bp+".attn2.to_q.weight"), k2=load_w(bp+".attn2.to_k.weight"), v2=load_w(bp+".attn2.to_v.weight");
        auto o2=load_w(bp+".attn2.to_out.0.weight"), ob2=load_w(bp+".attn2.to_out.0.bias");
        auto n3g=load_w(bp+".norm3.weight"), n3b=load_w(bp+".norm3.bias");
        auto f1w=load_w(bp+".ff.net.0.proj.weight"), f1b=load_w(bp+".ff.net.0.proj.bias");
        auto f2w=load_w(bp+".ff.net.2.weight"), f2b=load_w(bp+".ff.net.2.bias");
        int g=(int)f1w.size()/(2*C);   // gate dim = 5120

        std::vector<std::vector<float>> cur = h;   // residual stream
        for (int n=0;n<F;n++){
            // LN1
            std::vector<float> ln1((size_t)S*C); layer_norm_tokens(h[n].data(), ln1.data(), n1g.data(), n1b.data(), S, C);
            // self-attn attn1
            std::vector<float> Q((size_t)S*C),K((size_t)S*C),V((size_t)S*C),A((size_t)S*C),Ao((size_t)S*C);
            token_linear(ln1.data(), q1.data(), nullptr, Q.data(), S, C, C);
            token_linear(ln1.data(), k1.data(), nullptr, K.data(), S, C, C);
            token_linear(ln1.data(), v1.data(), nullptr, V.data(), S, C, C);
            mha_tokens(Q.data(), K.data(), V.data(), A.data(), S, S, C, heads);
            token_linear(A.data(), o1.data(), ob1.data(), Ao.data(), S, C, C);
            for (size_t i=0;i<cur[n].size();i++) cur[n][i]+=Ao[i];
            // LN2
            std::vector<float> ln2((size_t)S*C); layer_norm_tokens(cur[n].data(), ln2.data(), n2g.data(), n2b.data(), S, C);
            // cross-attn attn2 on ehs
            std::vector<float> Q2((size_t)S*C),K2((size_t)Tk*C),V2((size_t)Tk*C),A2((size_t)S*C),Ao2((size_t)S*C);
            token_linear(ln2.data(), q2.data(), nullptr, Q2.data(), S, C, C);
            token_linear(ehs[n].data(), k2.data(), nullptr, K2.data(), Tk, EHD, C);
            token_linear(ehs[n].data(), v2.data(), nullptr, V2.data(), Tk, EHD, C);
            mha_tokens(Q2.data(), K2.data(), V2.data(), A2.data(), S, Tk, C, heads);
            token_linear(A2.data(), o2.data(), ob2.data(), Ao2.data(), S, C, C);
            for (size_t i=0;i<cur[n].size();i++) cur[n][i]+=Ao2[i];
            // LN3 + GEGLU FFN
            std::vector<float> ln3((size_t)S*C); layer_norm_tokens(cur[n].data(), ln3.data(), n3g.data(), n3b.data(), S, C);
            std::vector<float> p1((size_t)S*2*g);
            token_linear(ln3.data(), f1w.data(), f1b.data(), p1.data(), S, C, 2*g);
            std::vector<float> glu((size_t)S*g);
            for (int s=0;s<S;s++) for (int c=0;c<g;c++) {
                float val=p1[(size_t)s*2*g+c], gate=p1[(size_t)s*2*g+g+c];
                glu[(size_t)s*g+c] = val * gelu_erf(gate);
            }
            std::vector<float> ff((size_t)S*C);
            token_linear(glu.data(), f2w.data(), f2b.data(), ff.data(), S, g, C);
            for (size_t i=0;i<cur[n].size();i++) cur[n][i]+=ff[i];
        }
        h = cur;

        // temporal cross-frame block after every stride-th
        if ((l+1) % stride == 0 && t_idx < n_temp) {
            std::string tp = pre+".temporal_transformer_blocks."+std::to_string(t_idx++);
            // cross_frame_block(m, x (C,S,F) -> permute (C,F,S) so attention across F),
            // then permute back and the CALLER (transformer3d) adds it as residual.
            // reference: h = h + cross_frame_block(h, tpre);  (result of t3d layer)
            auto tn_in_g=load_w(tp+".norm_in.weight"), tn_in_b=load_w(tp+".norm_in.bias");
            // ff_in GEGLU
            auto tff1w=load_w(tp+".ff_in.net.0.proj.weight"), tff1b=load_w(tp+".ff_in.net.0.proj.bias");
            auto tff2w=load_w(tp+".ff_in.net.2.weight"), tff2b=load_w(tp+".ff_in.net.2.bias");
            auto tn1g=load_w(tp+".norm1.weight"), tn1b=load_w(tp+".norm1.bias");
            auto tq1=load_w(tp+".attn1.to_q.weight"), tk1=load_w(tp+".attn1.to_k.weight"), tv1=load_w(tp+".attn1.to_v.weight");
            auto to1=load_w(tp+".attn1.to_out.0.weight"), tob1=load_w(tp+".attn1.to_out.0.bias");
            auto tn3g=load_w(tp+".norm3.weight"), tn3b=load_w(tp+".norm3.bias");
            auto tff1=load_w(tp+".ff.net.0.proj.weight"), tff1b2=load_w(tp+".ff.net.0.proj.bias");
            auto tff2=load_w(tp+".ff.net.2.weight"), tff2b2=load_w(tp+".ff.net.2.bias");
            int tg=(int)tff1w.size()/(2*C);
            // reference cross_frame_block on CURRENT h (C,S,F), permuted to (C,F,S):
            //   x2[c][f][s]  = h[f][s*C+c]  (i.e. tokens = frames F, batch = S)
            // res = x2
            // h = LN(norm_in)(x2); h = GEGLU(ff_in)(h); h += res
            // h = LN(norm1)(h); h += attn1 (self over F, Tq=F, C);
            // h = GEGLU(ff)(LN(norm3)(h))
            // out = permute back
            // We'll operate on tensor x2[F tokens][S * C]? No -- attention is over
            // F frames for each spatial token s, batched over S.
            // Represent x2 as x2mat[s][f*C + c] for s in S: token=f, C-dim.
            // Build from h (which is h[f][s*C+c]):
            std::vector<std::vector<float>> x2(S, std::vector<float>((size_t)F*C));
            for (int f=0;f<F;f++) for (int s=0;s<S;s++) for (int c=0;c<C;c++)
                x2[s][(size_t)f*C+c] = h[f][(size_t)s*C+c];
            // res2 = x2 (is_res residual after ff_in)
            std::vector<std::vector<float>> res2 = x2;
            // LN(norm_in) over token-dim C, per (s) token
            for (int s=0;s<S;s++){
                std::vector<float> ln((size_t)F*C);
                layer_norm_tokens(x2[s].data(), ln.data(), tn_in_g.data(), tn_in_b.data(), F, C);
                // GEGLU ff_in
                std::vector<float> p((size_t)F*2*tg);
                token_linear(ln.data(), tff1w.data(), tff1b.data(), p.data(), F, C, 2*tg);
                std::vector<float> glu((size_t)F*tg);
                for (int t=0;t<F;t++) for (int c=0;c<tg;c++){
                    float val=p[(size_t)t*2*tg+c], ga=p[(size_t)t*2*tg+tg+c];
                    glu[(size_t)t*tg+c]=val*gelu_erf(ga);
                }
                std::vector<float> ff((size_t)F*C);
                token_linear(glu.data(), tff2w.data(), tff2b.data(), ff.data(), F, tg, C);
                for (size_t i=0;i<x2[s].size();i++) x2[s][i]+=ff[i];
            }
            // LN(norm1) + self-attn over F (Tq=F), batched per s
            for (int s=0;s<S;s++){
                std::vector<float> ln((size_t)F*C);
                layer_norm_tokens(x2[s].data(), ln.data(), tn1g.data(), tn1b.data(), F, C);
                std::vector<float> Q((size_t)F*C),K((size_t)F*C),V((size_t)F*C),Ao((size_t)F*C);
                token_linear(ln.data(), tq1.data(), nullptr, Q.data(), F, C, C);
                token_linear(ln.data(), tk1.data(), nullptr, K.data(), F, C, C);
                token_linear(ln.data(), tv1.data(), nullptr, V.data(), F, C, C);
                mha_tokens(Q.data(), K.data(), V.data(), Ao.data(), F, F, C, C/64);
                std::vector<float> Ao2((size_t)F*C);
                token_linear(Ao.data(), to1.data(), tob1.data(), Ao2.data(), F, C, C);
                for (size_t i=0;i<x2[s].size();i++) x2[s][i]+=Ao2[i];
            }
            // final FF (no residual): h = GEGLU(ff)(LN(norm3))
            for (int s=0;s<S;s++){
                std::vector<float> ln((size_t)F*C);
                layer_norm_tokens(x2[s].data(), ln.data(), tn3g.data(), tn3b.data(), F, C);
                std::vector<float> p((size_t)F*2*tg);
                token_linear(ln.data(), tff1w.data(), tff1b2.data(), p.data(), F, C, 2*tg);
                std::vector<float> glu((size_t)F*tg);
                for (int t=0;t<F;t++) for (int c=0;c<tg;c++){
                    float val=p[(size_t)t*2*tg+c], ga=p[(size_t)t*2*tg+tg+c];
                    glu[(size_t)t*tg+c]=val*gelu_erf(ga);
                }
                std::vector<float> ff((size_t)F*C);
                token_linear(glu.data(), tff2w.data(), tff2b2.data(), ff.data(), F, tg, C);
                x2[s]=ff;
            }
            // permute back into h[f][s*C+c] and ADD to h (residual in t3d)
            for (int f=0;f<F;f++) for (int s=0;s<S;s++) for (int c=0;c<C;c++)
                h[f][(size_t)s*C+c] += x2[s][(size_t)f*C+c];
            // NOTE: also reference adds res2 residual inside cross_frame_block ... handled
            // above via x2 init (we made res2 but reference does h+=res AFTER ff_in; done)
        }
    }
    // 4. proj_out per frame
    for (int n=0;n<F;n++){ std::vector<float> po((size_t)S*C); token_linear(h[n].data(), pout_w.data(), pout_b.data(), po.data(), S, C, C); h[n]=po; }
    // 5. residual: transformer3d returns h + x
    for (int n=0;n<F;n++) for (size_t i=0;i<h[n].size();i++) h[n][i] += x_in[n][i];

    // ---------- compare against mid_attn ----------
    int mw,mh,mc,mf;
    auto ref = read_oracle("mid_attn.bin", mw,mh,mc,mf);
    printf("oracle mid_attn [%dx%dx%dx%d]\n", mw,mh,mc,mf);
    // mine per frame token-major; oracle WHCN: idx = w+h*mw + c*mw*mh + f*mw*mh*mc
    double max_abs=0, sum_abs=0; size_t cnt=(size_t)S*C*F;
    for (int n=0;n<F;n++)
      for (int s=0;s<S;s++){
        int w=s%W, row=s/W;
        for (int c=0;c<C;c++){
            size_t oidx=(size_t)n*mw*mh*mc + (size_t)c*mw*mh + (size_t)row*mw + w;
            double dd=fabs((double)h[n][(size_t)s*C+c]-(double)ref[oidx]);
            if (dd>max_abs)max_abs=dd; sum_abs+=dd;
        }
      }
    double cos=0,na=0,nb=0;
    for (int n=0;n<F;n++) for (int s=0;s<S;s++) for (int c=0;c<C;c++){
        int w=s%W, row=s/W; size_t oidx=(size_t)n*mw*mh*mc+(size_t)c*mw*mh+(size_t)row*mw+w;
        cos+=(double)h[n][(size_t)s*C+c]*ref[oidx];
        na+=(double)h[n][(size_t)s*C+c]*h[n][(size_t)s*C+c];
        nb+=(double)ref[oidx]*ref[oidx];
    }
    cos/=sqrt(na*nb);
    printf("mine[0]=%.4f ora[0]=%.4f\n", h[0][0], ref[0]);
    printf("transformer3d mid %d layers + %d temporal (%d/20 heads): max_abs=%.6f mean=%.6f cosine=%.6f\n",
           n_layers, n_temp, heads, max_abs, sum_abs/cnt, cos);
    printf("%s\n", max_abs < 1.0 && cos > 0.999 ? "VALIDATION PASS — transformer3d matches ggml oracle (16-bit weight rounding)"
                   : "VALIDATION FAIL");
    return (max_abs < 1.0 && cos > 0.999) ? 0 : 1;
}
