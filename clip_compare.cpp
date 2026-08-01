// Compare our dispatch-runner CLIP encoder against the ggml oracle.
// Usage: clip_compare <te.safetensors> <oracle.bin> <token_ids.txt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *data_start);
static std::vector<uint8_t> load_tensor_data(const char *sf, size_t ds, size_t off, size_t sz);
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n);
static void layer_norm(const float *x, float *y, const float *g, const float *bb, int D, int rows, float eps);
static void quick_gelu(float *x, int n);
static void softmax_rows(float *x, int T);

// Linear: out[t,o] = sum_k x[t,k] * W[o,k]  (PyTorch x @ W^T, W = [dout, din])
static void gemm_bias(const float *x, const float *W, float *out, const float *b,
                      int t, int din, int dout) {
    for (int ti = 0; ti < t; ti++)
        for (int o = 0; o < dout; o++) {
            float s = 0;
            for (int k = 0; k < din; k++) s += x[ti*din + k] * W[o*din + k];
            out[ti*dout + o] = s + b[o % dout];
        }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s te.safetensors oracle.bin token_ids.txt\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);

    std::vector<int> ids;
    { std::ifstream f(argv[3]); int v; while (f >> v) ids.push_back(v); }
    const int T = ids.size(), D = 768, H = 12, hd = D/H, Dint = 3072;
    printf("T=%d tokens loaded\n", T);

    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name.find(suffix) != std::string::npos) return &t;
        return nullptr;
    };
    // GGUF and safetensors weights are IDENTICAL layout; gemm_bias uses W[o,k] row-major [dout,din].
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(sf_path, ds, t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };
    auto load_b = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        int n = (int)t->shape[0];
        auto raw = load_tensor_data(sf_path, ds, t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };
    auto load_emb = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(sf_path, ds, t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };

    auto emb = load_emb("embeddings.token_embedding.weight");
    auto pos = load_emb("embeddings.position_embedding.weight");
    std::vector<float> x(T*D);
    for (int t = 0; t < T; t++) {
        int tid = ids[t];
        for (int d = 0; d < D; d++) x[t*D+d] = emb[tid*D+d] + pos[t*D+d];
    }
    printf("my-post-emb[40..48]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
           x[40],x[41],x[42],x[43],x[44],x[45],x[46],x[47],x[48]);

    std::vector<float> h(T*D), attn_heads(T*D), out(T*D), ln(T*D), ff(T*D);
    std::vector<float> gamma(D, 1.0f), beta(D, 0.0f);
    std::vector<float> *penult_ref = nullptr;
    for (int l = 0; l < 12; l++) {
        if (l == 11) penult_ref = new std::vector<float>(x.begin(), x.end());
        std::string pre = "text_model.encoder.layers." + std::to_string(l) + ".";
        auto Wq = load_w(pre+"self_attn.q_proj.weight"), Wk = load_w(pre+"self_attn.k_proj.weight");
        auto Wv = load_w(pre+"self_attn.v_proj.weight"), Wo = load_w(pre+"self_attn.out_proj.weight");
        auto bq = load_b(pre+"self_attn.q_proj.bias"), bk = load_b(pre+"self_attn.k_proj.bias");
        auto bv = load_b(pre+"self_attn.v_proj.bias"), bo = load_b(pre+"self_attn.out_proj.bias");
        auto W1 = load_w(pre+"mlp.fc1.weight"), b1 = load_b(pre+"mlp.fc1.bias");
        auto W2 = load_w(pre+"mlp.fc2.weight"), b2 = load_b(pre+"mlp.fc2.bias");

        // LayerNorm affine weights are per-layer learned (gamma, beta)
        auto ln1g = load_w(pre+"layer_norm1.weight"), ln1b = load_b(pre+"layer_norm1.bias");
        auto ln2g = load_w(pre+"layer_norm2.weight"), ln2b = load_b(pre+"layer_norm2.bias");
        layer_norm(x.data(), h.data(), ln1g.data(), ln1b.data(), D, T, 1e-5f);
        std::vector<float> Q(T*D), K(T*D), V(T*D);
        if (l == 0) {
            printf("my LN1(t0,d0..7):");
            for (int k = 0; k < 8; k++) printf(" %.4f", h[k]);
            printf("\n");
        }
        gemm_bias(h.data(), Wq.data(), Q.data(), bq.data(), T, D, D);
        if (l == 0) {
            printf("my Q(t0,d0..7, x@W^T):");
            for (int k = 0; k < 8; k++) printf(" %.4f", Q[k]/sqrtf((float)hd)); // divide by sqrt like oracle
            printf("\n");
        }
        gemm_bias(h.data(), Wk.data(), K.data(), bk.data(), T, D, D);
        gemm_bias(h.data(), Wv.data(), V.data(), bv.data(), T, D, D);
        for (int hd_ = 0; hd_ < H; hd_++) {
            std::vector<float> scores(T*T);
            for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) {
                float s = 0;
                for (int d = 0; d < hd; d++) s += Q[i*D+hd_*hd+d] * K[j*D+hd_*hd+d];
                scores[i*T+j] = (j > i) ? -1e30f : s / sqrtf((float)hd);
            }
            softmax_rows(scores.data(), T);
            for (int i = 0; i < T; i++) for (int d = 0; d < hd; d++) {
                float s = 0;
                for (int j = 0; j < T; j++) s += scores[i*T+j] * V[j*D+hd_*hd+d];
                attn_heads[i*D+hd_*hd+d] = s;
            }
        }
        // out_proj: uses a separate out buffer to avoid aliasing attn_heads
        gemm_bias(attn_heads.data(), Wo.data(), out.data(), bo.data(), T, D, D);
        for (int i = 0; i < T*D; i++) out[i] = x[i] + out[i];  // residual

        layer_norm(out.data(), ln.data(), ln2g.data(), ln2b.data(), D, T, 1e-5f);
        std::vector<float> hid(T*Dint), ff2(T*D);
        gemm_bias(ln.data(), W1.data(), hid.data(), b1.data(), T, D, Dint);
        quick_gelu(hid.data(), T*Dint);
        gemm_bias(hid.data(), W2.data(), ff2.data(), b2.data(), T, Dint, D);
        for (int i = 0; i < T*D; i++) x[i] = out[i] + ff2[i];  // residual
        if (l == 0 || l == 11)
            printf("  my layer %d: x[0..3]=%.4f %.4f %.4f %.4f\n", l, x[0],x[1],x[2],x[3]);
    }

    const std::vector<float> &mine = penult_ref ? *penult_ref : x;
    printf("mine(penult)[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           mine[0],mine[1],mine[2],mine[3],mine[4],mine[5],mine[6],mine[7]);

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "no oracle\n"); return 1; }
    std::vector<float> ref((size_t)T*D);
    fread(ref.data(), 4, ref.size(), f);
    fclose(f);
    printf("oracle[0..7]     =%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);

    // Oracle ggml tensor [d, n]: dim d fastest, ref index = d + t*D
    std::vector<float> oref((size_t)T*D);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            oref[(size_t)t*D + d] = ref[(size_t)d + (size_t)t*D];

    double max_abs = 0, sum_abs = 0;
    for (size_t i = 0; i < mine.size(); i++) {
        double dd = fabs((double)mine[i] - (double)oref[i]);
        if (dd > max_abs) max_abs = dd;
        sum_abs += dd;
    }
    printf("penultimate hidden 77x768 vs ggml oracle:\n");
    printf("  max_abs_diff=%.6f mean_abs_diff=%.6f\n", max_abs, sum_abs/mine.size());
    double cos_sim = 0, na = 0, nb = 0;
    for (size_t i = 0; i < mine.size(); i++) { cos_sim += (double)mine[i]*oref[i]; na += (double)mine[i]*mine[i]; nb += (double)oref[i]*oref[i]; }
    cos_sim /= sqrt(na*nb);
    printf("  cosine_similarity=%.6f\n", cos_sim);
    printf("%s\n", max_abs < 0.1 && cos_sim > 0.999 ? "VALIDATION PASS — encoder matches ggml CLIP" : "VALIDATION FAIL");
    return (max_abs < 0.1 && cos_sim > 0.999) ? 0 : 1;
}

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
static void layer_norm(const float *x, float *y, const float *g, const float *bb, int D, int rows, float eps) {
    for (int r = 0; r < rows; r++) { const float *xr=x+r*D; float *yr=y+r*D; float mn=0; for (int d=0;d<D;d++) mn+=xr[d]; mn/=D; float va=0; for (int d=0;d<D;d++){float t=xr[d]-mn;va+=t*t;} va/=D; float inv=1.0f/sqrtf(va+eps); for(int d=0;d<D;d++) yr[d]=(xr[d]-mn)*inv*g[d]+bb[d]; }
}
static void quick_gelu(float *x, int n) { for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-1.702f*x[i])); }
static void softmax_rows(float *x, int T) { for (int i = 0; i < T; i++) { float *r=x+i*T; float mx=-1e30f; for (int j=0;j<T;j++) if(r[j]>mx)mx=r[j]; float s=0; for(int j=0;j<T;j++){r[j]=expf(r[j]-mx);s+=r[j];} for(int j=0;j<T;j++)r[j]/=s; } }