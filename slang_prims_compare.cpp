// Slang primitives compare: drive the EXPORTED Slang layernorm + geglu kernels
// against the ggml temporal oracle taps (temporal2.bin: tap1=norm_in_out,
// tap2=ff_in_out) to validate the actual Slang kernels, not hand-written ones.
// Usage: slang_prims_compare <unet.safetensors> <temporal2.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>

// --- minimal safetensors reader ---
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
            if (field == "dtype") { if (json[i] == '"') { size_t vs = ++i; while (i < json.size() && json[i] != '"') i++; t.dtype = json.substr(vs, i - vs); i++; } }
            else if (field == "shape") { if (json[i] == '[') { i++; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; t.shape.push_back(strtol(&json[i], &end, 10)); i = end - &json[0]; } } i++; } }
            else if (field == "data_offsets") { if (json[i] == '[') { i++; int idx=0; uint64_t vals[2]={0,0}; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; vals[idx++]=strtoull(&json[i], &end, 10); i=end-&json[0]; } } i++; t.offset=vals[0]; t.size=vals[1]-vals[0]; } }
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

#include "shaders/cpp/layernorm_cpu.cpp"

static void cmp(const char *nm, const std::vector<float>&a, const std::vector<float>&b){
    double m=0,s=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; s+=d;}
    double c=0,na=0,nb=0; for(size_t i=0;i<n;i++){c+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} c/=sqrt(na*nb);
    printf("%s: max=%.6f mean=%.6f cos=%.6f\n",nm,m,s/n,c);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors temporal2.bin\n", argv[0]); return 1; }
    size_t ds = 0; auto tensors = read_sf(argv[1], &ds);
    auto find = [&](std::string suffix) -> const SfTensor* { for (auto &t : tensors) if (t.name == suffix) return &t; return nullptr; };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix); if (!t) { fprintf(stderr, "missing %s\n", suffix.c_str()); exit(1); }
        int n = 1; for (auto s : t->shape) n *= (int)s;
        return bf16_to_f32(load_tensor_data(argv[1], ds, t->offset, t->size), (size_t)n);
    };

    FILE *f = fopen(argv[2], "rb");
    int32_t hdr[5]; fread(hdr,4,5,f);
    int C=hdr[0], S=hdr[1], F=hdr[2], n_head=hdr[3], N=hdr[4];
    size_t n=(size_t)C*S*F;
    std::vector<float> xd(n); fread(xd.data(),4,n,f);
    std::vector<std::vector<float>> taps(N, std::vector<float>(n));
    for (int i=0;i<N;i++) fread(taps[i].data(),4,n,f);
    fclose(f);
    // taps: 0=xp, 1=norm_in_out, 2=ff_in_out, 3=ff_in+res, 4=norm1_out, 5=attn1_out, 6=+attn1, 7=ff_final, 8=out
    // xp is (C,F,S) ne0=C: xp[c+C*f+C*F*s]; our x2 token-major x2[s][f*C+c].
    // Validate Slang layernorm on xp input (token=frame f, batch s), T=F, D=C.
    const std::string pre = "mid_block.attentions.0.temporal_transformer_blocks.0";
    auto tn_in_g=load_w(pre+".norm_in.weight"), tn_in_b=load_w(pre+".norm_in.bias");
    auto tff1w=load_w(pre+".ff_in.net.0.proj.weight"), tff1b=load_w(pre+".ff_in.net.0.proj.bias");
    auto tff2w=load_w(pre+".ff_in.net.2.weight"), tff2b=load_w(pre+".ff_in.net.2.bias");
    int tg=(int)tff1w.size()/(2*C);   // g dim
    int Cout=C;

    // Reorganize oracle xp (C,F,S) -> token-major per-batch s: X[s][f*C+c]
    std::vector<float> X((size_t)F*C), Yln((size_t)F*C), ref_norm((size_t)F*C);
    for (int s=0;s<S;s++){
        for (int f=0;f<F;f++) for (int c=0;c<C;c++)
            X[(size_t)f*C+c] = taps[0][(size_t)c + (size_t)C*f + (size_t)(C*F)*s];
        // Slang layernorm on this batch's (F, C)
        GlobalParams_0 lp;
        lp.X_in_0.data = X.data();  lp.X_in_0.count = X.size();
        lp.G_in_0.data = tn_in_g.data(); lp.G_in_0.count = tn_in_g.size();
        lp.B_in_0.data = tn_in_b.data(); lp.B_in_0.count = tn_in_b.size();
        lp.Y_out_0.data = Yln.data(); lp.Y_out_0.count = Yln.size();
        lp.T_0 = F; lp.D_0 = C; lp.EPS_0 = 1e-5f;
        ComputeVaryingInput vi;
        vi.startGroupID.x=vi.startGroupID.y=vi.startGroupID.z=0;
        vi.endGroupID.x=(F+7)/8; vi.endGroupID.y=(C+7)/8; vi.endGroupID.z=1;
        layernorm_kern(&vi, nullptr, &lp);
        // oracle tap1 (norm_in_out) is (C,F,S): ref[..] = taps[1][c+C*f+C*F*s]
        for (int f=0;f<F;f++) for (int c=0;c<C;c++)
            ref_norm[(size_t)f*C+c] = taps[1][(size_t)c + (size_t)C*f + (size_t)(C*F)*s];
        // rebuild X from oracle xp for next... just compare per batch
        std::vector<float> mine((size_t)F*C);
        for (int f=0;f<F;f++) for (int c=0;c<C;c++){
            // mine Yln token-major f*C+c -> place into (C,F,S) for cross-batch compare
            mine[(size_t)c + (size_t)C*f] = Yln[(size_t)f*C+c];
        }
        std::vector<float> ref((size_t)F*C);
        for (int f=0;f<F;f++) for (int c=0;c<C;c++)
            ref[(size_t)c + (size_t)C*f] = taps[1][(size_t)c + (size_t)C*f + (size_t)(C*F)*s];
        if (s==0) cmp("Slang layernorm vs norm_in_out", mine, ref);
    }
    return 0;
}
