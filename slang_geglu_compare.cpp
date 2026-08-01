// Slang GEGLU compare: drive the EXPORTED Slang geglu_kern against the ggml
// temporal ff_in_out tap (temporal2.bin tap2).
// Usage: slang_geglu_compare <unet.safetensors> <temporal2.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *data_start) {
    FILE *f = fopen(path, "rb"); uint8_t buf[8]; fread(buf, 1, 8, f);
    uint64_t hdr_len = 0; for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)buf[j] << (j * 8);
    std::string json((size_t)hdr_len, 0); fread(&json[0], 1, hdr_len, f); *data_start = 8 + hdr_len; fclose(f);
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
#include "shaders/cpp/geglu_cpu.cpp"
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
    const std::string pre = "mid_block.attentions.0.temporal_transformer_blocks.0";
    auto tff1w=load_w(pre+".ff_in.net.0.proj.weight"), tff1b=load_w(pre+".ff_in.net.0.proj.bias");
    auto tff2w=load_w(pre+".ff_in.net.2.weight"), tff2b=load_w(pre+".ff_in.net.2.bias");
    int tg=(int)tff1w.size()/(2*C);
    std::vector<float> X((size_t)F*C), GLU((size_t)F*tg), Yg((size_t)F*C);
    GlobalParams_0 gg;
    gg.W1_in_0.data=tff1w.data(); gg.W1_in_0.count=tff1w.size();
    gg.B1_in_0.data=tff1b.data(); gg.B1_in_0.count=tff1b.size();
    gg.W2_in_0.data=tff2w.data(); gg.W2_in_0.count=tff2w.size();
    gg.B2_in_0.data=tff2b.data(); gg.B2_in_0.count=tff2b.size();
    gg.GLU_out_0.data=GLU.data(); gg.GLU_out_0.count=GLU.size();
    gg.GLU2_in_0.data=GLU.data(); gg.GLU2_in_0.count=GLU.size();
    gg.Y_out_0.data=Yg.data(); gg.Y_out_0.count=Yg.size();
    gg.T_0=F; gg.CIN_0=C; gg.G_0=tg; gg.T2_0=F; gg.G2_0=tg; gg.COUT2_0=C;
    ComputeVaryingInput vi;
    vi.startGroupID.x=vi.startGroupID.y=vi.startGroupID.z=0;
    vi.endGroupID.x=(F+7)/8; vi.endGroupID.y=(tg+7)/8; vi.endGroupID.z=1;
    ComputeVaryingInput vi2;
    vi2.startGroupID.x=vi2.startGroupID.y=vi2.startGroupID.z=0;
    vi2.endGroupID.x=(F+7)/8; vi2.endGroupID.y=(C+7)/8; vi2.endGroupID.z=1;
    for (int s=0;s<S;s++){
        for (int f=0;f<F;f++) for (int c=0;c<C;c++)
            X[(size_t)f*C+c] = taps[1][(size_t)c + (size_t)C*f + (size_t)(C*F)*s];
        gg.X_in_0.data=X.data(); gg.X_in_0.count=X.size();
        geglu_proj(&vi, nullptr, &gg);
        geglu_ff(&vi2, nullptr, &gg);
        std::vector<float> mine((size_t)F*C), ref((size_t)F*C);
        for (int f=0;f<F;f++) for (int c=0;c<C;c++){
            mine[(size_t)c + (size_t)C*f] = Yg[(size_t)f*C+c];
            ref[(size_t)c + (size_t)C*f] = taps[2][(size_t)c + (size_t)C*f + (size_t)(C*F)*s];
        }
        if (s==0) cmp("Slang GEGLU(ff_in) vs ff_in_out", mine, ref);
    }
    return 0;
}
