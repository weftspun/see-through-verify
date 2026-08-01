// Slang Conv2d compare: drive the EXPORTED Slang conv2d_kern against the ggml
// conv_in oracle tap, validating the actual Slang kernel with safetensors weights.
// Usage: slang_conv_compare <unet.safetensors> <conv_in.bin>
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
#include "shaders/cpp/conv2d_cpu.cpp"
static void cmp(const char *nm, const std::vector<float>&a, const std::vector<float>&b){
    double m=0,s=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; s+=d;}
    double c=0,na=0,nb=0; for(size_t i=0;i<n;i++){c+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} c/=sqrt(na*nb);
    printf("%s: max=%.6f mean=%.6f cos=%.6f\n",nm,m,s/n,c);
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors conv_in.bin\n", argv[0]); return 1; }
    size_t ds = 0; auto tensors = read_sf(argv[1], &ds);
    auto find = [&](std::string suffix) -> const SfTensor* { for (auto &t : tensors) if (t.name == suffix) return &t; return nullptr; };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix); if (!t) { fprintf(stderr, "missing %s\n", suffix.c_str()); exit(1); }
        int n = 1; for (auto s : t->shape) n *= (int)s;
        return bf16_to_f32(load_tensor_data(argv[1], ds, t->offset, t->size), (size_t)n);
    };
    auto *wt=find("conv_in.weight"), *bt=find("conv_in.bias");
    int OC=wt->shape[0], IC=wt->shape[1], KH=wt->shape[2], KW=wt->shape[3];
    auto Wt=load_w("conv_in.weight"), Bt=load_w("conv_in.bias");

    FILE *f = fopen(argv[2], "rb");
    int32_t hdr[5]; fread(hdr,4,5,f);
    int W=hdr[0], H=hdr[1], OCt=hdr[2], F=hdr[3], ICt=hdr[4];
    size_t on=(size_t)W*H*OCt*F;
    std::vector<float> oracle_raw(on); fread(oracle_raw.data(),4,on,f);
    std::vector<float> x((size_t)W*H*ICt*F); fread(x.data(),4,x.size(),f);   // oracle's exact input
    fclose(f);
    printf("conv_in tap [%dx%dx%dx%d] w[%d,%d,%d,%d] (feed oracle input)\n", W,H,OCt,F, OC,IC,KH,KW);

    std::vector<float> out(on, 0.f);
    GlobalParams_0 p;
    p.X_in_0.data=x.data(); p.X_in_0.count=x.size();
    p.W_in_0.data=Wt.data(); p.W_in_0.count=Wt.size();
    p.B_in_0.data=Bt.data(); p.B_in_0.count=Bt.size();
    p.Y_out_0.data=out.data(); p.Y_out_0.count=out.size();
    p.H_0=H; p.W_0=W; p.IC_0=IC; p.OC_0=OC; p.KH_0=KH; p.KW_0=KW; p.STRIDE_0=1; p.PAD_0=1;
    ComputeVaryingInput vi;
    vi.startGroupID.x=vi.startGroupID.y=vi.startGroupID.z=0;
    vi.endGroupID.x=(OC+7)/8; vi.endGroupID.y=(H*W+7)/8; vi.endGroupID.z=F;
    conv2d_kern(&vi, nullptr, &p);

    // oracle dumped RAW conv (ggml conv_2d, no bias); add bias to ref
    std::vector<float> ref(on);
    for (size_t i=0;i<on;i++) ref[i]=oracle_raw[i] + Bt[(i/H/W)%OC];
    cmp("Slang conv2d vs ggml conv_in (raw+bias)", out, ref);

    // Internal known-good direct conv (same math as validated conv2d_compare) to
    // confirm the oracle-dump input/ref is the inconsistent side.
    std::vector<float> dref(on, 0.f);
    for (uint32_t n=0;n<(uint32_t)F;n++)
      for (int oc=0;oc<OC;oc++)
        for (int oh=0;oh<H;oh++)
          for (int ow=0;ow<W;ow++) {
            float s=Bt[oc];
            for (int c=0;c<IC;c++)
              for (int kh=0;kh<KH;kh++)
                for (int kw=0;kw<KW;kw++){
                  int ih=oh-1+kh, iw=ow-1+kw;
                  if(ih<0||ih>=H||iw<0||iw>=W) continue;
                  float wv=Wt[((oc*IC+c)*KH+kh)*KW+kw];
                  float xv=x[(size_t)n*W*H*IC+(size_t)c*W*H+(size_t)ih*W+iw];
                  s+=wv*xv;
                }
            dref[(size_t)n*OC*H*W+(size_t)oc*H*W+(size_t)oh*W+ow]=s;
          }
    cmp("Slang conv2d vs direct-f32 (self)", out, dref);
    cmp("oracle vs direct-f32 (sanity)", ref, dref);
    return 0;
}
