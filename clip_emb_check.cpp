// Compare one row of the token embedding between GGUF (ggml oracle) and safetensors.
// The CLIP models must share the SAME embedding values for identical token ids.
// Usage: clip_emb_check <te1.gguf> <text_encoder.safetensors> <token_id>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

// safetensors reader
struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *ds);
static std::vector<uint8_t> load_td(const char *sf, size_t ds, size_t off, size_t sz);
static std::vector<float> bf16_f32(const std::vector<uint8_t> &raw, size_t n);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s te1.gguf safetensors token_id\n", argv[0]); return 1; }
    int tid = atoi(argv[3]);

    // --- load embedding from safetensors ---
    size_t ds = 0;
    auto tensors = read_sf(argv[2], &ds);
    auto find = [&](std::string suf) -> const SfTensor* {
        for (auto &t : tensors) if (t.name.find(suf) != std::string::npos) return &t;
        return nullptr;
    };
    auto *emb = find("embeddings.token_embedding.weight");
    printf("safetensors token_embedding: %ld elems, shape=%lld,%lld\n",
           emb ? (long)emb->size : 0L, (long long)(emb?emb->shape[0]:0), (long long)(emb?emb->shape[1]:0));
    if (!emb) return 1;
    auto raw = load_td(argv[2], ds, emb->offset, emb->size);
    auto w = bf16_f32(raw, (size_t)emb->shape[0]*emb->shape[1]);
    int D = (int)emb->shape[1];
    printf("safetensors emb[%d][0..3] = %.6f %.6f %.6f %.6f\n", tid, w[tid*D+0], w[tid*D+1], w[tid*D+2], w[tid*D+3]);

    // --- load same from GGUF via projector program on CPU ---
    // (we won't relink; just note the safetensors row for manual comparison)

    // Dump a known reference from the oracle: print embedding range to help diagnose
    double mn=1e9, mx=-1e9, mean=0;
    for (auto v : w) { if (v<mn)mn=v; if(v>mx)mx=v; mean+=v; }
    printf("safetensors embedding range [%.4f, %.4f] mean=%.6f\n", mn, mx, mean/ (double)w.size());
    return 0;
}

static std::vector<SfTensor> read_sf(const char *path, size_t *ds) {
    FILE *f=fopen(path,"rb"); uint8_t b[8]; fread(b,1,8,f);
    uint64_t hl=0; for(int j=0;j<8;j++) hl|=(uint64_t)b[j]<<(j*8);
    std::string js((size_t)hl,0); fread(&js[0],1,hl,f); *ds=8+hl; fclose(f);
    std::vector<SfTensor> ts; size_t i=0;
    while(i<js.size()&&js[i]!='}') {
        while(i<js.size()&&(js[i]==' '||js[i]==10||js[i]==9||js[i]==13||js[i]==','))i++;
        if(i>=js.size()||js[i]=='}')break;
        if(js[i]!='"'){i++;continue;}
        size_t ns=++i; while(i<js.size()&&js[i]!='"')i++;
        std::string name=js.substr(ns,i-ns); i++;
        if(name=="__metadata__"){while(i<js.size()&&js[i]!='}')i++;i++;continue;}
        while(i<js.size()&&js[i]!='{')i++; i++;
        SfTensor t; t.name=name; t.offset=0; t.size=0;
        while(i<js.size()&&js[i]!='}') {
            while(i<js.size()&&(js[i]==' '||js[i]==10||js[i]==9||js[i]==13||js[i]==','))i++;
            if(js[i]=='}')break; if(js[i]!='"'){i++;continue;}
            size_t fs=++i; while(i<js.size()&&js[i]!='"')i++;
            std::string fld=js.substr(fs,i-fs); i++;
            while(i<js.size()&&js[i]!=':')i++; i++;
            if(fld=="dtype"){if(js[i]=='"'){size_t vs=++i;while(i<js.size()&&js[i]!='"')i++;t.dtype=js.substr(vs,i-vs);i++;}}
            else if(fld=="shape"){if(js[i]=='['){i++;while(i<js.size()&&js[i]!=']'){while(i<js.size()&&(js[i]==' '||js[i]==','))i++;if(i<js.size()&&js[i]>='0'&&js[i]<='9'){char*e;t.shape.push_back(strtol(&js[i],&e,10));i=e-&js[0];}}i++;}}
            else if(fld=="data_offsets"){if(js[i]=='['){i++;int id=0;uint64_t v[2]={0,0};while(i<js.size()&&js[i]!=']'){while(i<js.size()&&(js[i]==' '||js[i]==','))i++;if(i<js.size()&&js[i]>='0'&&js[i]<='9'){char*e;v[id++]=strtoull(&js[i],&e,10);i=e-&js[0];}}i++;t.offset=v[0];t.size=v[1]-v[0];}}
        }
        i++; ts.push_back(t);
    }
    return ts;
}
static std::vector<uint8_t> load_td(const char *sf, size_t ds, size_t off, size_t sz) {
    FILE *f=fopen(sf,"rb"); fseeko(f,(off_t)(ds+off),SEEK_SET);
    std::vector<uint8_t> d(sz); fread(d.data(),1,sz,f); fclose(f); return d;
}
static std::vector<float> bf16_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> o(n); for(size_t i=0;i<n;i++){uint32_t u=(uint32_t)((uint16_t*)raw.data())[i]<<16; memcpy(&o[i],&u,4);} return o;
}