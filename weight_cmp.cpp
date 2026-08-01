// Compare a specific weight tensor between GGUF (f32, ggml) and safetensors (BF16)
// Dumps first 8 values + shape from the token embedding row for token 49406 (BOS)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "gguf.h"
#include "ggml.h"

struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *ds);
static std::vector<uint8_t> load_td(const char *sf, size_t ds, size_t off, size_t sz);
static std::vector<float> bf16_f32(const std::vector<uint8_t> &raw, size_t n);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s te1.gguf safetensors tensor_name\n", argv[0]); return 1; }
    // --- GGUF ---
    gguf_init_params gp = {false, nullptr};
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    int nt = gguf_get_n_tensors(g);
    // find matching tensor
    const char * want = argv[3];
    for (int i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        if (strcmp(name, want) == 0) {
            int64_t ne[4];
            ggml_type ty = gguf_get_tensor_type(g, i);
            // gguf_get_tensor_ne returns ne[] array
            const int64_t * nearr = gguf_get_tensor_ne(g, i);
            for (int k = 0; k < 4; k++) ne[k] = nearr[k];
            printf("GGUF %s: type=%s ne=[%lld,%lld,%lld,%lld]\n", name, ggml_type_name(ty),
                   (long long)ne[0],(long long)ne[1],(long long)ne[2],(long long)ne[3]);
            // f32: read raw bytes and interpret
            size_t off = gguf_get_tensor_offset(g, i);
            size_t nbytes = ggml_type_size(ty) * ne[0]*ne[1]*ne[2]*ne[3];
            FILE * f = fopen(argv[1],"rb");
            fseeko(f, (off_t)(gguf_get_data_offset(g)+off), SEEK_SET);
            std::vector<uint8_t> raw(nbytes);
            fread(raw.data(),1,nbytes,f); fclose(f);
            if (ty == GGML_TYPE_F32) {
                const float * fp = (const float*)raw.data();
                printf("  first 8: ");
                for (int k=0;k<8;k++) printf("%.4f ", fp[k]);
                printf("\n");
            } else if (ty == GGML_TYPE_F16) {
                const uint16_t * hp = (const uint16_t*)raw.data();
                printf("  first 8 f16: ");
                for (int k=0;k<8;k++) printf("%.4f ", ggml_fp16_to_fp32(hp[k]));
                printf("\n");
            }
            break;
        }
    }
    gguf_free(g);

    // --- safetensors ---
    size_t ds=0;
    auto tensors = read_sf(argv[2], &ds);
    for (auto &t : tensors) {
        if (t.name == want) {
            int R=(int)t.shape[0], C=(int)t.shape[1];
            auto raw = load_td(argv[2], ds, t.offset, t.size);
            auto w = bf16_f32(raw, (size_t)R*C);
            printf("SF %s: shape=[%d,%d] first 8: ", t.name.c_str(), R, C);
            for (int k=0;k<8;k++) printf("%.4f ", w[k]);
            printf("\n");
            break;
        }
    }
    return 0;
}
static std::vector<SfTensor> read_sf(const char *path, size_t *ds){
    FILE*f=fopen(path,"rb");uint8_t b[8];fread(b,1,8,f);
    uint64_t hl=0;for(int j=0;j<8;j++)hl|=(uint64_t)b[j]<<(j*8);
    std::string js((size_t)hl,0);fread(&js[0],1,hl,f);*ds=8+hl;fclose(f);
    std::vector<SfTensor> ts;size_t i=0;
    while(i<js.size()&&js[i]!='}'){
        while(i<js.size()&&(js[i]==' '||js[i]==10||js[i]==9||js[i]==13||js[i]==','))i++;
        if(i>=js.size()||js[i]=='}')break;if(js[i]!='"'){i++;continue;}
        size_t ns=++i;while(i<js.size()&&js[i]!='"')i++;
        std::string name=js.substr(ns,i-ns);i++;
        if(name=="__metadata__"){while(i<js.size()&&js[i]!='}')i++;i++;continue;}
        while(i<js.size()&&js[i]!='{')i++;i++;
        SfTensor t;t.name=name;t.offset=0;t.size=0;
        while(i<js.size()&&js[i]!='}'){
            while(i<js.size()&&(js[i]==' '||js[i]==10||js[i]==9||js[i]==13||js[i]==','))i++;
            if(js[i]=='}')break;if(js[i]!='"'){i++;continue;}
            size_t fs=++i;while(i<js.size()&&js[i]!='"')i++;
            std::string fld=js.substr(fs,i-fs);i++;
            while(i<js.size()&&js[i]!=':')i++;i++;
            if(fld=="dtype"){if(js[i]=='"'){size_t vs=++i;while(i<js.size()&&js[i]!='"')i++;t.dtype=js.substr(vs,i-vs);i++;}}
            else if(fld=="shape"){if(js[i]=='['){i++;while(i<js.size()&&js[i]!=']'){while(i<js.size()&&(js[i]==' '||js[i]==','))i++;if(i<js.size()&&js[i]>='0'&&js[i]<='9'){char*e;t.shape.push_back(strtol(&js[i],&e,10));i=e-&js[0];}}i++;}}
            else if(fld=="data_offsets"){if(js[i]=='['){i++;int id=0;uint64_t v[2]={0,0};while(i<js.size()&&js[i]!=']'){while(i<js.size()&&(js[i]==' '||js[i]==','))i++;if(i<js.size()&&js[i]>='0'&&js[i]<='9'){char*e;v[id++]=strtoull(&js[i],&e,10);i=e-&js[0];}}i++;t.offset=v[0];t.size=v[1]-v[0];}}
        }
        i++;ts.push_back(t);
    }
    return ts;
}
static std::vector<uint8_t> load_td(const char *sf,size_t ds,size_t off,size_t sz){
    FILE*f=fopen(sf,"rb");fseeko(f,(off_t)(ds+off),SEEK_SET);
    std::vector<uint8_t>d(sz);fread(d.data(),1,sz,f);fclose(f);return d;
}
static std::vector<float> bf16_f32(const std::vector<uint8_t>&raw,size_t n){
    std::vector<float>o(n);for(size_t i=0;i<n;i++){uint32_t u=(uint32_t)((uint16_t*)raw.data())[i]<<16;memcpy(&o[i],&u,4);}return o;
}