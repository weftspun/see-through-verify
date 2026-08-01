// Print tensor types + first weight dims from a GGUF file (no ggml needed)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static uint32_t rd_u32(FILE*f){uint32_t v;fread(&v,4,1,f);return v;}
static uint64_t rd_u64(FILE*f){uint64_t v;fread(&v,8,1,f);return v;}
static std::string rd_str(FILE*f){uint32_t n=rd_u32(f);std::string s(n,0);fread(&s[0],1,n,f);return s;}

const char * type_name(uint32_t t) {
    static const char * names[] = {"f32","f16","f64","i8","i16","i32","i64","q8_0","q4_0","q4_1","q2k","q3k","q4k","q5k","q6k","q8k","bf16"};
    return (t < sizeof(names)/sizeof(names[0])) ? names[t] : "?";
}

int main(int argc, char **argv) {
    FILE * f = fopen(argv[1], "rb");
    char magic[4]; fread(magic,1,4,f);
    uint32_t ver = rd_u32(f);
    uint64_t n_tensor = rd_u64(f), n_kv = rd_u64(f);
    printf("GGUF %s v%u tensors=%llu\n", magic, ver, (unsigned long long)n_tensor);
    // skip KV metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        auto key = rd_str(f);
        uint32_t ty = rd_u32(f);
        // skip value by type (simplified: strings + basic)
        switch (ty) {
            case 8: rd_str(f); break; // string
            case 0: { float v; fread(&v,4,1,f); } break;
            case 1: { float v; fread(&v,4,1,f); } break; // f32
            case 4: { uint32_t v; fread(&v,4,1,f); } break;
            case 5: { uint64_t v; fread(&v,8,1,f); } break;
            case 7: { int32_t v; fread(&v,4,1,f); } break;
            case 11: { int64_t v; fread(&v,8,1,f); } break;
            case 10: { uint32_t n=rd_u32(f); for(uint32_t j=0;j<n;j++){uint32_t v;fread(&v,4,1,f);} } break;
            default: { fprintf(stderr, "unknown kv type %u at key %s\n", ty, key.c_str()); return 1; }
        }
    }
    // tensor info
    for (uint64_t i = 0; i < n_tensor && i < 10; i++) {
        auto name = rd_str(f);
        uint32_t n_dims = rd_u32(f);
        uint64_t ne[4] = {1,1,1,1};
        for (uint32_t d = 0; d < n_dims; d++) ne[d] = rd_u64(f);
        uint32_t ty = rd_u32(f);
        uint64_t off = rd_u64(f);
        printf("  %s: type=%s ne=[%llu,%llu,%llu,%llu]\n", name.c_str(), type_name(ty),
               (unsigned long long)ne[0],(unsigned long long)ne[1],(unsigned long long)ne[2],(unsigned long long)ne[3]);
    }
    return 0;
}