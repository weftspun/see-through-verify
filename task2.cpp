#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>

// ---- safetensors reader from see-through.cpp ----

struct SfTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    size_t offset, size;
};

static std::vector<SfTensor> read_sf(const char *path, size_t *data_start) {
    FILE *f = fopen(path, "rb");
    uint8_t buf[8]; fread(buf, 1, 8, f);
    uint64_t hdr_len = 0;
    for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)buf[j] << (j * 8);
    std::string json((size_t)hdr_len, '\0');
    fread(&json[0], 1, hdr_len, f);
    *data_start = 8 + hdr_len;
    fclose(f);

    std::vector<SfTensor> tensors;
    size_t i = 0;
    while (i < json.size() && json[i] != '}') {
        // Skip whitespace/comma
        while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\t' || json[i] == '\r' || json[i] == ',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }

        // Tensor name
        size_t ns = ++i;
        while (i < json.size() && json[i] != '"') i++;
        std::string name = json.substr(ns, i - ns); i++;
        if (name == "__metadata__") {
            while (i < json.size() && json[i] != '}') i++; i++;
            continue;
        }
        while (i < json.size() && json[i] != '{') i++; i++;

        SfTensor t; t.name = name; t.offset = 0; t.size = 0;
        while (i < json.size() && json[i] != '}') {
            while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\t' || json[i] == ',')) i++;
            if (json[i] == '}') break;
            if (json[i] != '"') { i++; continue; }
            size_t fs = ++i;
            while (i < json.size() && json[i] != '"') i++;
            std::string field = json.substr(fs, i - fs); i++;
            while (i < json.size() && json[i] != ':') i++; i++;

            if (field == "dtype") {
                if (json[i] == '"') {
                    size_t vs = ++i;
                    while (i < json.size() && json[i] != '"') i++;
                    t.dtype = json.substr(vs, i - vs); i++;
                }
            } else if (field == "shape") {
                if (json[i] == '[') {
                    i++;
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            t.shape.push_back(strtol(&json[i], &end, 10));
                            i = end - &json[0];
                        }
                    }
                    i++;
                }
            } else if (field == "data_offsets") {
                if (json[i] == '[') {
                    i++;
                    int idx = 0; uint64_t vals[2] = {0,0};
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            vals[idx++] = strtoull(&json[i], &end, 10);
                            i = end - &json[0];
                        }
                    }
                    i++;
                    t.offset = vals[0]; t.size = vals[1] - vals[0];
                }
            }
        }
        i++;
        tensors.push_back(t);
    }
    return tensors;
}

int main() {
    const char *sf_path = "hf_cache/layerdifforg_seethroughv0.0.2_layerdiff3d/text_encoder/model.safetensors";
    size_t data_start = 0;
    auto tensors = read_sf(sf_path, &data_start);
    printf("tensors: %zu\n", tensors.size());

    // Find first BF16 weight matrix
    SfTensor *wt = nullptr;
    for (auto &t : tensors) {
        if (t.dtype == "BF16" && t.shape.size() == 2) {
            wt = &t; break;
        }
    }
    if (!wt) { printf("no BF16 weight found\n"); return 1; }

    printf("weight: %s %s [%lld x %lld] (%zu bytes)\n",
           wt->name.c_str(), wt->dtype.c_str(),
           (long long)wt->shape[0], (long long)wt->shape[1],
           wt->size);

    // Load raw data
    FILE *f = fopen(sf_path, "rb");
    fseeko(f, (off_t)(data_start + wt->offset), SEEK_SET);
    std::vector<uint8_t> raw(wt->size);
    fread(raw.data(), 1, wt->size, f);
    fclose(f);

    int R = (int)wt->shape[0], C = (int)wt->shape[1];
    printf("R=%d C=%d\n", R, C);

    // Convert BF16 to F32
    // BF16: upper 16 bits of F32, lower 16 bits zero
    std::vector<float> h_W(R * C);
    for (int i = 0; i < R * C; i++) {
        uint16_t u16 = ((uint16_t*)raw.data())[i];
        uint32_t u32 = (uint32_t)u16 << 16;
        memcpy(&h_W[i], &u32, 4);
    }

    // Simple validation — print first few weights
    printf("First 5 weights: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_W[i]);
    printf("\n");

    // GEMM with random input
    int N = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> h_X(C * N), h_Y(R * N);
    for (auto &v : h_X) v = dist(rng);

    for (int i = 0; i < R; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < C; k++)
                s += h_W[i * C + k] * h_X[k * N + j];
            h_Y[i * N + j] = s;
        }

    printf("GEMM %dx%dx%d: y[0]=%.4f y[%d]=%.4f\n",
           R, N, C, h_Y[0], R*N-1, h_Y[R*N-1]);
    printf("GREEN — real weights loaded and GEMM computed on CPU\n");
    return 0;
}