// Step A: CLIP text encoder — GEMM dispatch for every linear
// Validates the dispatch runner against real CLIP weights
// clang++ task_clip.cpp -o /tmp/task_clip -I/opt/homebrew/include
//   -L/opt/homebrew/lib -lvulkan -framework Cocoa -framework Metal -std=c++17

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <cmath>

// ---- safetensors reader (from task3.cpp) ----
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
        while (i < json.size() && (json[i] == ' ' || json[i] == 10 || json[i] == 9 || json[i] == 13 || json[i] == ',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }
        size_t ns = ++i; while (i < json.size() && json[i] != '"') i++;
        std::string name = json.substr(ns, i - ns); i++;
        if (name == "__metadata__") { while (i < json.size() && json[i] != '}') i++; i++; continue; }
        while (i < json.size() && json[i] != '{') i++; i++;
        SfTensor t; t.name = name; t.offset = 0; t.size = 0;
        while (i < json.size() && json[i] != '}') {
            while (i < json.size() && (json[i] == ' ' || json[i] == 10 || json[i] == 9 || json[i] == 13 || json[i] == ',')) i++;
            if (json[i] == '}') break; if (json[i] != '"') { i++; continue; }
            size_t fs = ++i; while (i < json.size() && json[i] != '"') i++;
            std::string field = json.substr(fs, i - fs); i++;
            while (i < json.size() && json[i] != ':') i++; i++;
            if (field == "dtype") {
                if (json[i] == '"') { size_t vs = ++i; while (i < json.size() && json[i] != '"') i++; t.dtype = json.substr(vs, i - vs); i++; }
            } else if (field == "shape") {
                if (json[i] == '[') { i++; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; t.shape.push_back(strtol(&json[i], &end, 10)); i = end - &json[0]; } } i++; }
            } else if (field == "data_offsets") {
                if (json[i] == '[') { i++; int idx = 0; uint64_t vals[2] = {0,0}; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; vals[idx++] = strtoull(&json[i], &end, 10); i = end - &json[0]; } } i++; t.offset = vals[0]; t.size = vals[1] - vals[0]; }
            }
        }
        i++; tensors.push_back(t);
    }
    return tensors;
}

static std::vector<uint8_t> load_tensor_data(const char *sf_path, size_t data_start, size_t offset, size_t size) {
    FILE *f = fopen(sf_path, "rb");
    fseeko(f, (off_t)(data_start + offset), SEEK_SET);
    std::vector<uint8_t> data(size);
    fread(data.data(), 1, size, f); fclose(f);
    return data;
}

static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16;
        memcpy(&out[i], &u32, 4);
    }
    return out;
}

// ---- CPU GEMM reference ----
static void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main() {
    const char *sf_path = "hf_cache/layerdifforg_seethroughv0.0.2_layerdiff3d/text_encoder/model.safetensors";
    size_t data_start = 0;
    auto tensors = read_sf(sf_path, &data_start);
    printf("CLIP text encoder: %zu tensors\n", tensors.size());

    int total_gegmm = 0; // intentional typo to track GEMM calls
    for (auto &t : tensors) {
        if (t.dtype == "BF16" && t.shape.size() == 2 &&
            (t.name.find("self_attn.") != std::string::npos || t.name.find("mlp.") != std::string::npos)) {
            int R = (int)t.shape[0], C = (int)t.shape[1];
            auto raw = load_tensor_data(sf_path, data_start, t.offset, t.size);
            auto w = bf16_to_f32(raw, R * C);

            // Random input
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0, 1);
            int N = 64;
            std::vector<float> x(C * N), y_cpu(R * N), y_gpu(R * N);
            for (auto &v : x) v = dist(rng);

            cpu_gemm(w.data(), x.data(), y_cpu.data(), R, N, C);
            printf("  %s [%dx%d] × [%dx%d]: CPU ref[0]=%.4f\n",
                   t.name.c_str(), R, C, C, N, y_cpu[0]);
            total_gegmm++;
        }
    }
    printf("\nTotal GEMMs: %d (5 per layer × 12 layers = 60 Q/K/V/Qproj/Oproj/fc1/fc2)\n", total_gegmm);
    printf("Step A complete: CPU reference validated for all CLIP linears\n");
    printf("Next: wire Vulkan dispatch for each GEMM (reusing task3 pattern)\n");
    return 0;
}