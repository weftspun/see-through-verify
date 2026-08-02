// Shared fast-loading + GEMM helpers for the see-through-verify compare
// harnesses. Provides:
//   - mmap-once safetensors loading (slice tensor data from one mapping
//     instead of reopening the multi-GB file per tensor)
//   - Accelerate-BLAS-backed token_linear (the same C BLAS numpy uses), so
//     the C++ harnesses run in seconds, not minutes, while staying pure C++.
//
// Include AFTER <vector>/<string>/<cstdint>. Consumed by:
//   unet_t3d_compare, btblock_compare, btblock_stage_compare,
//   temporal_compare (+ vertex of new harnesses).
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <Accelerate/Accelerate.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };

// Whole-file mapping; tensor data is sliced from this. Lifetime: process.
struct SfMap {
    const uint8_t * data = nullptr;
    size_t size = 0;
    int fd = -1;
    ~SfMap() { if (fd >= 0) { if (data) munmap((void*)data, size); close(fd); } }
};

// Map the whole safetensors file. Call once; tensors then slice via
// load_tensor_data(off, sz) with offsets already containing the data start.
static SfMap g_sfmap;

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

// mmap the whole file once; fills g_sfmap. All offsets from read_sf are
// relative to data_start, so absolute slice offset = data_start + tensor.offset.
static int map_safetensors(const char *path, size_t data_start) {
    g_sfmap.fd = open(path, O_RDONLY);
    if (g_sfmap.fd < 0) { perror("open"); return -1; }
    struct stat st; fstat(g_sfmap.fd, &st);
    g_sfmap.size = (size_t)st.st_size;
    g_sfmap.data = (const uint8_t*)mmap(nullptr, g_sfmap.size, PROT_READ, MAP_PRIVATE, g_sfmap.fd, 0);
    if (g_sfmap.data == MAP_FAILED) { perror("mmap"); return -1; }
    (void)data_start;
    return 0;
}

// Slice [off, off+sz) from the mapped file (off is an ABSOLUTE byte offset).
static std::vector<uint8_t> load_tensor_data(size_t off, size_t sz) {
    std::vector<uint8_t> data(sz);
    if (off + sz <= g_sfmap.size) memcpy(data.data(), g_sfmap.data + off, sz);
    return data;
}

// token-major linear y[t,o] = sum_k x[t,k]*W[o,k] + b[o], W stored [Cout,Cin].
// C = X·W^T + b, via Accelerate BLAS (numpy's C backend) for seconds-speed.
static void token_linear(const float *x, const float *W, const float *b, float *y,
                         int T, int Cin, int Cout) {
    if (T <= 0 || Cin <= 0 || Cout <= 0) return;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < Cout; o++) y[t*Cout+o] = b ? b[o] : 0.f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, Cout, Cin, 1.f, x, Cin, W, Cin, 1.f, y, Cout);
}
#pragma clang diagnostic pop

// Read an oracle tap file: header {i32 W,H,C,N,dtype} + raw f32 in WHCN layout.
// (Kept out of this header: each harness prefixes its oracle dir differently.)
