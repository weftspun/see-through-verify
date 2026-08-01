// GEGLU oracle: run ggml_geglu_erf_swapped on a synthetic [2g, T] input and
// dump, for comparison against our direct erf-GELU implementation.
// diffusers: out[c] = value[c]*gelu(gate[c]); ggml_geglu_erf_swapped has
// the gate in the SECOND half (swapped => x = second half, g = first half).
// Usage: geglu_oracle <out.bin> [g T]
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s out.bin [g T]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int g = 5120, T = 16;
    if (argc >= 4) { g = atoi(argv[2]); T = atoi(argv[3]); }

    const size_t max_nodes = 4096;
    ggml_init_params ip = { ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*g, T);
    ggml_set_input(x);
    ggml_tensor * y = ggml_geglu_erf_swapped(ctx, x);   // ne0 = g
    ggml_set_output(y);

    uint32_t seed = 5;
    std::vector<float> xd((size_t)2*g*T), yd((size_t)g*T);
    for (auto &v : xd) { seed = seed*1664525u+1013904223u; v = ((seed>>8)&0xffff)/32767.5f*4.0f - 2.0f; }

    ggml_backend_t bk = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(bk, 4);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
    if (ggml_backend_graph_compute(bk, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    ggml_backend_tensor_get(y, yd.data(), 0, yd.size()*4);

    FILE * f = fopen(argv[1], "wb");
    int32_t hdr[3] = { g, T, 4 };
    fwrite(hdr, 4, 3, f);
    fwrite(yd.data(), 4, yd.size(), f);
    fclose(f);
    // also dump input so the compare reconstructs it identically
    FILE * fi = fopen((std::string(argv[1])+".in").c_str(), "wb");
    fwrite(xd.data(), 4, xd.size(), fi);
    fclose(fi);
    printf("wrote %s (%zu) y[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", argv[1], yd.size(),
           yd[0],yd[1],yd[2],yd[3],yd[4],yd[5]);
    return 0;
}
