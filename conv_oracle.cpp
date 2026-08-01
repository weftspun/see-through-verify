// Conv2d oracle: run ggml conv_in on deterministic input, dump input+output.
// Usage: conv_oracle <layerdiff-unet.gguf> <out.bin> [W H C F seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "test_common.h"
#include "unet_frame.h"
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf out.bin [W H C F seed]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int W=8,H=8,C=8,F=13, seedv=99;
    if (argc >= 8) { W=atoi(argv[3]); H=atoi(argv[4]); C=atoi(argv[5]); F=atoi(argv[6]); seedv=atoi(argv[7]); }
    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    init_graph_ctx(m, 262144);
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, F);
    ggml_set_input(x);
    ggml_tensor * w = m.get("conv_in.weight");
    ggml_tensor * out = ggml_conv_2d(ctx, w, x, 1, 1, 1, 1, 1, 1);   // no bias add (broadcast asserts)
    ggml_set_output(out);
    int OC = (int)w->ne[3];
    std::vector<float> xd((size_t)W*H*C*F);
    uint32_t seed = (uint32_t)seedv;
    for (auto &v : xd) { seed = seed*1664525u+1013904223u; v = ((seed>>8)&0xffff)/32767.5f*0.5f; }
    if (!compute_cpu_multi(m, {out}, 262144, [&]{ ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4); })) return 1;
    size_t on=(size_t)W*H*OC*F;
    std::vector<float> od(on);
    ggml_backend_tensor_get(out, od.data(), 0, on*4);
    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[5] = { W, H, OC, F, C };
    fwrite(hdr, 4, 5, f);
    fwrite(od.data(), 4, on, f);
    fwrite(xd.data(), 4, xd.size(), f);
    fclose(f);
    printf("wrote %s conv_out [%dx%dx%dx%d] oc=%d first=%.4f\n", argv[2], W,H,OC,F, OC, od[0]);
    return 0;
}
