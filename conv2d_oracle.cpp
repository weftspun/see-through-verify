// Conv2d oracle: run ggml's conv_in convolution on a synthetic latent and
// dump the output, for comparison against our Lean Conv2d shader semantics.
// conv_in: latent (WHCN) 8ch -> 320ch, k3 stride1 pad1.
// Usage: conv2d_oracle <layerdiff-unet.gguf> <out.bin> [W H C F]
#include "test_common.h"
#include "ops.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf out.bin [W H C F]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    int W = 16, H = 16, C = 8, F = 1;
    if (argc >= 7) { W = atoi(argv[3]); H = atoi(argv[4]); C = atoi(argv[5]); F = atoi(argv[6]); }

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }

    init_graph_ctx(m, 65536);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, F);
    ggml_set_input(x);

    ggml_tensor * out = conv2d(m, x, "conv_in", 1, 1);  // 320ch out
    ggml_set_output(out);

    // deterministic pseudo-random latent in [-1, 1]
    std::vector<float> xdata((size_t)W*H*C*F);
    uint32_t seed = 42;
    for (auto & v : xdata) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) & 0xffff) / 32767.5f - 1.0f; }

    if (!compute_cpu(m, out, 65536, [&]{ ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size()*4); })) return 1;

    int64_t oC = out->ne[2], oW = out->ne[0], oH = out->ne[1], oN = out->ne[3];
    printf("conv_in out: [%lld x %lld x %lld x %lld]  (W=%d,H=%d,C=%d,F=%d)\n",
           (long long)oW, (long long)oH, (long long)oC, (long long)oN, W, H, C, F);

    std::vector<float> odata((size_t)oC*oH*oW*oN);
    ggml_backend_tensor_get(out, odata.data(), 0, odata.size()*4);
    // ggml WHCN layout: index = w + h*oW + c*oW*oH + n*oW*oH*oC

    FILE * f = fopen(argv[2], "wb");
    // header: W,H,C,N,f32
    int32_t hdr[5] = { (int32_t)oW, (int32_t)oH, (int32_t)oC, (int32_t)oN, 4 };
    fwrite(hdr, 4, 5, f);
    fwrite(odata.data(), 4, odata.size(), f);
    fclose(f);
    printf("wrote %s (%zu bytes, WHCN f32)\n", argv[2], odata.size()*4);
    printf("out[oc=%d] first 8 vals: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           0, odata[0],odata[1],odata[2],odata[3],odata[4],odata[5],odata[6],odata[7]);
    return 0;
}
