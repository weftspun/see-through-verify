// BasicTransformerBlock oracle: run one mid_block transformer_blocks.0
// through real ggml (basic_transformer_block + attn_tokens) on a synthetic
// spatial input, dump output for comparison. Isolates self/cross attention,
// layernorm and GEGLU wiring from the surrounding transformer3d plumbing.
// Usage: btblock_oracle <layerdiff-unet.gguf> <out.bin> [S F seed]
#include "test_common.h"
#include "unet_frame.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf out.bin [S F seed]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int S = 16, F = 1, seed = 1;
    if (argc >= 6) { S = atoi(argv[3]); F = atoi(argv[4]); seed = atoi(argv[5]); }
    const int C = 1280, Tk = 77, EHD = 2048, n_head = 20;

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    init_graph_ctx(m, 131072);
    ggml_context * ctx = m.ctx_g;

    // x as (C, S, F) token tensor (ggml ne0=C, ne1=S, ne2=F)
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, S, F);
    ggml_tensor * ehs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, EHD, Tk, F);
    ggml_set_input(x); ggml_set_input(ehs);

    const std::string pre = "mid_block.attentions.0.transformer_blocks.0";
    ggml_tensor * out = basic_transformer_block(m, x, ehs, pre, n_head);
    ggml_set_output(out);

    // x / ehs data (token-major per frame)
    std::vector<float> xd((size_t)C*S*F), ed((size_t)EHD*Tk*F);
    uint32_t s2 = seed*12345u;
    auto rnd=[&s2](){ s2=s2*1664525u+1013904223u; return ((s2>>8)&0xffff)/32767.5f*0.4f; };
    for (auto&v:xd) v=rnd();
    for (auto&v:ed) v=rnd();

    // dump input too (same layout as x: ne0=C fastest)
    if (!compute_cpu_multi(m, { out }, 131072, [&]{
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
        ggml_backend_tensor_set(ehs, ed.data(), 0, ed.size()*4);
    })) return 1;

    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[5] = { C, S, F, Tk, EHD };
    fwrite(hdr, 4, 5, f);
    std::vector<float> od((size_t)C*S*F);
    ggml_backend_tensor_get(out, od.data(), 0, od.size()*4);
    fwrite(od.data(), 4, od.size(), f);
    // inputs
    fwrite(xd.data(), 4, xd.size(), f);
    fwrite(ed.data(), 4, ed.size(), f);
    fclose(f);
    printf("wrote %s: out[%dx%dx%dx?] first=%.4f\n", argv[2], C, S, F, od[0]);
    printf("out[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           od[0],od[1],od[2],od[3],od[4],od[5],od[6],od[7]);
    return 0;
}
