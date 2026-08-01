// Temporal cross-frame block oracle: run ONE mid temporal_transformer_blocks.0
// (cross_frame_block) against ggml, dump output for comparison. x is (C,S,F).
// Usage: temporal_oracle <layerdiff-unet.gguf> <out.bin> [S F seed]
#include "test_common.h"
#include "unet_frame.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf out.bin [S F seed]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int S = 4, F = 13, seedv = 1;
    if (argc >= 6) { S = atoi(argv[3]); F = atoi(argv[4]); seedv = atoi(argv[5]); }
    const int C = 1280, n_head = 20;

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    init_graph_ctx(m, 262144);
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, S, F);
    ggml_set_input(x);
    const std::string pre = "mid_block.attentions.0.temporal_transformer_blocks.0";
    ggml_tensor * out = cross_frame_block(m, x, pre, n_head);
    ggml_set_output(out);

    std::vector<float> xd((size_t)C*S*F);
    uint32_t s2 = seedv*12345u;
    auto rnd=[&s2](){ s2=s2*1664525u+1013904223u; return ((s2>>8)&0xffff)/32767.5f*0.4f; };
    for (auto&v:xd) v=rnd();

    if (!compute_cpu_multi(m, {out}, 262144, [&]{ ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4); })) return 1;
    std::vector<float> od((size_t)C*S*F);
    ggml_backend_tensor_get(out, od.data(), 0, od.size()*4);
    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[4] = { C, S, F, n_head };
    fwrite(hdr, 4, 4, f);
    fwrite(od.data(), 4, od.size(), f);
    fwrite(xd.data(), 4, xd.size(), f);
    fclose(f);
    printf("wrote %s out[%d,%d,%d] first=%.4f\n", argv[2], C, S, F, od[0]);
    printf("out[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", od[0],od[1],od[2],od[3],od[4],od[5],od[6],od[7]);
    return 0;
}
