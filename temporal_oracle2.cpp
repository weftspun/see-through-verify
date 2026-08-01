// Temporal cross-frame block oracle v2: replicate cross_frame_block manually
// with intermediate taps to bisect our implementation.
// Usage: temporal_oracle2 <layerdiff-unet.gguf> <out.bin> [S F seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
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

    // replicate cross_frame_block with taps
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    ggml_tensor * res = xp;
    ggml_tensor * h  = layer_norm_affine(m, xp, pre + ".norm_in");
    ggml_tensor * hf = geglu_ff(m, h, pre + ".ff_in");
    ggml_tensor * hr = ggml_add(ctx, hf, res);
    ggml_tensor * n1 = layer_norm_affine(m, hr, pre + ".norm1");
    ggml_tensor * a1 = attn_tokens(m, n1, n1, pre + ".attn1", n_head);
    ggml_tensor * ha = ggml_add(ctx, hr, a1);
    ggml_tensor * ff = geglu_ff(m, layer_norm_affine(m, ha, pre + ".norm3"), pre + ".ff");
    ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, ff, 0, 2, 1, 3));

    for (ggml_tensor * t : {xp,h,hf,hr,n1,a1,ha,ff,out}) ggml_set_output(t);
    std::vector<ggml_tensor *> outs = {xp,h,hf,hr,n1,a1,ha,ff,out};

    std::vector<float> xd((size_t)C*S*F);
    uint32_t s2 = seedv*12345u;
    auto rnd=[&s2](){ s2=s2*1664525u+1013904223u; return ((s2>>8)&0xffff)/32767.5f*0.4f; };
    for (auto&v:xd) v=rnd();
    if (!compute_cpu_multi(m, outs, 262144, [&]{ ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4); })) return 1;

    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[5] = { C, S, F, n_head, (int32_t)outs.size() };
    fwrite(hdr, 4, 5, f);
    fwrite(xd.data(), 4, xd.size(), f);
    static const char * names[] = {"xp(norm_in)","norm_in_out","ff_in_out","ff_in+res","norm1_out","attn1_out","+attn1","ff_final","out"};
    for (size_t i=0;i<outs.size();i++){
        ggml_tensor * t = outs[i];
        size_t n=1; for(int d=0;d<(int)ggml_n_dims(t);d++) n*=(size_t)t->ne[d];
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n*4);
        fwrite(buf.data(), 4, n, f);
        printf("%s ne=[%lld,%lld,%lld,%lld] [0..7]=%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", names[i],
               (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
               buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7]);
    }
    fclose(f);
    printf("wrote %s\n", argv[2]);
    return 0;
}
