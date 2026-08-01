// Transformer3D oracle with per-layer + per-temporal taps: replicate the
// mid transformer3d manually, dumping after each spatial layer and each
// temporal cross-frame block, to bisect where our t3d compare diverges.
// Usage: t3d_oracle <layerdiff-unet.gguf> <out.bin> [S F seed]
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
    const int C = 1280, EHD = 2048, Tk = 77;

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    init_graph_ctx(m, 262144);
    ggml_context * ctx = m.ctx_g;
    // x: (W,H,C,F) = (S/ sqrt? ) -- transformer3d takes (W,H,C,F). Use W=H so W*H=S.
    int W = (int)sqrt((double)S), H = S / W;
    if (W*H != S) { W=S; H=1; }
    ggml_tensor * x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, F);
    ggml_tensor * ehs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, EHD, Tk, F);
    ggml_set_input(x); ggml_set_input(ehs);

    const std::string pre = "mid_block.attentions.0";
    auto pin_w=load_tensor(ctx); (void)pin_w;
    ggml_tensor * h = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));
    ggml_tensor * out_t3d = transformer3d(m, x, h, pre, C/64, 0);  // placeholder, n_layers from probes below

    // probe layer count
    int n_layers=0; while (m.has(pre+".transformer_blocks."+std::to_string(n_layers)+".norm1.weight")) n_layers++;
    int n_temp=0;   while (m.has(pre+".temporal_transformer_blocks."+std::to_string(n_temp)+".norm_in.weight")) n_temp++;
    printf("n_layers=%d n_temp=%d\n", n_layers, n_temp);

    // replicate transformer3d manually with taps
    m.gn_groups=32; m.gn_eps=1e-6f;
    ggml_tensor * gn = group_norm_affine(m, x, pre + ".norm");
    m.gn_groups=32; m.gn_eps=1e-5f;
    gn = ggml_reshape_3d(ctx, gn, W*H, C, F);
    gn = ggml_cont(ctx, ggml_permute(ctx, gn, 1, 0, 2, 3));   // (C, S, F)
    ggml_tensor * cur = linear(m, gn, pre + ".proj_in");
    std::vector<ggml_tensor*> taps;
    taps.push_back(cur);   // after proj_in
    int stride = n_layers >= 3 ? 2 : 1;
    int tt = 0;
    ggml_tensor * ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));
    for (int l = 0; l < n_layers; l++) {
        std::string bp = pre+".transformer_blocks."+std::to_string(l);
        cur = basic_transformer_block(m, cur, ehs2, bp, C/64);
        taps.push_back(cur);   // after spatial layer l
        if ((l+1) % stride == 0 && tt < n_temp) {
            std::string tp = pre+".temporal_transformer_blocks."+std::to_string(tt++);
            cur = ggml_add(ctx, cur, cross_frame_block(m, cur, tp, C/64));
            taps.push_back(cur);  // after temporal block (added)
        }
    }
    cur = linear(m, cur, pre + ".proj_out");
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 0, 2, 3));   // (S, C, F)
    cur = ggml_reshape_4d(ctx, cur, W, H, C, F);
    cur = ggml_add(ctx, cur, x);    // residual
    taps.push_back(cur);            // final out

    for (ggml_tensor * t : taps) ggml_set_output(t);

    std::vector<float> xd((size_t)W*H*C*F), ed((size_t)EHD*Tk*F);
    uint32_t s2 = seedv*12345u;
    auto rnd=[&s2](){ s2=s2*1664525u+1013904223u; return ((s2>>8)&0xffff)/32767.5f*0.4f; };
    for (auto&v:xd) v=rnd();
    for (auto&v:ed) v=rnd();
    if (!compute_cpu_multi(m, taps, 262144, [&]{
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
        ggml_backend_tensor_set(ehs, ed.data(), 0, ed.size()*4);
    })) return 1;

    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[6] = { C, S, F, EHD, Tk, (int32_t)taps.size() };
    fwrite(hdr, 4, 6, f);
    fwrite(xd.data(), 4, xd.size(), f);
    fwrite(ed.data(), 4, ed.size(), f);
    for (size_t i=0;i<taps.size();i++){
        ggml_tensor * t = taps[i];
        size_t n=1; for(int d=0;d<(int)ggml_n_dims(t);d++) n*=(size_t)t->ne[d];
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n*4);
        fwrite(buf.data(), 4, n, f);
        printf("tap%zu ne=[%lld,%lld,%lld,%lld] [0..3]=%.3f %.3f %.3f %.3f\n", i,
               (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
               buf[0],buf[1],buf[2],buf[3]);
    }
    fclose(f);
    printf("wrote %s (%zu taps)\n", argv[2], taps.size());
    return 0;
}
