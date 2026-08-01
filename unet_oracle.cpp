// UNet oracle: run the full ggml unet_frame_forward with synthetic
// (but production-shaped) conditioning and dump per-stage taps for
// comparison against our verify dispatch-runner implementation.
// Inputs mirror pipeline.cpp layerdiff_pass (group_index 0, body pass).
// Usage: unet_oracle <layerdiff-unet.gguf> <outdir> [ZR F]
#include "test_common.h"
#include "unet_frame.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf outdir [ZR F]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    int ZR = 16, F = 1;
    if (argc >= 5) { ZR = atoi(argv[3]); F = atoi(argv[4]); }
    const int RES = ZR * 8;

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    init_graph_ctx(m, 294912);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 8, F);
    ggml_tensor * ehs_t  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
    ggml_tensor * text   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tids   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
    ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { sample, ehs_t, text, tids, ts }) ggml_set_input(t);

    const std::string gi = "0";   // group_index 0 (body pass)
    ggml_tensor * ehs2 = ggml_add(ctx, ehs_t, group_embedding(m, ehs_t, "group_embeds2." + gi));
    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000), "time_embedding");
    ggml_tensor * aug = ggml_add(ctx, text, group_embedding(m, text, "group_embeds." + gi));
    emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug, tids));

    std::vector<ggml_tensor *> taps;
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2, &taps,
                                           /*fine_taps_down0=*/true, /*fine_taps_mid=*/true);
    ggml_set_output(out);
    for (ggml_tensor * t : taps) ggml_set_output(t);
    ggml_set_output(emb);   // the 1280-time-embed the resnets consume

    std::vector<ggml_tensor *> outs = { out };
    outs.insert(outs.end(), taps.begin(), taps.end());
    outs.push_back(emb);

    // deterministic pseudo-random inputs
    std::vector<float> sdata((size_t)ZR*ZR*8*F), edata((size_t)2048*77*F),
                       xdata((size_t)1280*F), td((size_t)6*F);
    uint32_t seed = 7;
    auto rnd = [&seed](){ seed = seed*1664525u + 1013904223u; return ((seed>>8)&0xffff)/32767.5f - 1.0f; };
    for (auto & v : sdata) v = rnd() * 0.5f;
    for (auto & v : edata) v = rnd() * 0.2f;
    for (auto & v : xdata) v = rnd() * 0.2f;
    for (int f = 0; f < F; f++) {
        const float ids[6] = { (float)RES, (float)RES, 0, 0, (float)RES, (float)RES };
        for (int i = 0; i < 6; i++) td[f*6+i] = ids[i];
    }
    std::vector<float> tv(F, 961.0f);   // t=961 as the reference tests use

    if (!compute_cpu_multi(m, outs, 294912, [&]{
        ggml_backend_tensor_set(sample, sdata.data(), 0, sdata.size()*4);
        ggml_backend_tensor_set(ehs_t,  edata.data(), 0, edata.size()*4);
        ggml_backend_tensor_set(text,   xdata.data(), 0, xdata.size()*4);
        ggml_backend_tensor_set(tids,   td.data(), 0, td.size()*4);
        ggml_backend_tensor_set(ts,     tv.data(), 0, F*4);
    })) return 1;

    // write each output: header W,H,C,N + raw f32 in ggml WHCN layout
    auto write_bin = [&](const char * name, ggml_tensor * t) {
        FILE * f = fopen((std::string(argv[2]) + "/" + name).c_str(), "wb");
        int32_t hdr[5] = { (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3], 4 };
        fwrite(hdr, 4, 5, f);
        size_t n = (size_t)t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3];
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n*4);
        fwrite(buf.data(), 4, n, f);
        fclose(f);
        printf("  %s [%dx%dx%dx%d] first=%.4f\n", name, (int)t->ne[0],(int)t->ne[1],(int)t->ne[2],(int)t->ne[3], buf[0]);
    };

    write_bin("out.bin", out);
    // with fine taps enabled the order is:
    // conv_in, down0.resnet0, down0(post-downsample), down1, down2,
    // mid.resnet0, mid.attn, mid.resnet1
    static const char * tapnames[] = { "conv_in.bin", "down0_resnet0.bin", "down0.bin",
                                       "down1.bin", "down2.bin",
                                       "mid_resnet0.bin", "mid_attn.bin", "mid_resnet1.bin" };
    for (size_t i = 0; i < taps.size() && i < 8; i++) write_bin(tapnames[i], taps[i]);
    write_bin("emb.bin", emb);
    printf("wrote %zu + 2 outputs to %s\n", taps.size()+1, argv[2]);
    return 0;
}
