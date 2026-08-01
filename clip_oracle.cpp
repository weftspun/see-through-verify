// Oracle: produce CLIP-L penultimate hidden states via see-through-cpp ggml
// (the ggml CLIP is validated against transformers in test_clip ocean-gate)
// Then compare against our Slang-runner clip_encoder.
// Usage: clip_oracle <te.gguf> <prompt_ids.oracle.json>
#include "/Users/ernest.lee/Forges/see-through-cpp/src/clip.h"
#include "/Users/ernest.lee/Forges/see-through-cpp/tests/test_common.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s te.gguf out.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }

    std::string cfg, vocab_js, merges;
    for (const auto & kv : m.config_json) {
        auto ends = [&](const char * s) { size_t n=strlen(s); return kv.first.size()>n && kv.first.compare(kv.first.size()-n,n,s)==0; };
        if (ends(".config_json")) cfg = kv.second;
        if (ends(".vocab_json")) vocab_js = kv.second;
        if (ends(".merges_txt")) merges = kv.second;
    }
    ClipTokenizer tok;
    tok.load(vocab_js, merges);
    ClipParams p = clip_params_from_config(cfg);
    printf("params: %dL/%dH d=%d quick_gelu=%d\n", p.n_layer, p.n_head, p.d_model, p.quick_gelu);

    const char * PROMPT = "solo, 1girl, blue hair, cat ears, school uniform";
    int eos_pos = 0;
    auto ids = tok.encode_padded(PROMPT, 77, tok.eos_id, &eos_pos);
    printf("token ids (%zu): ", ids.size());
    for (size_t i = 0; i < 10; i++) printf("%d ", ids[i]);
    printf("...\n");

    init_graph_ctx(m, 8192);
    ggml_tensor * ids_t = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, ids.size());
    ggml_set_input(ids_t);
    ggml_tensor * penult = nullptr, * final_out = nullptr;
    clip_text_graph(m, ids_t, p, &penult, &final_out);
    ggml_set_output(penult);
    if (!compute_cpu(m, penult, 8192, [&]{ ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size()*4); })) return 1;

    // Dump token ids + penultimate hidden [d, n] as raw f32
    FILE * fi = fopen((std::string(argv[2]) + ".ids").c_str(), "w");
    for (size_t i = 0; i < ids.size(); i++) fprintf(fi, "%d ", ids[i]);
    fclose(fi);
    int n = ids.size(), d = p.d_model;
    FILE * f = fopen(argv[2], "wb");
    std::vector<float> out((size_t)n*d);
    ggml_backend_tensor_get(penult, out.data(), 0, out.size()*4);
    fwrite(out.data(), 4, out.size(), f);
    fclose(f);
    printf("wrote %d x %d penultimate hidden to %s\n", d, n, argv[2]);
    printf("raw[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);

    // Dump Q projection (layer 0, after scale) for orientation check
    {
        const int hd_dim = d / p.n_head;
        ggml_context * ec = m.ctx_g;
        ggml_tensor * ids2 = ggml_new_tensor_1d(ec, GGML_TYPE_I32, ids.size());
        ggml_set_input(ids2);
        ggml_tensor * xx = ggml_get_rows(ec, m.get("text_model.embeddings.token_embedding.weight"), ids2);
        ggml_tensor * pw = m.get("text_model.embeddings.position_embedding.weight");
        ggml_tensor * po = ggml_view_2d(ec, pw, d, ids.size(), pw->nb[1], 0);
        xx = ggml_add(ec, xx, ggml_cast(ec, po, GGML_TYPE_F32));
        ggml_tensor * hln = layer_norm_affine(m, xx, "text_model.encoder.layers.0.layer_norm1");
        ggml_set_output(hln);
        if (!compute_cpu(m, hln, 8192, [&]{ ggml_backend_tensor_set(ids2, ids.data(), 0, ids.size()*4); })) return 1;
        std::vector<float> hdata((size_t)ids.size()*d);
        ggml_backend_tensor_get(hln, hdata.data(), 0, hdata.size()*4);
        printf("oracle LN1(t0,d0..7):");
        for (int k = 0; k < 8; k++) printf(" %.4f", hdata[k]);
        printf("\n");

        ggml_tensor * qq = linear(m, hln, "text_model.encoder.layers.0.self_attn.q_proj");
        qq = ggml_scale(ec, qq, 1.0f / sqrtf((float) hd_dim));
        ggml_set_output(qq);
        if (!compute_cpu(m, qq, 8192, [&]{ ggml_backend_tensor_set(ids2, ids.data(), 0, ids.size()*4); })) return 1;
        std::vector<float> qdata((size_t)ids.size()*d);
        ggml_backend_tensor_get(qq, qdata.data(), 0, qdata.size()*4);
        printf("oracle Q(t0,d0..7):");
        for (int k = 0; k < 8; k++) printf(" %.4f", qdata[k]);
        printf("\n");
        // also dump post-embedding at dims 40..48 (token 0)
        std::vector<float> edata((size_t)ids.size()*d);
        memset(edata.data(), 0, edata.size()*4);
        // recompute x and read dims 40-48
        ggml_tensor * xo = ggml_get_rows(ec, m.get("text_model.embeddings.token_embedding.weight"), ids2);
        ggml_tensor * pw2 = m.get("text_model.embeddings.position_embedding.weight");
        ggml_tensor * po2 = ggml_view_2d(ec, pw2, d, ids.size(), pw2->nb[1], 0);
        xo = ggml_add(ec, xo, ggml_cast(ec, po2, GGML_TYPE_F32));
        ggml_set_output(xo);
        if (!compute_cpu(m, xo, 8192, [&]{ ggml_backend_tensor_set(ids2, ids.data(), 0, ids.size()*4); })) return 1;
        std::vector<float> edata2((size_t)ids.size()*d);
        ggml_backend_tensor_get(xo, edata2.data(), 0, edata2.size()*4);
        printf("oracle post-emb[40..48]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               edata2[40],edata2[41],edata2[42],edata2[43],edata2[44],edata2[45],edata2[46],edata2[47],edata2[48]);
    }

    // Dump per-layer hidden states for debugging
    {
        const int hd_dim = d / p.n_head;
        ggml_context * ec = m.ctx_g;
        ggml_tensor * ids2 = ggml_new_tensor_1d(ec, GGML_TYPE_I32, ids.size());
        ggml_set_input(ids2);
        ggml_tensor * xx = ggml_get_rows(ec, m.get("text_model.embeddings.token_embedding.weight"), ids2);
        ggml_tensor * pw = m.get("text_model.embeddings.position_embedding.weight");
        ggml_tensor * po = ggml_view_2d(ec, pw, d, ids.size(), pw->nb[1], 0);
        xx = ggml_add(ec, xx, ggml_cast(ec, po, GGML_TYPE_F32));
        for (int l = 0; l < 12; l++) {
            std::string pre = "text_model.encoder.layers." + std::to_string(l) + ".";
            ggml_tensor * hln = layer_norm_affine(m, xx, pre + "layer_norm1");
            ggml_tensor * qq = linear(m, hln, pre + "self_attn.q_proj");
            ggml_tensor * kk = linear(m, hln, pre + "self_attn.k_proj");
            ggml_tensor * vv = linear(m, hln, pre + "self_attn.v_proj");
            qq = ggml_scale(ec, qq, 1.0f / sqrtf((float) hd_dim));
            qq = ggml_cont(ec, ggml_permute(ec, ggml_reshape_3d(ec, qq, hd_dim, p.n_head, ids.size()), 0, 2, 1, 3));
            kk = ggml_cont(ec, ggml_permute(ec, ggml_reshape_3d(ec, kk, hd_dim, p.n_head, ids.size()), 0, 2, 1, 3));
            vv = ggml_cont(ec, ggml_permute(ec, ggml_reshape_3d(ec, vv, hd_dim, p.n_head, ids.size()), 1, 2, 0, 3));
            ggml_tensor * kq = ggml_mul_mat(ec, kk, qq);
            kq = ggml_soft_max(ec, ggml_diag_mask_inf(ec, kq, 0));
            ggml_tensor * kqv = ggml_mul_mat(ec, vv, kq);
            kqv = ggml_cont(ec, ggml_permute(ec, kqv, 0, 2, 1, 3));
            kqv = ggml_reshape_2d(ec, kqv, d, ids.size());
            xx = ggml_add(ec, xx, linear(m, kqv, pre + "self_attn.out_proj"));
            ggml_tensor * hln2 = layer_norm_affine(m, xx, pre + "layer_norm2");
            ggml_tensor * hmlp = linear(m, hln2, pre + "mlp.fc1");
            hmlp = p.quick_gelu ? ggml_gelu_quick(ec, hmlp) : ggml_gelu_erf(ec, hmlp);
            hmlp = linear(m, hmlp, pre + "mlp.fc2");
            xx = ggml_add(ec, xx, hmlp);
            ggml_set_output(xx);
            if (!compute_cpu(m, xx, 8192, [&]{ ggml_backend_tensor_set(ids2, ids.data(), 0, ids.size()*4); })) return 1;
            std::vector<float> lay((size_t)ids.size()*d);
            ggml_backend_tensor_get(xx, lay.data(), 0, lay.size()*4);
            printf("oracle layer %d: [0..3]=%.4f %.4f %.4f %.4f\n", l, lay[0],lay[1],lay[2],lay[3]);
            // reset output mark for next pass
            ggml_tensor * tmp = xx; (void)tmp;
        }
    }
    return 0;
}