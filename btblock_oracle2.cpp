// BasicTransformerBlock oracle v2: replicate basic_transformer_block +
// attn_tokens manually with INTERMEDIATE taps (LN1, attn1, LN2, attn2, final)
// so we can pinpoint exactly where our direct implementation diverges.
// Usage: btblock_oracle2 <layerdiff-unet.gguf> <out.bin> [S F seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "test_common.h"
#include "unet_frame.h"

// dump everything needed: q_perm,k_perm,v_perm,kq,kqsoft,kqv_raw,a1raw with shapes
static ggml_tensor * attn_dump(Model & m, ggml_tensor * q_src, ggml_tensor * kv_src,
                               const std::string & pre, int n_head,
                               ggml_tensor * out_taps[], int & ntap) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = q_src->ne[0], Tq = q_src->ne[1], B = q_src->ne[2];
    const int64_t Tk = kv_src->ne[1];
    const int64_t hd = C / n_head;
    ggml_tensor * q = linear(m, q_src, pre + ".to_q");
    ggml_tensor * k = linear(m, kv_src, pre + ".to_k");
    ggml_tensor * v = linear(m, kv_src, pre + ".to_v");
    q = ggml_scale(ctx, q, 1.0f / sqrtf((float) hd));
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, Tq, B), 0, 2, 1, 3));
    out_taps[ntap++]=q;
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, Tk, B), 0, 2, 1, 3));
    out_taps[ntap++]=k;
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, Tk, B), 1, 2, 0, 3));
    out_taps[ntap++]=v;
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    out_taps[ntap++]=kq;
    kq = ggml_soft_max(ctx, kq);
    out_taps[ntap++]=kq;
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
    out_taps[ntap++]=kqv;
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    kqv = ggml_reshape_3d(ctx, kqv, C, Tq, B);
    out_taps[ntap++]=kqv;
    ggml_tensor * o = linear(m, kqv, pre + ".to_out.0");
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.gguf out.bin [S F seed]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int S = 8, F = 1, seedv = 1;
    if (argc >= 6) { S = atoi(argv[3]); F = atoi(argv[4]); seedv = atoi(argv[5]); }
    const int C = 1280, Tk = 77, EHD = 2048, n_head = 20;

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    init_graph_ctx(m, 262144);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, S, F);
    ggml_set_input(x);

    const std::string pre = "mid_block.attentions.0.transformer_blocks.0";
    ggml_tensor * ataps[32]; int nta=0;
    ggml_tensor * attn1 = attn_dump(m, x, x, pre + ".attn1", n_head, ataps, nta);

    for (int i=0;i<nta;i++) ggml_set_output(ataps[i]);
    ggml_set_output(attn1);
    std::vector<ggml_tensor *> outs; for(int i=0;i<nta;i++) outs.push_back(ataps[i]); outs.push_back(attn1);

    std::vector<float> xd((size_t)C*S*F);
    uint32_t s2 = seedv*12345u;
    auto rnd=[&s2](){ s2=s2*1664525u+1013904223u; return ((s2>>8)&0xffff)/32767.5f*0.4f; };
    for (auto&v:xd) v=rnd();

    if (!compute_cpu_multi(m, outs, 262144, [&]{
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
    })) return 1;

    FILE * f = fopen(argv[2], "wb");
    int32_t hdr[6] = { C, S, F, Tk, EHD, nta };
    fwrite(hdr, 4, 6, f);
    fwrite(xd.data(), 4, xd.size(), f);
    std::vector<float> ed_dummy((size_t)EHD*Tk*F, 0.f);
    fwrite(ed_dummy.data(), 4, ed_dummy.size(), f);
    static const char * names[] = {"q_perm","k_perm","v_perm","kq","kqsoft","kqv_raw","kqv(a1raw)"};
    for (int i = 0; i < nta; i++) {
        ggml_tensor * t = ataps[i];
        size_t n = 1; for (int d=0;d<(int)ggml_n_dims(t);d++) n *= (size_t)t->ne[d];
        printf("%s ne=[%lld,%lld,%lld,%lld] n=%zu\n", names[i],
               (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3], n);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n*4);
        fwrite(buf.data(), 4, n, f);
        printf("   [0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7]);
    }
    fclose(f);
    printf("wrote %s\n", argv[2]);
    return 0;
}
