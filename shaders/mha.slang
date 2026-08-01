// MHA kernel (Slang → slangc -target cpp for CPU validation).
// Per-thread over output element (i, c): out[i][c] = sum_j softmax_j(Q_i·K_j/sqrt(hd)) * V[j][c]
// Token-major Q,K,V,O in (T, C); channel c = h*hd + d (head-major). Barrier-free
// (recomputes per-head softmax per thread) so the cpp target accepts it.
[[vk::binding(0, 0)]] StructuredBuffer<float> Q_in;
[[vk::binding(1, 0)]] StructuredBuffer<float> K_in;
[[vk::binding(2, 0)]] StructuredBuffer<float> V_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> O_out;
[[vk::binding(4, 0)]] uint TQ;
[[vk::binding(5, 0)]] uint TK;
[[vk::binding(6, 0)]] uint C;
[[vk::binding(7, 0)]] uint H;

[shader("compute")] [numthreads(8, 8, 1)]
void mha_kern(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;      // query token
    uint c = dtid.y;      // channel
    if (i >= TQ || c >= C) return;
    uint h = c / (C / H);
    uint hd = C / H;
    float mx = -1e30;
    for (uint j = 0; j < TK; ++j) {
        float s = 0;
        for (uint d = 0; d < hd; ++d)
            s += Q_in[(i*C) + (h*hd + d)] * K_in[(j*C) + (h*hd + d)];
        s = s / sqrt(float(hd));
        if (s > mx) mx = s;
    }
    float denom = 0;
    for (uint j = 0; j < TK; ++j) {
        float s = 0;
        for (uint d = 0; d < hd; ++d)
            s += Q_in[(i*C) + (h*hd + d)] * K_in[(j*C) + (h*hd + d)];
        s = exp(s / sqrt(float(hd)) - mx);
        denom += s;
    }
    float acc = 0;
    for (uint j = 0; j < TK; ++j) {
        float s = 0;
        for (uint d = 0; d < hd; ++d)
            s += Q_in[(i*C) + (h*hd + d)] * K_in[(j*C) + (h*hd + d)];
        float p = exp(s / sqrt(float(hd)) - mx) / denom;
        acc += p * V_in[(j*C) + c];
    }
    O_out[(i*C) + c] = acc;
}
