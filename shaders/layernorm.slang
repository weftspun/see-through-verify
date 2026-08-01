// LayerNorm kernel (Slang -> slangc -target cpp for CPU validation).
// Per-thread over (token t, dim d): y[t][d] = (x[t][d]-mean)/sqrt(var+eps) * g[d] + b[d]
// Token-major x in (T, D). Barrier-free, per-thread partial reduction over dims.
[[vk::binding(0, 0)]] StructuredBuffer<float> X_in;
[[vk::binding(1, 0)]] StructuredBuffer<float> G_in;
[[vk::binding(2, 0)]] StructuredBuffer<float> B_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y_out;
[[vk::binding(4, 0)]] uint T;
[[vk::binding(5, 0)]] uint D;
[[vk::binding(6, 0)]] float EPS;

[shader("compute")] [numthreads(8, 8, 1)]
void layernorm_kern(uint3 dtid : SV_DispatchThreadID) {
    uint t = dtid.x;
    uint d = dtid.y;
    if (t >= T || d >= D) return;
    // reduce mean/var over full row d (per token t) -- each thread recomputes
    float mean = 0;
    for (uint k = 0; k < D; ++k) mean += X_in[(t*D) + k];
    mean = mean / float(D);
    float var = 0;
    for (uint k = 0; k < D; ++k) {
        float q = X_in[(t*D) + k] - mean;
        var += q * q;
    }
    var = var / float(D);
    float inv = 1.0 / sqrt(var + EPS);
    Y_out[(t*D) + d] = (X_in[(t*D) + d] - mean) * inv * G_in[d] + B_in[d];
}
