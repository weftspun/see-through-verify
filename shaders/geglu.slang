// GEGLU (Slang -> slangc -target cpp for CPU validation). Matches ggml geglu_ff.
// Two kernels:
//   geglu_proj: glu[t][c] = p[c]*gelu_erf(p[g+c]), p = W1*x+B1, W1 is [2g, Cin],
//               output glu buffer (T, g).
//   geglu_ff:   out[t][o] = B2[o] + sum_c glu[t][c]*W2[o][c]  (W2 [Cout, g])
[[vk::binding(0, 0)]] StructuredBuffer<float> X_in;
[[vk::binding(1, 0)]] StructuredBuffer<float> W1_in;   // [2g, Cin]
[[vk::binding(2, 0)]] StructuredBuffer<float> B1_in;   // [2g]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> GLU_out; // [T, g]
[[vk::binding(4, 0)]] uint T;
[[vk::binding(5, 0)]] uint CIN;
[[vk::binding(6, 0)]] uint G;

float erf_(float x) {
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    float sign = (x < 0.0) ? -1.0 : 1.0;
    return sign * y;
}
float gelu_erf(float x) { return 0.5 * x * (1.0 + erf_(x / 1.4142135623730951)); }

[shader("compute")] [numthreads(8, 8, 1)]
void geglu_proj(uint3 dtid : SV_DispatchThreadID) {
    uint t = dtid.x, c = dtid.y;
    if (t >= T || c >= G) return;
    float pv = B1_in[c], pg = B1_in[G + c];
    for (uint cin = 0; cin < CIN; ++cin) {
        float xv = X_in[(t*CIN) + cin];
        pv += xv * W1_in[(c*CIN) + cin];
        pg += xv * W1_in[((G + c)*CIN) + cin];
    }
    GLU_out[(t*G) + c] = pv * gelu_erf(pg);
}

[[vk::binding(0, 0)]] StructuredBuffer<float> GLU2_in;   // [T, g]
[[vk::binding(1, 0)]] StructuredBuffer<float> W2_in;     // [Cout, g]
[[vk::binding(2, 0)]] StructuredBuffer<float> B2_in;     // [Cout]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y_out;   // [T, Cout]
[[vk::binding(4, 0)]] uint T2;
[[vk::binding(5, 0)]] uint G2;
[[vk::binding(6, 0)]] uint COUT2;

[shader("compute")] [numthreads(8, 8, 1)]
void geglu_ff(uint3 dtid : SV_DispatchThreadID) {
    uint t = dtid.x, o = dtid.y;
    if (t >= T2 || o >= COUT2) return;
    float acc = B2_in[o];
    for (uint k = 0; k < G2; ++k)
        acc += GLU2_in[(t*G2) + k] * W2_in[(o*G2) + k];
    Y_out[(t*COUT2) + o] = acc;
}
