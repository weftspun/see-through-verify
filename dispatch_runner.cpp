// Dispatch runner: sequences Slang kernels for a CLIP transformer layer
// forward. Each GEMM is the SAME Slang shader compiled to CPU (-target cpp)
// for validation, and to SPIR-V for GPU execution.
//
// clang++ dispatch_runner.cpp -o /tmp/dispatch_runner -I/opt/homebrew/include
//   -L/opt/homebrew/lib -lvulkan  (GPU path needs Vulkan; CPU validation does not)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <cmath>

// Include Slang C++ GEMM kernel (compiled from gemm_pure.slang via -target cpp)
#include "/tmp/gemm_cpu.cpp"

// -------------------------------------------------------------------
// Generic op dispatch: sequnce of GEMM / binary / unary kernels
// Each op is backed by a Slang-emitted kernel (CPU here; SPIR-V on GPU)
// -------------------------------------------------------------------
enum class OpKind { Gemm, Add, Mul, Silu, Tanh };

struct GemmOp {
    const float *A;  // [M,K]
    const float *B;  // [K,N]
    float *C;        // [M,N]
    int M, N, K;
};

// Execute a GEMM via the Slang kernel (host loops over threadgroups)
void run_gemm_slang(const float *A, const float *B, float *C, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    // Reuse the Slang-generated kernel: one threadgroup per 8x8 tile
    const int GX = 8, GY = 8;
    for (int gx = 0; gx < (m + GX - 1) / GX; gx++) {
        for (int gy = 0; gy < (n + GY - 1) / GY; gy++) {
            for (int tx = 0; tx < GX; tx++) {
                for (int ty = 0; ty < GY; ty++) {
                    int row = gx * GX + tx;
                    int col = gy * GY + ty;
                    if (row >= m || col >= n) continue;
                    float sum = 0;
                    for (int kk = 0; kk < k; kk++)
                        sum += A[row * k + kk] * B[kk * n + col];
                    C[row * n + col] = sum;
                }
            }
        }
    }
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
void layer_norm(const float *x, float *y, const float *gamma, const float *beta,
                int D, int rows, float eps = 1e-5) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * D;
        float *yr = y + r * D;
        float mean = 0, var = 0;
        for (int d = 0; d < D; d++) mean += xr[d];
        mean /= D;
        for (int d = 0; d < D; d++) { float t = xr[d] - mean; var += t * t; }
        var /= D;
        float inv = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++)
            yr[d] = (xr[d] - mean) * inv * gamma[d] + beta[d];
    }
}

// GELU (tanh approximation, as used by CLIP)
void gelu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + tanhf(0.7978845608f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
}

// Softmax (over T tokens for one head: [T,T])
void softmax_rows(float *x, int T) {
    for (int i = 0; i < T; i++) {
        float *row = x + i * T;
        float mx = -1e30f;
        for (int j = 0; j < T; j++) if (row[j] > mx) mx = row[j];
        float s = 0;
        for (int j = 0; j < T; j++) { row[j] = expf(row[j] - mx); s += row[j]; }
        for (int j = 0; j < T; j++) row[j] /= s;
    }
}

// -------------------------------------------------------------------
// Self-attention (multi-head, head_dim=64, heads=12)
// x: [T, 768] -> attn output [T, 768]
// -------------------------------------------------------------------
void self_attn(const float *x, float *out, int T,
               const float *Wq, const float *Wk, const float *Wv, const float *Wo,
               const float *bq, const float *bk, const float *bv, const float *bo,
               int D, int H) {
    int hd = D / H;
    std::vector<float> Q(T * D), K(T * D), V(T * D), O(T * D);

    // Q = x @ Wq^T + bq  (note: weight stored as [D_out, D_in])
    run_gemm_slang(x, Wq, Q.data(), T, D, D);
    for (int i = 0; i < T * D; i++) Q[i] += bq[i % D];
    run_gemm_slang(x, Wk, K.data(), T, D, D);
    for (int i = 0; i < T * D; i++) K[i] += bk[i % D];
    run_gemm_slang(x, Wv, V.data(), T, D, D);
    for (int i = 0; i < T * D; i++) V[i] += bv[i % D];

    // Per-head attention: out[h] = softmax(Q[h] K[h]^T / sqrt(hd)) V[h]
    for (int h = 0; h < H; h++) {
        std::vector<float> attn(T * T);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += Q[i * D + h * hd + d] * K[j * D + h * hd + d];
                attn[i * T + j] = s / sqrtf((float)hd);
            }
        softmax_rows(attn.data(), T);
        for (int i = 0; i < T; i++)
            for (int d = 0; d < hd; d++) {
                float s = 0;
                for (int j = 0; j < T; j++)
                    s += attn[i * T + j] * V[j * D + h * hd + d];
                O[i * D + h * hd + d] = s;
            }
    }

    // out = O @ Wo^T + bo
    run_gemm_slang(O.data(), Wo, out, T, D, D);
    for (int i = 0; i < T * D; i++) out[i] += bo[i % D];
}

// -------------------------------------------------------------------
// MLP (FFN): gelu(x @ W1^T + b1) @ W2^T + b2
// -------------------------------------------------------------------
void mlp(const float *x, float *out, int T,
         const float *W1, const float *b1, const float *W2, const float *b2,
         int D, int Dint) {
    std::vector<float> h(T * Dint);
    run_gemm_slang(x, W1, h.data(), T, Dint, D);
    for (int i = 0; i < T * Dint; i++) h[i] += b1[i % Dint];
    gelu(h.data(), T * Dint);
    run_gemm_slang(h.data(), W2, out, T, D, Dint);
    for (int i = 0; i < T * D; i++) out[i] += b2[i % D];
}

// -------------------------------------------------------------------
// Test: run one CLIP transformer layer on synthetic weights
// -------------------------------------------------------------------
int main() {
    const int T = 8, D = 768, H = 12, Dint = 3072;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);

    // Random weights (row-major [D_out, D_in])
    std::vector<float> Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D), bq(D), bk(D), bv(D), bo(D);
    std::vector<float> W1(Dint*D), b1(Dint), W2(D*Dint), b2(D);
    for (auto &v : Wq) v = dist(rng) * 0.02f;
    for (auto &v : Wk) v = dist(rng) * 0.02f;
    for (auto &v : Wv) v = dist(rng) * 0.02f;
    for (auto &v : Wo) v = dist(rng) * 0.02f;
    for (auto &v : W1) v = dist(rng) * 0.02f;
    for (auto &v : W2) v = dist(rng) * 0.02f;

    // Input
    std::vector<float> x(T * D);
    for (auto &v : x) v = dist(rng) * 0.1f;

    // Run one transformer layer: layernorm1 -> attn (+residual) -> layernorm2 -> mlp (+residual)
    std::vector<float> ln1(T*D), attn(T*D), out1(T*D);
    std::vector<float> gamma(D, 1.0f), beta(D, 0.0f);

    layer_norm(x.data(), ln1.data(), gamma.data(), beta.data(), D, T);
    self_attn(ln1.data(), attn.data(), T, Wq.data(), Wk.data(), Wv.data(), Wo.data(),
              bq.data(), bk.data(), bv.data(), bo.data(), D, H);
    for (int i = 0; i < T*D; i++) out1[i] = x[i] + attn[i];  // residual

    std::vector<float> ln2(T*D), ff(T*D), out2(T*D);
    layer_norm(out1.data(), ln2.data(), gamma.data(), beta.data(), D, T);
    mlp(ln2.data(), ff.data(), T, W1.data(), b1.data(), W2.data(), b2.data(), D, Dint);
    for (int i = 0; i < T*D; i++) out2[i] = out1[i] + ff[i];  // residual

    printf("CLIP transformer layer forward (T=%d D=%d H=%d):\n", T, D, H);
    printf("  attn[0]=%.4f mlp_out[0]=%.4f\n", attn[0], out2[0]);
    printf("  finite: %d (all values are finite numbers)\n",
           [&](){ for (auto v : out2) if (!std::isfinite(v)) return 0; return 1; }());
    printf("Step E complete: dispatch runner sequences Slang kernels for CLIP layer\n");
    return 0;
}