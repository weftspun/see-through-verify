// Validate the actual Slang-generated gemm() entry point against a reference
// Proves slangc -target cpp produces numerically correct kernels
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include "/tmp/gemm_cpu.cpp"

int main() {
    const int M = 64, N = 64, K = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> A(M*K), B(K*N), C(M*N, 0.0f), C_ref(M*N);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C_ref[i*N+j] = s;
        }

    // Call the actual Slang-generated gemm() entry point
    GlobalParams_0 params;
    params.A_0.data = A.data(); params.A_0.count = A.size();
    params.B_0.data = B.data(); params.B_0.count = B.size();
    params.C_0.data = C.data(); params.C_0.count = C.size();
    params.M_0 = M; params.N_0 = N; params.K_0 = K;

    ComputeVaryingInput vi;
    vi.startGroupID.x = 0; vi.startGroupID.y = 0; vi.startGroupID.z = 0;
    vi.endGroupID.x = M / 8; vi.endGroupID.y = N / 8; vi.endGroupID.z = 1;
    gemm(&vi, nullptr, &params);

    double max_err = 0, max_val = 0;
    for (int i = 0; i < M*N; i++) {
        double e = fabs((double)C[i] - (double)C_ref[i]);
        double v = fabs((double)C_ref[i]);
        if (e > max_err) max_err = e;
        if (v > max_val) max_val = v;
    }
    printf("Slang-generated gemm() entry point %dx%dx%d: max_err=%.6f (%.4f%%)\n",
           M, N, K, max_err, max_val ? max_err/max_val*100 : 0);
    printf("%s\n", max_err < 1e-4 ? "GREEN - actual Slang kernel correct" : "RED - mismatch");
    return max_err < 1e-4 ? 0 : 1;
}