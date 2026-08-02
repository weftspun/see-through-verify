// GEMM parity: does Accelerate-BLAS token_linear (verify_common.h) match the
// scalar reference exactly? This validates the token_linear soundness across
// random shapes typical of the transformer stack (attention QKV, GEGLU, etc).
// Usage: parity_gemm
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "verify_common.h"

static void token_linear_scalar(const float *x, const float *W, const float *b, float *y,
                                int T, int Cin, int Cout) {
    for (int t = 0; t < T; t++)
        for (int o = 0; o < Cout; o++) {
            float s = b ? b[o] : 0.f;
            for (int k = 0; k < Cin; k++) s += x[t*Cin+k] * W[o*Cin+k];
            y[t*Cout+o] = s;
        }
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f,1.f);
    int bad=0;
    // shapes: (T, Cin, Cout) covering transformer GEMMs
    const int shapes[][3] = {
        {8,1280,1280},   // attention Q/K/V
        {8,1280,5120},   // GEGLU proj up
        {8,5120,1280},   // GEGLU proj down
        {77,2048,1280},  // cross-attn from EHD
        {8,1280,77},     // (small) inverse-ish
        {1,1280,1280},
        {13,1280,2560},
    };
    for (auto &sh : shapes) {
        int T=sh[0],Cin=sh[1],Cout=sh[2];
        std::vector<float> x(T*Cin), W(Cout*Cin), b(Cout), A(T*Cout), B(T*Cout);
        for(auto&v:x)v=U(rng); for(auto&v:W)v=U(rng); for(auto&v:b)v=U(rng);
        token_linear(x.data(), W.data(), b.data(), A.data(), T, Cin, Cout);   // BLAS
        token_linear_scalar(x.data(), W.data(), b.data(), B.data(), T, Cin, Cout);
        double m=0,c=0,na=0,nb=0; for(int i=0;i<T*Cout;i++){ double dd=fabs((double)A[i]-(double)B[i]); if(dd>m)m=dd; c+=(double)A[i]*B[i];na+=(double)A[i]*A[i];nb+=(double)B[i]*B[i];} c/=sqrt(na*nb);
        bool ok = c>0.999999 && m<1e-4;
        if(!ok) bad++;
        printf("T=%-3d C=%-5d Cout=%-5d  BLAS-vs-scalar  max=%.2e cos=%.9f  %s\n",
               T,Cin,Cout, m, c, ok?"OK":"MISMATCH");
    }
    printf("== %s ==\n", bad==0?"GEMM PARITY GREEN":"GEMM PARITY FAIL");
    return bad==0?0:1;
}
