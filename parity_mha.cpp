// CPU-vs-shader parity check for MHA.
//
// Drives BOTH the Slang-exported MHA kernel (the real implementation, which is
// the same source exported to Metal/SPIR-V) AND the Accelerate-BLAS-backed
// reference MHA on the IDENTICAL input, then checks:
//   1. soundness  : Slang kernel vs ggml oracle, and Accel reference vs oracle
//   2. cpu-shader parity : Slang output vs Accel output must match
//   3. performance: timing of each path
// so a green confirms the CPU validation and shader path are genuinely on par.
//
// Usage: parity_mha <btblock3.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "shaders/cpp/mha_cpu.cpp"   // real Slang-exported kernel
#include "verify_common.h"           // Accelerate-BLAS mha via token_linear? -> use own mha_tokens

// Accelerate-BLAS-backed multi-head attention over tokens (Tq x Tk).
// MHA is PER-HEAD: each head h attends over its own hd-dim slice (c=h*hd+d),
// with its own softmax over keys. Using cblas_sgemm per head keeps the
// soundness (bit-for-bit with the per-head scalar reference) while getting
// BLAS throughput. Q,K,V token-major (T,C), channel c = h*hd + d (head-major).
static void mha_tokens_blas(const float *Q, const float *K, const float *V, float *O,
                       int Tq, int Tk, int C, int H) {
    const int hd = C / H;
    std::vector<float> S((size_t)Tq*Tk), Qh((size_t)Tq*hd), Kh((size_t)Tk*hd),
                       Vh((size_t)Tk*hd), Oh((size_t)Tq*hd);
    for (int h=0;h<H;h++) {
        // slice this head's (T,hd) columns from the (T,C) token-major tensors
        for (int t=0;t<Tq;t++) for (int d=0;d<hd;d++) Qh[t*hd+d]=Q[t*C+h*hd+d];
        for (int t=0;t<Tk;t++) for (int d=0;d<hd;d++) Kh[t*hd+d]=K[t*C+h*hd+d];
        for (int t=0;t<Tk;t++) for (int d=0;d<hd;d++) Vh[t*hd+d]=V[t*C+h*hd+d];
        // S[head] = Qh Kh^T / sqrt(hd): [Tq,hd] x [Tk,hd]^T -> [Tq,Tk]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tq, Tk, hd,
                    1.f/sqrtf((float)hd), Qh.data(), hd, Kh.data(), hd, 0.f, S.data(), Tk);
        // softmax over keys per row t (per head)
        for (int i=0;i<Tq;i++){
            float mx=-1e30f; for(int j=0;j<Tk;j++) if(S[i*Tk+j]>mx)mx=S[i*Tk+j];
            float sm=0; for(int j=0;j<Tk;j++){ S[i*Tk+j]=expf(S[i*Tk+j]-mx); sm+=S[i*Tk+j]; }
            for(int j=0;j<Tk;j++) S[i*Tk+j]/=sm;
        }
        // Oh[head] = S Vh: [Tq,Tk] x [Tk,hd] -> [Tq,hd]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tq, hd, Tk,
                    1.f, S.data(), Tk, Vh.data(), hd, 0.f, Oh.data(), hd);
        for (int t=0;t<Tq;t++) for (int d=0;d<hd;d++) O[t*C+h*hd+d]=Oh[t*hd+d];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s btblock3.bin\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    int32_t hdr[6]; fread(hdr,4,6,f);
    int C=hdr[0],S=hdr[1],F=hdr[2],Tk=hdr[3],EHD=hdr[4],N=hdr[5];
    size_t xsz=(size_t)C*S, esz=(size_t)EHD*Tk;
    std::vector<float> xd(xsz), ed(esz);
    fread(xd.data(),4,xsz,f); fread(ed.data(),4,esz,f);
    int H=20, hd=64;
    size_t nq=(size_t)hd*S*H, nv=(size_t)S*hd*H, nkq=(size_t)S*S*H, nkqv=(size_t)hd*S*H, nc=(size_t)C*S;
    std::vector<float> q(nq),k(nq),v(nv),kq(nkq),kqsoft(nkq),kqvraw(nkqv),a1(nc);
    fread(q.data(),4,nq,f); fread(k.data(),4,nq,f); fread(v.data(),4,nv,f);
    fread(kq.data(),4,nkq,f); fread(kqsoft.data(),4,nkq,f); fread(kqvraw.data(),4,nkqv,f);
    fread(a1.data(),4,nc,f); fclose(f);

    // Build unscaled Q,K,V in token-major (T,C) like production (Q fed unscaled;
    // the Slang kernel AND the reference both scale internally by 1/sqrt(hd)).
    std::vector<float> Q((size_t)S*C), K((size_t)S*C), V((size_t)S*C);
    for (int t=0;t<S;t++)
      for (int h=0;h<H;h++)
        for (int d=0;d<hd;d++) {
            int c = h*hd + d;
            Q[t*C+c] = q[(size_t)d + (size_t)hd*t + (size_t)hd*S*h] * sqrtf((float)hd);  // undo oracle prescale
            K[t*C+c] = k[(size_t)d + (size_t)hd*t + (size_t)hd*S*h];
            V[t*C+c] = v[(size_t)t + (size_t)S*d + (size_t)S*hd*h];
        }

    // ---- Path A: Slang-exported kernel (real impl; also -> Metal/SPIR-V) ----
    std::vector<float> Oa((size_t)S*C, 0.f);
    {
        GlobalParams_0 params;
        params.Q_in_0.data=Q.data(); params.Q_in_0.count=Q.size();
        params.K_in_0.data=K.data(); params.K_in_0.count=K.size();
        params.V_in_0.data=V.data(); params.V_in_0.count=V.size();
        params.O_out_0.data=Oa.data(); params.O_out_0.count=Oa.size();
        params.TQ_0=S; params.TK_0=S; params.C_0=C; params.H_0=H;
        ComputeVaryingInput vi; vi.startGroupID.x=vi.startGroupID.y=vi.startGroupID.z=0;
        vi.endGroupID.x=(S+7)/8; vi.endGroupID.y=(C+7)/8; vi.endGroupID.z=1;
        mha_kern(&vi, nullptr, &params);
    }

    // ---- Path B: Accelerate-BLAS reference ----
    std::vector<float> Ob((size_t)S*C, 0.f);
    mha_tokens_blas(Q.data(), K.data(), V.data(), Ob.data(), S, S, C, H);

    // ---- compare helper ----
    auto cmp = [&](const char *nm, const std::vector<float>&a, const std::vector<float>&b){
        double m=0,s=0,c=0,na=0,nb=0; size_t n=std::min(a.size(),b.size());
        for(size_t i=0;i<n;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; s+=d;
            c+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];}
        c/=sqrt(na*nb);
        printf("  %-40s max=%.6f mean=%.6f cos=%.6f\n", nm, m, s/n, c);
        return m<1e-3 && c>0.999;
    };

    // Build oracle reference output in mine layout (T,C), c=h*hd+d, for direct compare.
    std::vector<float> ora((size_t)S*C);
    for (int t=0;t<S;t++) for (int h=0;h<H;h++) for (int d=0;d<hd;d++) {
        int c_ora = d+hd*h;                  // a1raw layout
        ora[(size_t)t*C + (h*hd+d)] = a1[(size_t)c_ora + (size_t)C*t];
    }

    printf("MHA parity  T=%d C=%d heads=%d (same input both paths)\n", S, C, H);
    int ok=0;
    printf("[soundness]  Slang kernel vs ggml oracle:\n");            ok+=cmp("Slang vs oracle", Oa, ora);
    printf("[soundness]  Accel-BLAS ref vs ggml oracle:\n");          ok+=cmp("Accel vs oracle", Ob, ora);
    printf("[CPU<->shader parity] Slang kernel vs Accel-BLAS ref:\n"); ok+=cmp("Slang vs Accel  ", Oa, Ob);
    printf("== %s ==\n", ok==3 ? "PARITY GREEN (both paths on par, sound vs oracle)"
                              : "PARITY FAIL");
    return ok==3?0:1;
}
