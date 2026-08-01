// Slang MHA compare: drive the EXPORTED Slang mha_kern (via slangc -target cpp)
// against the oracle taes (q_perm/k_perm/v_perm/a1raw from btblock3.bin) to
// validate the actual Slang kernel, not a hand-written one.
// Usage: slang_mha_compare <btblock3.bin [oidx]>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "shaders/cpp/mha_cpu.cpp"

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

    // The oracle tensors are in ggml permuted layout: Q,K (hd,T,H); V (T,hd,H).
    // ggml pre-scales q_perm by 1/sqrt(hd); the Slang kernel scales internally,
    // so feed the UNSCALED Q (multiply by sqrt(hd)) to match production.
    std::vector<float> Q((size_t)S*C), K((size_t)S*C), V((size_t)S*C), O((size_t)S*C, 0.f);
    for (int t=0;t<S;t++)
      for (int h=0;h<H;h++)
        for (int d=0;d<hd;d++) {
            int c = h*hd + d;
            Q[(size_t)t*C+c] = q[(size_t)d + (size_t)hd*t + (size_t)hd*S*h] * sqrtf((float)hd);
            K[(size_t)t*C+c] = k[(size_t)d + (size_t)hd*t + (size_t)hd*S*h];
            V[(size_t)t*C+c] = v[(size_t)t + (size_t)S*d + (size_t)S*hd*h];
        }

    // Drive the exported Slang kernel
    GlobalParams_0 params;
    params.Q_in_0.data = Q.data();   params.Q_in_0.count = Q.size();
    params.K_in_0.data = K.data();   params.K_in_0.count = K.size();
    params.V_in_0.data = V.data();   params.V_in_0.count = V.size();
    params.O_out_0.data = O.data();  params.O_out_0.count = O.size();
    params.TQ_0 = S; params.TK_0 = S; params.C_0 = C; params.H_0 = H;
    ComputeVaryingInput vi;
    vi.startGroupID.x = vi.startGroupID.y = vi.startGroupID.z = 0;
    vi.endGroupID.x = (S + 7)/8;
    vi.endGroupID.y = (C + 7)/8;
    vi.endGroupID.z = 1;
    mha_kern(&vi, nullptr, &params);

    // mine O is token-major (T,C). a1raw oracle is (C,T) with c=d+hd*h.
    // Map: a1raw[c + C*t] with c=d+hd*h; mine[t*C+c'] c'=h*hd+d.
    // These differ only by head-major vs dim-major within head. Verify both.
    double max_abs=0, sum_abs=0; size_t cnt=(size_t)S*C;
    double cos=0,na=0,nb=0;
    for (int t=0;t<S;t++)
      for (int h=0;h<H;h++)
        for (int d=0;d<hd;d++) {
            int c_mine = h*hd+d;
            int c_ora  = d+hd*h;   // same value, c=d+hd*h
            // a1raw index = c_ora + C*t
            double mv=O[(size_t)t*C+c_mine], ov=a1[(size_t)c_ora + (size_t)C*t];
            double dd=fabs(mv-ov); if(dd>max_abs)max_abs=dd; sum_abs+=dd;
            cos+=mv*ov; na+=mv*mv; nb+=ov*ov;
        }
    cos/=sqrt(na*nb);
    printf("Slang MHA kernel vs ggml oracle (T=%d C=%d heads=%d):\n", S, C, H);
    printf("  max_abs=%.6f mean=%.6f cosine=%.6f\n", max_abs, sum_abs/cnt, cos);
    printf("%s\n", max_abs < 1e-4 && cos > 0.999 ? "VALIDATION PASS — Slang mha_kern matches ggml attention"
                   : "VALIDATION FAIL");
    return (max_abs < 1e-4 && cos > 0.999) ? 0 : 1;
}
