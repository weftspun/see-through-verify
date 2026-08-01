// GEGLU compare: our direct erf-GELU implementation of diffusers GEGLU
// against the ggml_geglu_erf_swapped oracle. Input has [2g, T]; output [g, T].
// diffusers: out[c] = value[c] * gelu(gate[c]) where gate is in the SECOND
// half of the projection (ggml_geglu_erf_swapped with switched halves).
// Usage: geglu_compare <oracle.bin.in> <oracle.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static float gelu_erf(float x) { return 0.5f*x*(1.0f+erff(x/1.41421356f)); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { fprintf(stderr, "no %s\n", argv[1]); return 1; }
    long sz; fseek(fi,0,SEEK_END); sz=ftell(fi); fseek(fi,0,SEEK_SET);
    std::vector<float> x(sz/4); fread(x.data(),4,x.size(),fi); fclose(fi);
    FILE *fo = fopen(argv[2], "rb");
    int32_t hdr[3]; fread(hdr,4,3,fo);
    int g=hdr[0], T=hdr[1];
    size_t n=(size_t)g*T;
    std::vector<float> ref(n); fread(ref.data(),4,n,fo); fclose(fo);

    // ggml_geglu_erf_swapped: x halves swapped so src0(gelu input)=second half,
    // src1(multiplier)=first half. So out[c] = gelu(x[c+g]) * x[c].
    std::vector<float> mine(n);
    for (int t=0;t<T;t++)
        for (int c=0;c<g;c++) {
            float value   = x[(size_t)t*2*g + c];
            float gate    = x[(size_t)t*2*g + g + c];
            mine[(size_t)t*g + c] = value * gelu_erf(gate);
        }
    double max_abs=0, sum_abs=0; size_t cnt=n;
    for (size_t i=0;i<cnt;i++){ double dd=fabs((double)mine[i]-(double)ref[i]); if(dd>max_abs)max_abs=dd; sum_abs+=dd; }
    double cos=0,na=0,nb=0;
    for (size_t i=0;i<cnt;i++){cos+=(double)mine[i]*ref[i]; na+=(double)mine[i]*mine[i]; nb+=(double)ref[i]*ref[i];}
    cos/=sqrt(na*nb);
    printf("mine[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", mine[0],mine[1],mine[2],mine[3],mine[4],mine[5]);
    printf("ora [0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5]);
    printf("geglu %dx%d: max_abs_diff=%.6f mean=%.6f cosine=%.6f\n", g, T, max_abs, sum_abs/cnt, cos);
    printf("%s\n", max_abs < 1e-5 && cos > 0.9999 ? "VALIDATION PASS — GEGLU matches ggml oracle" : "VALIDATION FAIL");
    return (max_abs < 1e-5 && cos > 0.9999) ? 0 : 1;
}
