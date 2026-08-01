// Slang GroupNorm compare: drive the EXPORTED Slang groupnorm_kern against the
// ggml group_norm oracle tap (non-affine, so G=1,B=0). Validates the kernel's
// mean/var/reduction semantics.
// Usage: slang_gn_compare <gn.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "shaders/cpp/groupnorm_cpu.cpp"
static void cmp(const char *nm, const std::vector<float>&a, const std::vector<float>&b){
    double m=0,s=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;i++){double d=fabs((double)a[i]-(double)b[i]); if(d>m)m=d; s+=d;}
    double c=0,na=0,nb=0; for(size_t i=0;i<n;i++){c+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} c/=sqrt(na*nb);
    printf("%s: max=%.6f mean=%.6f cos=%.6f\n",nm,m,s/n,c);
}
int main(int argc, char**argv){
    if(argc<2){fprintf(stderr,"usage: %s gn.bin\n",argv[0]);return 1;}
    FILE*f=fopen(argv[1],"rb");
    int32_t hdr[5]; fread(hdr,4,5,f);
    int W=hdr[0],H=hdr[1],C=hdr[2],F=hdr[3],GR=hdr[4];
    size_t n=(size_t)W*H*C*F;
    std::vector<float> ref(n); fread(ref.data(),4,n,f);
    std::vector<float> x(n); fread(x.data(),4,n,f);
    fclose(f);
    std::vector<float> G(C,1.f), B(C,0.f), out(n,0.f);
    GlobalParams_0 p;
    p.X_in_0.data=x.data(); p.X_in_0.count=n;
    p.G_in_0.data=G.data(); p.G_in_0.count=C;
    p.B_in_0.data=B.data(); p.B_in_0.count=C;
    p.Y_out_0.data=out.data(); p.Y_out_0.count=n;
    p.C_0=C; p.HW_0=(uint32_t)(W*H); p.N_GROUPS_0=GR; p.EPS_0=1e-6f; p.FCOUNT_0=F;
    ComputeVaryingInput vi;
    vi.startGroupID.x=vi.startGroupID.y=vi.startGroupID.z=0;
    // dispatch: x dim = batch F, y dim = channel C (numthreads y=8); z unused
    vi.endGroupID.x=F; vi.endGroupID.y=(C+7)/8; vi.endGroupID.z=1;
    groupnorm_kern(&vi, nullptr, &p);
    cmp("Slang groupnorm vs ggml group_norm", out, ref);
    return 0;
}
