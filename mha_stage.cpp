// Focused mha validation: feed the oracle's EXACT q_perm/k_perm/v_perm
// tensors into our mha_tokens and check against the oracle's a1raw.
// Proves whether the C++ attention kernel is right independent of weights.
// Usage: mha_stage <btblock3.bin>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static void mha_tokens(const float *Q, const float *K, const float *V, float *O,
                       int Tq, int Tk, int C, int H, int hd) {
    // Q,K in (hd,T,H); V in (T,hd,H); O in (hd,T,H) -> caller reshapes.
    std::vector<float> sc(Tq*Tk);
    for (int h=0;h<H;h++) {
        // scores for this head
        for (int i=0;i<Tq;i++)
          for (int j=0;j<Tk;j++) {
              float s=0;
              for (int d=0;d<hd;d++)
                  s += Q[d+hd*i+hd*Tq*h] * K[d+hd*j+hd*Tk*h];
              sc[i*Tk+j] = s;
          }
        // softmax over keys j per query i
        for (int i=0;i<Tq;i++) {
            float mx=-1e30f;
            for (int j=0;j<Tk;j++) if (sc[i*Tk+j]>mx) mx=sc[i*Tk+j];
            float sm=0; for (int j=0;j<Tk;j++){ sc[i*Tk+j]=expf(sc[i*Tk+j]-mx); sm+=sc[i*Tk+j]; }
            for (int j=0;j<Tk;j++) sc[i*Tk+j]/=sm;
        }
        // PV for this head, writing O (hd,Tq,H)
        for (int i=0;i<Tq;i++)
          for (int d=0;d<hd;d++) {
              float s=0;
              for (int j=0;j<Tk;j++)
                  s += sc[i*Tk+j] * V[j + Tk*d + Tk*hd*h];
              O[d+hd*i+hd*Tq*h] = s;
          }
    }
}

int main(int argc, char **argv) {
    if (argc<2){fprintf(stderr,"usage: %s btblock3.bin\n",argv[0]);return 1;}
    FILE*f=fopen(argv[1],"rb");
    int32_t hdr[6];fread(hdr,4,6,f);
    int C=hdr[0],S=hdr[1],F=hdr[2],Tk=hdr[3],EHD=hdr[4],N=hdr[5];
    size_t xsz=(size_t)C*S, esz=(size_t)EHD*Tk;
    std::vector<float> xd(xsz),ed(esz);
    fread(xd.data(),4,xsz,f); fread(ed.data(),4,esz,f);
    int H=20,hd=64;
    size_t nq=(size_t)hd*S*H, nv=(size_t)S*hd*H, nkq=(size_t)S*S*H, nkqv=(size_t)hd*S*H, nc=(size_t)C*S;
    std::vector<float> q(nq),k(nq),v(nv),kq(nkq),kqsoft(nkq),kqvraw(nkqv),a1(nc);
    fread(q.data(),4,nq,f); fread(k.data(),4,nq,f); fread(v.data(),4,nv,f);
    fread(kq.data(),4,nkq,f); fread(kqsoft.data(),4,nkq,f); fread(kqvraw.data(),4,nkqv,f);
    fread(a1.data(),4,nc,f); fclose(f);

    std::vector<float> out(nkqv);
    mha_tokens(q.data(), k.data(), v.data(), out.data(), S, S, C, H, hd);

    // Also recompute kq and softmax to compare against oracle's
    double kqmax=0;
    for (int ti=0;ti<S;ti++) for(int tj=0;tj<S;tj++) for(int h=0;h<H;h++){
        float s=0;
        for(int d=0;d<hd;d++) s+=q[d+hd*ti+hd*S*h]*k[d+hd*tj+hd*S*h];
        size_t ix=tj+S*ti+S*S*h; // oracle kq ne0=tk(tj), ne1=tq(ti)
        double dd=fabs((double)s-(double)kq[ix]);
        if(dd>kqmax)kqmax=dd;
    }
    printf("recomputed kq vs oracle kq max=%.6f\n", kqmax);
    // compare my softmax to oracle kqsoft
    auto tj2ix = [&](int ti,int tj,int h){ return tj + S*ti + S*S*h; };
    double smx=0;
    for (int ti=0;ti<S;ti++) for(int tj=0;tj<S;tj++) for(int h=0;h<H;h++){
        float mx=-1e30f;
        for (int jj=0;jj<S;jj++){ float v=kq[tj2ix(ti,jj,h)]; if(v>mx)mx=v; }
        float s=0; std::vector<float> ex(S);
        for (int jj=0;jj<S;jj++){ ex[jj]=expf(kq[tj2ix(ti,jj,h)]-mx); s+=ex[jj]; }
        size_t ix=tj+S*ti+S*S*h;
        smx=std::max(smx,(double)fabs((double)(ex[tj]/s)-(double)kqsoft[ix]));
    }
    printf("my softmax vs oracle kqsoft max=%.6f\n", smx);
    // verify V decode: V ne=[8,64,20]=(T,hd,H) index=t+S*d+S*hd*h
    double vmax=0;
    for (int t=0;t<S;t++) for(int d=0;d<hd;d++) for(int h=0;h<H;h++){
        // nothing to compare against externally; just sanity print
        if (t==0 && d==0) printf("V(t0,d0,h0)=%f V(t0,d0,h1)=%f\n", v[0], v[0+0+ S*hd*1]);
    }
    printf("V size=%zu ne_total=%zu\n", v.size(), (size_t)S*hd*H);
    // replicate PV manually and print a few
    double pvmax=0;
    {
        for (int ti=0;ti<S;ti++) for(int h=0;h<H;h++) {
            // soft for query ti
            float mx=-1e30f;
            for (int jj=0;jj<S;jj++){ float vv=kq[tj2ix(ti,jj,h)]; if(vv>mx)mx=vv; }
            std::vector<float> ex(S);
            float s=0; for (int jj=0;jj<S;jj++){ ex[jj]=expf(kq[tj2ix(ti,jj,h)]-mx); s+=ex[jj]; }
            for (int d=0;d<hd;d++) {
                float acc=0;
                for (int jj=0;jj<S;jj++) acc += (ex[jj]/s) * v[jj + S*d + S*hd*h];
                double dd=fabs((double)acc-(double)kqvraw[d+hd*ti+hd*S*h]);
                if (dd>pvmax) pvmax=dd;
            }
        }
    }
    printf("manual PV vs kqv_raw max=%.6f\n", pvmax);
    // compare out to kqv_raw
    double max=0,sum=0; size_t n=nkqv;
    for (size_t i=0;i<n;i++){double dd=fabs((double)out[i]-(double)kqvraw[i]); if(dd>max)max=dd; sum+=dd;}
    double cos=0,na=0,nb=0;
    for (size_t i=0;i<n;i++){cos+=(double)out[i]*kqvraw[i]; na+=(double)out[i]*out[i]; nb+=(double)kqvraw[i]*kqvraw[i];}
    cos/=sqrt(na*nb);
    printf("mha vs kqv_raw: max=%.6f mean=%.6f cos=%.6f\n",max,sum/n,cos);
    // also compare vs a1raw (C,T) reshape: c = d+hd*h
    double m2=0;
    for (int t=0;t<S;t++) for(int h=0;h<H;h++) for(int d=0;d<hd;d++){
        size_t i=d+hd*t+hd*S*h; int c=d+hd*h; size_t j=(size_t)c+C*t;
        m2=std::max(m2,(double)fabs((double)out[i]-(double)a1[j]));
    }
    printf("mha vs a1raw(c=d+hd*h): max=%.6f\n", m2);
    printf("kqraw[0]=%.4f mine_kq[0]=%.4f\n", kq[0], out[0]);
    return 0;
}
