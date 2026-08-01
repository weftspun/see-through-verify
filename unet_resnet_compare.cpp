// UNet down_blocks.0.resnets.0 compare: direct f32 groupnorm+silu+conv+
// time-emb projection vs the ggml oracle tap. Validates group norm, SiLU
// and resnet wiring (the next primitives after conv2d on the UNet path).
// Inputs come from the oracle: conv_in.bin (the resnet input x, [W,H,320,F])
// and emb.bin (the 1280-time-embed [1280,F]).
// Usage: unet_resnet_compare <unet.safetensors> <oracle_dir> [W H F]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

struct SfTensor { std::string name, dtype; std::vector<int64_t> shape; size_t offset, size; };
static std::vector<SfTensor> read_sf(const char *path, size_t *data_start) {
    FILE *f = fopen(path, "rb");
    uint8_t buf[8]; fread(buf, 1, 8, f);
    uint64_t hdr_len = 0;
    for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)buf[j] << (j * 8);
    std::string json((size_t)hdr_len, 0);
    fread(&json[0], 1, hdr_len, f); *data_start = 8 + hdr_len; fclose(f);
    std::vector<SfTensor> tensors; size_t i = 0;
    while (i < json.size() && json[i] != '}') {
        while (i < json.size() && (json[i]==' '||json[i]==10||json[i]==9||json[i]==13||json[i]==',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }
        size_t ns = ++i; while (i < json.size() && json[i] != '"') i++;
        std::string name = json.substr(ns, i - ns); i++;
        if (name == "__metadata__") { while (i < json.size() && json[i] != '}') i++; i++; continue; }
        while (i < json.size() && json[i] != '{') i++; i++;
        SfTensor t; t.name = name; t.offset = 0; t.size = 0;
        while (i < json.size() && json[i] != '}') {
            while (i < json.size() && (json[i]==' '||json[i]==10||json[i]==9||json[i]==13||json[i]==',')) i++;
            if (json[i] == '}') break; if (json[i] != '"') { i++; continue; }
            size_t fs = ++i; while (i < json.size() && json[i] != '"') i++;
            std::string field = json.substr(fs, i - fs); i++;
            while (i < json.size() && json[i] != ':') i++; i++;
            if (field == "dtype") {
                if (json[i] == '"') { size_t vs = ++i; while (i < json.size() && json[i] != '"') i++; t.dtype = json.substr(vs, i - vs); i++; }
            } else if (field == "shape") {
                if (json[i] == '[') { i++; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; t.shape.push_back(strtol(&json[i], &end, 10)); i = end - &json[0]; } } i++; }
            } else if (field == "data_offsets") {
                if (json[i] == '[') { i++; int idx=0; uint64_t vals[2]={0,0}; while (i < json.size() && json[i] != ']') { while (i < json.size() && (json[i]==' '||json[i]==',')) i++; if (i < json.size() && json[i] >= '0' && json[i] <= '9') { char *end; vals[idx++]=strtoull(&json[i], &end, 10); i=end-&json[0]; } } i++; t.offset=vals[0]; t.size=vals[1]-vals[0]; }
            }
        }
        i++; tensors.push_back(t);
    }
    return tensors;
}
static std::vector<uint8_t> load_tensor_data(const char *sf, size_t ds, size_t off, size_t sz) {
    FILE *f = fopen(sf, "rb"); fseeko(f, (off_t)(ds+off), SEEK_SET);
    std::vector<uint8_t> data(sz); fread(data.data(), 1, sz, f); fclose(f); return data;
}
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> &raw, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) { uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16; memcpy(&out[i], &u32, 4); }
    return out;
}
static void silu_inplace(float *x, size_t n) { for (size_t i=0;i<n;i++) x[i] = x[i] / (1.0f + expf(-x[i])); }

// group_norm (ggml semantics): normalizes over channels-in-group x H x W,
// per spatial-pixel/batch. WHCN layout.
static void group_norm_affine(const float *x, float *y, const float *g, const float *b,
                              int W, int H, int C, int F, int n_groups, float eps) {
    int cpg = (C + n_groups - 1) / n_groups;
    for (int n = 0; n < F; n++)
      for (int gr = 0; gr < n_groups; gr++) {
        int start = gr * cpg, end = std::min(start + cpg, C);
        int step = end - start;
        // mean/var over [start,end) channels x H x W
        double sum = 0;
        for (int c = start; c < end; c++)
          for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
              float xv = x[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)h*W + w];
              sum += xv;
            }
        double mean = sum / ((double)W*H*step);
        double sum2 = 0;
        for (int c = start; c < end; c++)
          for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
              float xv = x[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)h*W + w];
              double d = xv - mean; sum2 += d*d;
            }
        double var = sum2 / ((double)W*H*step);
        double scale = 1.0 / sqrt(var + eps);
        for (int c = start; c < end; c++)
          for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
              size_t idx = (size_t)n*W*H*C + (size_t)c*W*H + (size_t)h*W + w;
              y[idx] = (float)((x[idx] - mean) * scale) * g[c] + b[c];
            }
      }
}

// conv2d stride 1 pad 1, WHCN in/out, weight [oc,ic,kh,kw]
static void conv2d_3x3(const float *x, const float *Wt, const float *Bt, float *out,
                       int W, int H, int IC, int OC, int F) {
    for (int n = 0; n < F; n++)
      for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < H; oh++)
          for (int ow = 0; ow < W; ow++) {
            float s = Bt[oc];
            for (int c = 0; c < IC; c++)
              for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++) {
                  int ih = oh - 1 + kh, iw = ow - 1 + kw;
                  if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                  float wv = Wt[((oc*IC + c)*3 + kh)*3 + kw];
                  float xv = x[(size_t)n*W*H*IC + (size_t)c*W*H + (size_t)ih*W + iw];
                  s += wv * xv;
                }
            out[(size_t)n*W*H*OC + (size_t)oc*W*H + (size_t)oh*W + ow] = s;
          }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors oracle_dir [W H F]\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);

    int W = 8, H = 8, F = 13;
    if (argc >= 6) { W = atoi(argv[3]); H = atoi(argv[4]); F = atoi(argv[5]); }

    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name.find(suffix) != std::string::npos) return &t;
        return nullptr;
    };
    auto load_w = [&](std::string suffix) -> std::vector<float> {
        auto *t = find(suffix);
        int n = 1; for (auto s : t->shape) n *= (int)s;
        auto raw = load_tensor_data(sf_path, ds, t->offset, t->size);
        return bf16_to_f32(raw, (size_t)n);
    };

    auto read_oracle = [&](const char *name, int &oW,int &oH,int &oC,int &oF) -> std::vector<float> {
        FILE *f = fopen((std::string(argv[2]) + "/" + name).c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s\n", name); exit(1); }
        int32_t hdr[5]; fread(hdr, 4, 5, f);
        oW=hdr[0]; oH=hdr[1]; oC=hdr[2]; oF=hdr[3];
        size_t n = (size_t)oW*oH*oC*oF;
        std::vector<float> buf(n); fread(buf.data(), 4, n, f); fclose(f);
        return buf;
    };

    // Inputs from oracle
    int cw,ch,cc,cf;
    auto x = read_oracle("conv_in.bin", cw,ch,cc,cf);   // [W,H,320,F] WHCN
    int ew,eh,ec,ef;
    auto emb = read_oracle("emb.bin", ew,eh,ec,ef);      // [1280,F] -> ne[1]=F
    printf("resnet input [%dx%dx%dx%d], emb [%d x %d]\n", cw,ch,cc,cf, ew, ef);

    const int C = cc;   // 320
    const std::string pre = "down_blocks.0.resnets.0";
    auto gn1g = load_w(pre+".norm1.weight"), gn1b = load_w(pre+".norm1.bias");
    auto cv1w = load_w(pre+".conv1.weight"), cv1b = load_w(pre+".conv1.bias");
    auto tepw = load_w(pre+".time_emb_proj.weight"), tepb = load_w(pre+".time_emb_proj.bias");
    auto gn2g = load_w(pre+".norm2.weight"), gn2b = load_w(pre+".norm2.bias");
    auto cv2w = load_w(pre+".conv2.weight"), cv2b = load_w(pre+".conv2.bias");
    int sh_;
    bool has_sc = find(pre+".conv_shortcut.weight") != nullptr;
    auto scw = has_sc ? load_w(pre+".conv_shortcut.weight") : std::vector<float>();
    auto scb = has_sc ? load_w(pre+".conv_shortcut.bias") : std::vector<float>();

    // h = GN(norm1)(x); silu; conv1
    std::vector<float> h((size_t)W*H*C*F);
    group_norm_affine(x.data(), h.data(), gn1g.data(), gn1b.data(), W,H,C,F, 32, 1e-5f);
    silu_inplace(h.data(), h.size());
    std::vector<float> h1((size_t)W*H*C*F);
    int oc1 = (int)cv1w.size()/(C*3*3);
    conv2d_3x3(h.data(), cv1w.data(), cv1b.data(), h1.data(), W,H,C,oc1,F);
    // + time_emb_proj(silu(emb))
    std::vector<float> t_emb((size_t)1280*F);
    for (int n=0;n<F;n++) for (int i=0;i<1280;i++) t_emb[(size_t)n*1280+i] = emb[n*1280+i];
    std::vector<float> temb_s((size_t)1280*F);
    for (size_t i=0;i<temb_s.size();i++) temb_s[i] = t_emb[i];
    silu_inplace(temb_s.data(), temb_s.size());
    // time_emb_proj: linear [1280 -> 1280] (weight [out,in])
    std::vector<float> te_out((size_t)1280*F, 0.f);
    {
        int din=1280, dow=1280;
        for (int n=0;n<F;n++)
          for (int o=0;o<dow;o++) {
            float s = tepb[o];
            for (int k=0;k<din;k++) s += tepw[(size_t)o*din+k] * temb_s[(size_t)n*din+k];
            te_out[(size_t)n*dow+o] = s;
          }
    }
    // reshape [1280,F] -> [1,1,1280,F] and add to h1 (per-channel broadcast over W,H)
    for (int n=0;n<F;n++)
      for (int c=0;c<C;c++)
        for (int hm=0;hm<H;hm++)
          for (int wm=0;wm<W;wm++)
            h1[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)hm*W + wm] += te_out[(size_t)n*1280+c];

    // h = GN(norm2)(h1); silu; conv2
    std::vector<float> h2((size_t)W*H*C*F);
    group_norm_affine(h1.data(), h2.data(), gn2g.data(), gn2b.data(), W,H,C,F, 32, 1e-5f);
    silu_inplace(h2.data(), h2.size());
    std::vector<float> h3((size_t)W*H*C*F);
    conv2d_3x3(h2.data(), cv2w.data(), cv2b.data(), h3.data(), W,H,C,oc1,F);

    // shortcut
    std::vector<float> sc((size_t)W*H*C*F);
    if (has_sc) {
        conv2d_3x3(x.data(), scw.data(), scb.data(), sc.data(), W,H,C,oc1,F);
    } else {
        sc = x;
    }
    // out = h3 + sc
    std::vector<float> mine((size_t)W*H*C*F);
    for (size_t i=0;i<mine.size();i++) mine[i] = h3[i] + sc[i];

    // read oracle down0_resnet0
    int rw,rh,rc,rf;
    auto ref = read_oracle("down0_resnet0.bin", rw,rh,rc,rf);
    printf("oracle down0_resnet0 [%dx%dx%dx%d]\n", rw,rh,rc,rf);

    double max_abs=0, sum_abs=0; size_t cnt=mine.size();
    for (size_t i=0;i<cnt;i++) { double dd=fabs((double)mine[i]-(double)ref[i]); if(dd>max_abs)max_abs=dd; sum_abs+=dd; }
    double cos_sim=0, na=0, nb=0;
    for (size_t i=0;i<cnt;i++){ cos_sim+=(double)mine[i]*ref[i]; na+=(double)mine[i]*mine[i]; nb+=(double)ref[i]*ref[i]; }
    cos_sim /= sqrt(na*nb);
    printf("mine[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", mine[0],mine[1],mine[2],mine[3],mine[4],mine[5],mine[6],mine[7]);
    printf("ora [0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
    printf("resnet.0 (%dx%dx%dx%d): max_abs_diff=%.6f mean=%.6f cosine=%.6f\n", W,H,C,F, max_abs, sum_abs/cnt, cos_sim);
    bool pass = max_abs < 2e-2 && cos_sim > 0.999;
    int rc2 = 0;

    // ---- Downsampler: stride-2 pad-1 k3 conv on the resnet output -> down0.bin
    std::string ds_pre = "down_blocks.0.downsamplers.0.conv";
    if (find(ds_pre+".weight")) {
        // down_block.0 = resnets.0 -> resnets.1 -> downsampler (no attention)
        // apply resnets.1 to `mine` (the resnets.0 output) first
        std::string r1 = "down_blocks.0.resnets.1";
        auto r1n1g = load_w(r1+".norm1.weight"), r1n1b = load_w(r1+".norm1.bias");
        auto r1c1w = load_w(r1+".conv1.weight"), r1c1b = load_w(r1+".conv1.bias");
        auto r1tpw = load_w(r1+".time_emb_proj.weight"), r1tpb = load_w(r1+".time_emb_proj.bias");
        auto r1n2g = load_w(r1+".norm2.weight"), r1n2b = load_w(r1+".norm2.bias");
        auto r1c2w = load_w(r1+".conv2.weight"), r1c2b = load_w(r1+".conv2.bias");
        std::vector<float> r1in = mine;   // resnet.1 input = resnet.0 output
        // norm1 + silu + conv1
        std::vector<float> hh((size_t)W*H*C*F);
        group_norm_affine(r1in.data(), hh.data(), r1n1g.data(), r1n1b.data(), W,H,C,F, 32, 1e-5f);
        silu_inplace(hh.data(), hh.size());
        std::vector<float> hh1((size_t)W*H*C*F);
        conv2d_3x3(hh.data(), r1c1w.data(), r1c1b.data(), hh1.data(), W,H,C,oc1,F);
        // + time_emb_proj(silu(emb))
        std::vector<float> r1te((size_t)1280, 0.f);
        for (int o=0;o<1280;o++){ float s=r1tpb[o]; for(int k=0;k<1280;k++) s+=r1tpw[(size_t)o*1280+k]*temb_s[k]; r1te[o]=s; }
        for (int n=0;n<F;n++) for (int c=0;c<C;c++) for (int hm=0;hm<H;hm++) for (int wm=0;wm<W;wm++)
            hh1[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)hm*W + wm] += r1te[c];
        // norm2 + silu + conv2
        std::vector<float> hh2((size_t)W*H*C*F);
        group_norm_affine(hh1.data(), hh2.data(), r1n2g.data(), r1n2b.data(), W,H,C,F, 32, 1e-5f);
        silu_inplace(hh2.data(), hh2.size());
        std::vector<float> hh3((size_t)W*H*C*F);
        conv2d_3x3(hh2.data(), r1c2w.data(), r1c2b.data(), hh3.data(), W,H,C,oc1,F);
        // shortcut: no conv_shortcut in resnet.1 (skip=identity)
        std::vector<float> mine1((size_t)W*H*C*F);
        for (size_t i=0;i<mine1.size();i++) mine1[i] = hh3[i] + r1in[i];
        mine = mine1;

        auto dsw = load_w(ds_pre+".weight"), dsb = load_w(ds_pre+".bias");
        int dOC = 320; // input==output channels
        int oW2 = W/2, oH2 = H/2;
        // stride-2 conv: out[n,oc,oh,ow] = bias + SUM_c SUM_kh SUM_kw x[n,c,oh*2-1+kh, ow*2-1+kw]*w
        std::vector<float> down((size_t)oH2*oW2*dOC*F, 0.f);
        for (int n=0;n<F;n++)
          for (int oc=0;oc<dOC;oc++)
            for (int oh=0;oh<oH2;oh++)
              for (int ow=0;ow<oW2;ow++) {
                float s = dsb[oc];
                for (int c=0;c<320;c++)
                  for (int kh=0;kh<3;kh++)
                    for (int kw=0;kw<3;kw++) {
                      int ih = oh*2 - 1 + kh, iw = ow*2 - 1 + kw;
                      if (ih<0 || ih>=H || iw<0 || iw>=W) continue;
                      float wv = dsw[((oc*320 + c)*3 + kh)*3 + kw];
                      float xv = mine[(size_t)n*W*H*320 + (size_t)c*W*H + (size_t)ih*W + iw];
                      s += wv*xv;
                    }
                down[(size_t)n*oH2*oW2*dOC + (size_t)oc*oH2*oW2 + (size_t)oh*oW2 + ow] = s;
              }
        int dw,dh,dc,df;
        auto dref = read_oracle("down0.bin", dw,dh,dc,df);
        printf("downsampled [%dx%dx%dx%d] vs oracle [%dx%dx%dx%d]\n", oW2,oH2,dOC,F, dw,dh,dc,df);
        double dmax=0, dsum=0, dmaxv=0; size_t dcnt=down.size();
        for (size_t i=0;i<dcnt;i++){ double dd=fabs((double)down[i]-(double)dref[i]); if(dd>dmax)dmax=dd; dsum+=dd; double v=fabs((double)dref[i]); if(v>dmaxv)dmaxv=v; }
        double dcos=0, dna=0, dnb=0;
        for (size_t i=0;i<dcnt;i++){ dcos+=(double)down[i]*dref[i]; dna+=(double)down[i]*down[i]; dnb+=(double)dref[i]*dref[i]; }
        dcos /= sqrt(dna*dnb);
        printf("down0 (downsampled resnet.1): max_abs_diff=%.6f (max|ref|=%.3f) mean=%.6f cosine=%.6f\n",
               dmax, dmaxv, dsum/dcnt, dcos);
        bool dp = (dmax < 0.1 && dcos > 0.999);   // CLIP-compare convention (16-bit weight rounding)
        printf("%s\n", dp ? "down_block.0 GREEN — resnet.0+resnet.1+downsample matches ggml oracle"
                          : "down_block.0 FAIL");
        if (!dp) rc2 = 1;
    } else {
        printf("down_block.0 has no downsampler (skip)\n");
    }
    printf("%s\n", (pass && rc2==0) ? "VALIDATION PASS — resnet+groupnorm matches ggml oracle (16-bit weight rounding)" : "VALIDATION FAIL");
    return (pass && rc2==0) ? 0 : 1;
}
