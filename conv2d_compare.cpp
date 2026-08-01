// Conv2d compare: our direct f32 conv2d (Lean Conv2d shader semantics)
// vs the ggml conv_in oracle. Validates the Conv2d primitive for the UNet.
// Usage: conv2d_compare <unet.safetensors> <oracle.bin> [W H C F]
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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s unet.safetensors oracle.bin [W H C F]\n", argv[0]); return 1; }
    const char *sf_path = argv[1];
    size_t ds = 0;
    auto tensors = read_sf(sf_path, &ds);

    int W = 16, H = 16, C = 8, F = 1;
    if (argc >= 7) { W = atoi(argv[3]); H = atoi(argv[4]); C = atoi(argv[5]); F = atoi(argv[6]); }

    auto find = [&](std::string suffix) -> const SfTensor* {
        for (auto &t : tensors) if (t.name.find(suffix) != std::string::npos) return &t;
        return nullptr;
    };

    // conv_in.weight: safetensors layout [out_ch, in_ch, kh, kw]
    auto *wt = find("conv_in.weight");
    auto *bt = find("conv_in.bias");
    if (!wt) { fprintf(stderr, "no conv_in.weight\n"); return 1; }
    const int OC = (int)wt->shape[0], IC = (int)wt->shape[1], KH = (int)wt->shape[2], KW = (int)wt->shape[3];
    auto wraw = load_tensor_data(sf_path, ds, wt->offset, wt->size);
    auto Wt = bf16_to_f32(wraw, (size_t)OC*IC*KH*KW);
    std::vector<float> Bt;
    if (bt) { auto braw = load_tensor_data(sf_path, ds, bt->offset, bt->size); Bt = bf16_to_f32(braw, (size_t)bt->shape[0]); }
    else Bt.assign((size_t)OC, 0.f);
    printf("conv_in.weight [%d,%d,%d,%d], bias=%zu, oc=%d\n", OC, IC, KH, KW, Bt.size(), OC);

    // Reconstruct the oracle input latent deterministically (must match conv2d_oracle).
    // WHCN layout: idx = w + h*W + c*W*H + n*W*H*C
    std::vector<float> x((size_t)W*H*C*F);
    uint32_t seed = 42;
    for (auto &v : x) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) & 0xffff) / 32767.5f - 1.0f; }

    // Direct conv2d, f32, stride 1, pad 1, WHCN output.
    // out[n, oc, oh, ow] = bias[oc] + SUM_c SUM_kh SUM_kw x[n, c, oh-1+kh, ow-1+kw] * W[oc, c, kh, kw]
    const int oW = W, oH = H;
    std::vector<float> out((size_t)oH*oW*OC*F, 0.f);
    for (int n = 0; n < F; n++)
      for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < oH; oh++)
          for (int ow = 0; ow < oW; ow++) {
            float s = Bt[oc];
            for (int c = 0; c < IC; c++)
              for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) {
                  int ih = oh - 1 + kh, iw = ow - 1 + kw;
                  if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                  // W layout [oc, c, kh, kw]
                  float wv = Wt[((oc*IC + c)*KH + kh)*KW + kw];
                  float xv = x[(size_t)n*W*H*C + (size_t)c*W*H + (size_t)ih*W + iw];
                  s += wv * xv;
                }
            out[(size_t)n*oH*oW*OC + (size_t)oc*oH*oW + (size_t)oh*oW + ow] = s;
          }

    // Read oracle
    FILE *f = fopen(argv[2], "rb");
    int32_t hdr[5]; fread(hdr, 4, 5, f);
    size_t on = (size_t)hdr[3]*hdr[2]*hdr[1]*hdr[0];
    std::vector<float> ref(on);
    fread(ref.data(), 4, on, f);
    fclose(f);
    printf("oracle [%dx%dx%dx%d]\n", hdr[0],hdr[1],hdr[2],hdr[3]);

    double max_abs = 0, sum_abs = 0;
    size_t cnt = out.size();
    for (size_t i = 0; i < cnt; i++) {
        double dd = fabs((double)out[i] - (double)ref[i]);
        if (dd > max_abs) max_abs = dd;
        sum_abs += dd;
    }
    printf("mine[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    printf("ora [0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
    printf("conv_in %dx%dx%dx%d (%d oc, k%d): max_abs_diff=%.6f mean=%.6f\n",
           W,H,C,F, OC, KH, max_abs, sum_abs/cnt);
    printf("%s\n", max_abs < 1e-3 ? "VALIDATION PASS — conv2d matches ggml oracle" : "VALIDATION FAIL");
    return max_abs < 1e-3 ? 0 : 1;
}
