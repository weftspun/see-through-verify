// Conv2d kernel (Slang -> slangc -target cpp for CPU validation).
// Matches ggml conv_in + validated conv2d_compare:
//   out[n, oc, oh, ow] = bias[oc] + SUM_c SUM_kh SUM_kw x[n, c, ih, iw] * W[oc, c, kh, kw]
//   ih = oh*stride - pad + kh, iw = ow*stride - pad + kw
//   W layout [oc, c, kh, kw]; x IN layout [n, c, ih, iw]; out WHCN [n, oc, oh, ow].
// Dispatch: dtid.x = oc, dtid.y = oh*OW+ow (pixel), dtid.z = n (batch).
[[vk::binding(0, 0)]] StructuredBuffer<float> X_in;
[[vk::binding(1, 0)]] StructuredBuffer<float> W_in;
[[vk::binding(2, 0)]] StructuredBuffer<float> B_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y_out;
[[vk::binding(4, 0)]] uint H;
[[vk::binding(5, 0)]] uint W;
[[vk::binding(6, 0)]] uint IC;
[[vk::binding(7, 0)]] uint OC;
[[vk::binding(8, 0)]] uint KH;
[[vk::binding(9, 0)]] uint KW;
[[vk::binding(10, 0)]] uint STRIDE;
[[vk::binding(11, 0)]] uint PAD;

[shader("compute")] [numthreads(8, 8, 1)]
void conv2d_kern(uint3 dtid : SV_DispatchThreadID) {
    uint oc = dtid.x;
    uint ow = dtid.y % W;
    uint oh = dtid.y / W;
    uint n = dtid.z;
    if (oc >= OC || oh >= H) return;
    float s = B_in[oc];
    for (uint c = 0; c < IC; ++c)
      for (uint kh = 0; kh < KH; ++kh)
        for (uint kw = 0; kw < KW; ++kw) {
            int ih = int(oh * STRIDE) - int(PAD) + int(kh);
            int iw = int(ow * STRIDE) - int(PAD) + int(kw);
            if (ih < 0 || ih >= int(H) || iw < 0 || iw >= int(W)) continue;
            float wv = W_in[((oc*IC + c)*KH + kh)*KW + kw];
            float xv = X_in[(n*H*W*IC) + (c*H*W) + (uint(ih)*W) + uint(iw)];
            s += wv * xv;
        }
    Y_out[(n*OC*H*W) + (oc*H*W) + (oh*W) + ow] = s;
}
