// GroupNorm kernel (Slang -> slangc -target cpp for CPU validation).
// Matches ggml group_norm: mean/var over (C_g x H x W) per batch n, then
//   y[n,c,h,w] = (x - mean)*inv*G[c] + B[c],  inv = 1/sqrt(var+eps).
// Layout: WHCN native [n, c, h*w]. Dispatch: dtid.x = n, dtid.y = c, dtid.z = h*w
// (dtid.z unused; one thread covers its (n,c) full spatial via redux).
[[vk::binding(0, 0)]] StructuredBuffer<float> X_in;
[[vk::binding(1, 0)]] StructuredBuffer<float> G_in;
[[vk::binding(2, 0)]] StructuredBuffer<float> B_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y_out;
[[vk::binding(4, 0)]] uint C;
[[vk::binding(5, 0)]] uint HW;
[[vk::binding(6, 0)]] uint N_GROUPS;
[[vk::binding(7, 0)]] float EPS;

[shader("compute")] [numthreads(8, 8, 1)]
void groupnorm_kern(uint3 dtid : SV_DispatchThreadID) {
    uint n = dtid.x;
    uint c = dtid.y;
    if (c >= C) return;
    uint cpg = (C + N_GROUPS - 1) / N_GROUPS;
    uint gstart = (c / cpg) * cpg;
    uint gend = min(gstart + cpg, C);
    uint nch = gend - gstart;   // channels in this group
    float mean = 0;
    for (uint cc = gstart; cc < gend; ++cc)
      for (uint p = 0; p < HW; ++p)
        mean += X_in[(n*C*HW) + (cc*HW) + p];
    mean = mean / float(nch * HW);
    float var = 0;
    for (uint cc = gstart; cc < gend; ++cc)
      for (uint p = 0; p < HW; ++p) {
        float q = X_in[(n*C*HW) + (cc*HW) + p] - mean;
        var += q * q;
      }
    var = var / float(nch * HW);
    float inv = 1.0 / sqrt(var + EPS);
    for (uint p = 0; p < HW; ++p)
        Y_out[(n*C*HW) + (c*HW) + p] = (X_in[(n*C*HW) + (c*HW) + p] - mean) * inv * G_in[c] + B_in[c];
}
