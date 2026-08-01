groupshared half sA[128];
groupshared half sB[128];

[[vk::binding(0, 0)]]
StructuredBuffer<float> A;
[[vk::binding(1, 0)]]
StructuredBuffer<float> B;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]]
uint M;
[[vk::binding(4, 0)]]
uint N;
[[vk::binding(5, 0)]]
uint K;

[shader("compute")] [numthreads(16, 16, 1)]
void gemm(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
  uint row = dtid[0u];
  uint col = dtid[1u];
  uint tx = gtid[0u];
  uint ty = gtid[1u];
  float sum = 0.000000;
  uint numTiles = ((K + 7u) / 8u);
  for (uint kt = 0u; kt < numTiles; ++kt) {
    uint kk = (kt * 8u);
    sA[((tx * 8u) + ty)] = half(A[((row * K) + (kk + ty))]);
    sB[((tx * 8u) + ty)] = half(B[(((kk + tx) * N) + col)]);
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0u; k < 8u; ++k) {
      sum = (sum + (float(sA[((tx * 8u) + k)]) * float(sB[((k * 8u) + ty)])));
    }
    GroupMemoryBarrierWithGroupSync();
  }
  C[((row * N) + col)] = sum;
}