groupshared float sA[1024];
groupshared float sB[1024];

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

[shader("compute")] [numthreads(32, 32, 1)]
void gemm(uint3 dtid : SV_DispatchThreadID) {
  uint row = dtid[0u];
  uint col = dtid[1u];
  uint tx = (row % 32u);
  uint ty = (col % 32u);
  float sum = 0.000000;
  uint numTiles = ((K + 31u) / 32u);
  for (uint kkTile = 0u; kkTile < numTiles; ++kkTile) {
    uint kk = (kkTile * 32u);
    sA[((tx * 32u) + ty)] = A[((row * K) + (kk + ty))];
    sB[((tx * 32u) + ty)] = B[(((kk + tx) * N) + col)];
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0u; k < 32u; ++k) {
      sum = (sum + (sA[((tx * 32u) + k)] * sB[((k * 32u) + ty)]));
    }
    GroupMemoryBarrierWithGroupSync();
  }
  C[(row + (col * N))] = sum;
}