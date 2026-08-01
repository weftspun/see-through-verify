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
void gemm(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint sgtg : SV_GroupID) {
  uint row = dtid[0u];
  uint col = dtid[1u];
  uint tx = gtid[0u];
  uint ty = gtid[1u];
  uint laneId = (((tx / 8u) * 8u) + (ty / 8u));
  uint laneInSG = (((tx % 8u) * 8u) + (ty % 8u));
  simdgroup_half8x8 mAcc;
  simdgroup_half8x8 mA;
  simdgroup_half8x8 mB;
  simdgroup_load(mAcc, 0.000000, 0u);
  uint numTiles = ((K + 7u) / 8u);
  for (uint kkTile = 0u; kkTile < numTiles; ++kkTile) {
    uint kk = (kkTile * 8u);
    sA[((tx * 8u) + ty)] = half(A[((row * K) + (kk + ty))]);
    sB[((tx * 16u) + ty)] = half(B[(((kk + tx) * N) + col)]);
    GroupMemoryBarrierWithGroupSync();
    mA[laneInSG] = sA[laneInSG];
    mB[laneInSG] = sB[laneInSG];
    simdgroup_multiply_accumulate(mAcc, mA, mB, mAcc);
    GroupMemoryBarrierWithGroupSync();
  }
  C[(row + (col * N))] = mAcc[laneInSG];
}