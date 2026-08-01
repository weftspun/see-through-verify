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

[shader("compute")] [numthreads(64, 64, 1)]
void gemm(uint3 dtid : SV_DispatchThreadID) {
  uint row = dtid[0u];
  uint col = dtid[1u];
  float sum = 0.000000;
  for (uint k = 0u; k < K; ++k) {
    sum = (sum + (A[((row * K) + k)] * B[((k * N) + col)]));
  }
  C[(row + (col * N))] = sum;
}