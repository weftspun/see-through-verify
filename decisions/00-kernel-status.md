All implementation kernels are authored in Slang and exported to C++ via
`slangc -target cpp` (per-thread, no barriers), then validated against ggml
oracle taps — no hand-coded C++:

| Kernel | Shader | Status |
|--------|--------|--------|
| GEMM | `shaders/gemm.slang` | GREEN |
| MHA attention | `shaders/mha.slang` | GREEN (cos=1.0) |
| LayerNorm | `shaders/layernorm.slang` | GREEN (cos=1.0) |
| GEGLU | `shaders/geglu.slang` | GREEN (cos=0.999997) |
| Conv2d | `shaders/conv2d.slang` | GREEN (cos=1.0) |
| GroupNorm | `shaders/groupnorm.slang` | GREEN (cos=1.0) |
