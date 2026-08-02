# RFD 1 — Lean Shader Slang See-Through

**Status:** Draft  
**Author:** fire  
**Date:** 2026-07-31  
**License:** Apache 2.0

---

## Problem Statement

The [weftspun/see-through-cpp](https://github.com/weftspun/see-through-cpp) pipeline decomposes a single anime illustration into semantically layered PSD output. It currently runs on ggml (C++ GPU backends). We need a verified, portable implementation using Lean 4 for formal kernel correctness and Slang for portable GPU compute.

## Goals

1. **Correctness by construction**: Every GPU compute kernel is emitted from Lean 4 via [lean-slang](https://github.com/V-Sekai-fire/lean-slang), verified with `native_decide` reference fixtures, and cross-checked against a CPU harness.

2. **Portable weights**: Model weights from [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) (HuggingFace safetensors) are converted to Parquet with zstd compression and consumed by the Lean pipeline via [lean-duckdb](https://github.com/v-sekai-multiplayer-fabric/lean-duckdb).

3. **Performant**: Slang → `slangc -target spirv` → Vulkan GPU dispatch on NVIDIA/AMD; Slang → Metal Shading Language path for Apple Silicon.

## Repository Layout

```
see-through-verify/
├── RFD/                   # Decision records (Oxide-style)
├── Compute/               # Lean Slang compute shader modules
│   ├── Gemm.lean          # f32×f32 GEMM
│   ├── Conv2d.lean        # f32×f32 direct conv2d
│   ├── Gemm.lean          # im2col + GEMM conv
│   └── Verify.lean        # native_decide reference fixtures
├── Weights/               # Parquet weight files (gitignored)
├── Tools/                 # Conversion scripts
│   ├── download-models.py # fetch safetensors from HF
│   └── to-parquet.py      # safetensors → Parquet with zstd
├── Runtime/               # GPU runtime harness
│   ├── loader.cpp         # Parquet → Vulkan buffer upload
│   └── dispatcher.cpp     # Slang SPIR-V dispatch
├── lakefile.lean
├── lean-toolchain
└── LICENSE
```

## Weight Pipeline

```
shitagaki-lab/see-through (HuggingFace safetensors)
        │
        ▼
  Tools/download-models.py
        │
        ▼
  Tools/to-parquet.py  (extract tensors, f16→f32, write Parquet zstd)
        │
        ▼
  Weights/*.parquet  (indexed by model/tensor_name)
        │
        ▼
  lean-duckdb  (read at runtime, verify shapes/ranges)
        │
        ▼
  Slang StructuredBuffer<float>  (uploaded to GPU)
```

### Parquet Schema

Each tensor becomes one Parquet file:

```sql
CREATE TABLE tensor AS SELECT * FROM read_parquet('weights/layerdiff-unet/conv_in.weight.parquet');
-- Schema: idx UINT64, val FLOAT
-- Shape metadata in file comments or sidecar JSON
```

## Compute Shader Modules

| Shader                  | Role                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| GEMM                    | C[M,N] = A[M,K] × B[K,N] — foundation for all linears, convs, attention |
| Direct conv2d           | out[oc,oh,ow] = Σ_c Σ_kh Σ_kw in[c,oh·s+kh,ow·s+kw] × w[c,oc,kh,kw]     |
| MHA attention           | QK^T / sqrt(hd) softmax PV, per-head workspace                          |
| LayerNorm / GroupNorm   | Normalization for transformer blocks & resnets                          |
| GEGLU                   | LayerNorm GEGLU FFN (gate in 2nd half)                                  |

All kernels are authored in Slang and exported to C++ via `slangc -target cpp`
(per-thread, no barriers), then validated against ggml oracle taps — no
hand-coded C++. Live verification status (which kernels are GREEN) is tracked
in `decisions/0006`; the critical-path plan is `decisions/0006` and its PERT
diagram is `decisions/0007`.

## GPU Dispatch

The runtime flow:

1. **Compile time** (Lean): `Compute/Gemm.lean` → `LeanSlang.emit` → `.slang` source
2. **Build time** (slangc): `slangc -target spirv -o gemm.spv gemm.slang`
3. **Launch time** (C++/Rust harness):  
   a. lean-duckdb reads Parquet weight files → float buffers in host memory  
   b. Vulkan `vkCreateBuffer` + `vkCmdCopy` uploads weights to GPU  
   c. `vkCmdDispatch` runs the SPIR-V shader with the weight buffers bound  
   d. Readback results to host

## Verification Gate

The existing `KernelGate.lean` continues to test kernel correctness by comparing GPU output against CPU reference at every production shape. The only change: the GPU kernel is now compiled from Lean-emitted Slang instead of ggml's Metal backend.

## Next Steps

1. **Clone seed**: Fork [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) to preserve the code exactly as published; then use `git subtree` to pull the conversion scripts.

2. **Download models**: Write `tools/download-models.py` that fetches the safetensors from HuggingFace (LayerDiff 3D, Marigold depth, VAE).

3. **Convert to Parquet**: Write `tools/to-parquet.py` that reads safetensors, extracts f16 tensors, casts to f32, and writes zstd-compressed Parquet.

4. **Add lean-duckdb**: Add `require lean-duckdb from git` to the lakefile.

5. **Extend compute shaders**: Add attention, normalization, and activation shaders to `Compute/`.

6. **Runtime harness**: Write the C++ Vulkan dispatch loader that reads Parquet and runs the Slang SPIR-V.

7. **End-to-end test**: Run the full pipeline on a test image and compare output against the C++ reference.
