# see-through CLI — Lean Shader Slang implementation

This directory contains the new see-through CLI that replicates the
[weftspun/see-through-cpp](https://github.com/weftspun/see-through-cpp)
pipeline using Lean Slang compute shaders + DuckDB + Vulkan.

## Pipeline architecture

```
Input image (PNG)
    │
    ▼
CLIP text encoder ───→ text embeddings
(Lean Slang compute)      │
    │                     ▼
LayerDiff UNet body ──→ 30-step diffusion → per-tag RGBA layers
(Lean Slang compute)      │
    │                     ▼
LayerDiff UNet head ──→ head refinement pass
(Lean Slang compute)      │
    │                     ▼
Marigold depth ───────→ pseudo-depth maps
(Lean Slang compute)      │
    │                     ▼
VAE decode ───────────→ pixel-space layers
(Lean Slang compute)      │
    │                     ▼
Postprocess ──────────→ layered PSD + depth PSD + JSON metadata
(C++/PSD SDK)
```

## Weight format

All weights are stored as zstd-compressed Parquet files under `weights/`:

```
weights/
├── layerdiff3d/          # from layerdifforg/seethroughv0.0.2_layerdiff3d
│   ├── unet/
│   │   ├── down_blocks_0_attentions_0_to_q.weight.parquet
│   │   ├── ...
│   ├── text_encoder/
│   ├── vae/
├── marigold/             # from 24yearsold/seethroughv0.0.1_marigold
│   ├── unet/
│   ├── vae/
```

## CLI usage

```bash
./see-through -i input.png -o out.psd --steps 30 --res 1280

# All flags match the original C++ CLI:
#   -i <input.png>          Input image
#   -o <output.psd>         Output PSD
#   --seed N                Random seed (default: 42)
#   --steps N               Diffusion steps (default: 30)
#   --res N                 Inference resolution (default: 1280)
#   --depth-res N           Depth model resolution (default: 768)
#   --threads N             CPU threads (default: 8)
#   --device <auto/vulkan>  GPU backend (default: auto)
#   --debug-dir <dir>       Dump per-stage debug output
#   --png-dir <dir>         Save per-layer PNGs
#   --no-split-depth        Disable depth-based layer splitting
#   --no-split-lr           Disable left-right splitting
#   --split-depth-tags ...  Tags for depth splitting
#   --split-lr-tags ...     Tags for left-right splitting
```

## Build

```bash
# Compile Slang shaders
for f in shaders/*.slang; do
  slangc -target spirv -o ${f%.slang}.spv "$f"
done

# Build CLI
clang++ -std=c++17 see-through.cpp Runtime/harness.cpp \
  -o see-through -lduckdb -lvulkan -lpsd_sdk \
  -I/path/to/duckdb/include -I/path/to/vulkan/include
```

## Current implementation status

| Stage | Status | Backend |
|-------|--------|---------|
| CLI interface | Written | C++ |
| Weight loading (Parquet → DuckDB) | Written | C++ |
| GEMM compute shader | Written | Lean → Slang → SPIR-V |
| Conv2d compute shader | Written | Lean → Slang → SPIR-V |
| Attention compute shader | Written | Lean → Slang → SPIR-V |
| Layer norm / SiLU | Written | Lean → Slang → SPIR-V |
| Full UNet pipeline | Stub | Falls back to ggml |
| CLIP text encoder | Stub | Falls back to ggml |
| VAE decode | Stub | Falls back to ggml |
| Postprocess → PSD | Stub | C++ |