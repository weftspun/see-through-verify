# Lean Shader Slang See-Through Runtime Harness

## Overview

The runtime harness is a C++ program that:
1. Reads model weights from Parquet files (via DuckDB)
2. Loads Slang-compiled SPIR-V shaders
3. Uploads weights to Vulkan buffers
4. Dispatches compute shaders in pipeline order
5. Reads back results and writes PSD output

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Runtime Harness                 │
│  (C++, Vulkan, DuckDB C API)                    │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Parquet  │  │  Slang   │  │  Slangc       │  │
│  │ Loader   │  │ Compiler │  │  (external)   │  │
│  │ (DuckDB) │  │ (.slang  │  │  .slang→.spv  │  │
│  │          │  │  →.spv)  │  │               │  │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │             │                │          │
│       ▼             ▼                ▼          │
│  ┌──────────────────────────────────────────┐   │
│  │           Vulkan Compute Engine          │   │
│  │  - VkBuffer upload (weights)             │   │
│  │  - VkDescriptorSet binding               │   │
│  │  - VkCmdDispatch (pipeline order)        │   │
│  └──────────────────────────────────────────┘   │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐                                   │
│  │  Output  │                                   │
│  │  (PSD)   │                                   │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

## Pipeline Stages

The see-through pipeline runs in order:

1. **CLIP encode** — text embeddings → conditioning
2. **LayerDiff UNet** — 30-step diffusion (body pass)
3. **LayerDiff UNet** — head pass (refinement)
4. **Marigold depth** — pseudo-depth estimation
5. **VAE decode** — latent → pixel space
6. **Postprocess** — layer extraction → PSD

Each stage is a sequence of compute shader dispatches (GEMM, conv2d, attention, norm, activation).

## Usage

```bash
# Compile Slang shaders to SPIR-V
for f in shaders/*.slang; do
  slangc -target spirv -o ${f%.slang}.spv "$f"
done

# Run the pipeline
./see-through --input image.png --output out.psd --steps 30 \
  --weights-dir weights/
```

## Dependencies

- Vulkan SDK (for `slangc` and Vulkan headers)
- DuckDB C API (for Parquet reading)
- slangc (included in Vulkan SDK)