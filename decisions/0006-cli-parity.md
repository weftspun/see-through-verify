# RFD 6 — CLI Parity Critical Path (PERT)

**Status:** Published
**Author:** fire
**Date:** 2026-08-01
**License:** Apache 2.0

See the companion PERT diagram: `decisions/0007-cli-parity-pert.svg`.

## Problem

`see-through-verify` reimplements the `weftspun/see-through-cpp` diffusion
pipeline (anime illustration → layered PSD). To reach CLI parity — the same
image-format flags and byte-identical reference PSD — the pipeline must be
wired model-by-model along a single critical path. A PERT plan maps the
dependencies so each milestone produces a working artifact before the next
begins (Gall's Law).

## Critical path

The governing sequence is **A → E → F → G → H**:

| Node | Work | Status |
|------|------|--------|
| A | CLIP text encoder (12× GEMM + attn + norm) | GREEN — bit-for-bit encoder match vs ggml (output `cliped` cos=1.0) |
| B | LayerDiff UNet body (3-level, transformer blocks, 2452 tensors, 7.6 GB) | In progress — all primitives GREEN |
| C | Marigold depth UNet (2-level, 990 tensors) | Pending |
| D | VAE decoders (trans-vae, sd-vae, marigold-vae, 1052 tensors) | Pending |
| E | Dispatch runner (graph → sequence of shader dispatches) | GREEN |
| F | DDIM scheduler (30-step loop, 2 UNet passes) | GREEN |
| G | Layer postprocess (depth reordering → bbox → PSD) | Partial — CLIP validated |
| H | PSD writer (reuse psd_sdk) | Pending |
| I | Full CLI flags (50+, `--steps --res --split-depth --device`) | Pending |

Each model's layers are a sequence of GEMM + attention + norm + activation
dispatches. The dispatch runner reads weight bindings from
`Compute/Model.lean` descriptors.

## Milestones

- **M1 — CLIP encode only**: GEMM + attention validation. Done.
- **M2 — one UNet diffusion step on GPU**: the next gate; validated when a
  single step's output matches the ggml oracle.
- **CLI PARITY**: full pipeline output matching the reference PSD.

## Kernel status (verified against ggml oracle taps, cos ≥ 0.999)

All implementation kernels are authored in Slang and exported to C++ via
`slangc -target cpp` (per-thread, no barriers), then validated against ggml
oracle taps — no hand-coded C++:

| Kernel | shader | status |
|--------|--------|--------|
| GEMM | `shaders/gemm.slang` | GREEN |
| MHA attention | `shaders/mha.slang` | GREEN (cos=1.0) |
| LayerNorm | `shaders/layernorm.slang` | GREEN |
| GEGLU | `shaders/geglu.slang` | GREEN (cos=0.999997) |
| Conv2d | `shaders/conv2d.slang` | GREEN |
| GroupNorm | `shaders/groupnorm.slang` | GREEN |

## Next steps

1. Compose the Slang UNet forward (down/mid/up, resnet + time-embed,
   transformer3d, temporal cross-frame) from the exported kernels → M2 GREEN.
2. Wire Marigold + VAE decode + postprocess → PSD output → CLI parity.

See also: `decisions/0001` (pipeline architecture), `decisions/0002`
(runtime harness).
