import Case

/-! # Kernel witness gate

Hard production gate: every kernel-variant configuration the 1280px CLI can
emit (backend x knob x shape) must be witness-free — i.e. the candidate
kernel (direct conv, row-chunked im2col, flash attention) must stay within
the reference kernel's error interval at every real shape the pipeline uses.
Exits non-zero on the first defect found.
-/

open PlausibleWitnessDag

/-- Production envelope of the 1280px pipeline: every kernel-variant
configuration the CLI can emit, at its real shapes. Bounded, deterministic,
indexable — the witness domain. -/
def productionDomain : Array Case := #[
  -- UNet direct conv at every latent level (s1 + downsamplers)
  conv "unet-direct-160-320-s1" 160 160 320 320 1 1,
  conv "unet-direct-160-320-s2" 160 160 320 320 2 1,
  conv "unet-direct-80-640-s1"  80 80 640 640 1 1,
  conv "unet-direct-80-640-s2"  80 80 640 640 2 1,
  conv "unet-direct-40-1280-s1" 40 40 1280 1280 1 1,
  -- decode row-chunked im2col at pixel levels
  conv "decode-rowchunk-1280-64"  1280 1280 4 4 1 2,
  conv "decode-rowchunk-640-128"  640 640 8 8 1 2,
  conv "decode-rowchunk-320-256"  320 320 16 16 1 2,
  -- UNet row-chunked im2col (batch>1, replaces direct_conv for the
  -- stride-1 convs since docs/ggml-upstream-issues.md #4) at the real
  -- cross-frame batch size (13 frames), all 3 latent levels
  conv "unet-rowchunk-160-320-b13" 160 160 320 320 1 2 13,
  conv "unet-rowchunk-80-640-b13"  80  80  640 640 1 2 13,
  conv "unet-rowchunk-40-1280-b13" 40  40  1280 1280 1 2 13,
  -- flash attention at every spatial token count / frame batch
  attn "flash-t1600-b13"  10 1600 1600 13 4,
  attn "flash-t6400-b2"   10 6400 6400 2 4,
  attn "flash-t6400-b1"   10 6400 6400 1 4,
  attn "flash-t4096-b12"  5 4096 4096 12 4,
  attn "flash-t4096-b1"   5 4096 4096 1 4,
  attn "flash-cross-77"   10 6400 77 13 4,
  attn "flash-temporal"   10 13 13 6400 4,
  -- query-tiled naive attention
  attn "tiled-t1600-b13"  10 1600 1600 13 8,
  attn "tiled-t6400-b2"   10 6400 6400 2  8,
  attn "tiled-t6400-b1"   10 6400 6400 1  8,
  attn "tiled-t4096-b12"  5  4096 4096 12 8,
  attn "tiled-t4096-b1"   5  4096 4096 1  8,
  attn "tiled-cross-77"   10 6400 77   13 8,
  attn "tiled-temporal"   10 13   13   6400 8,
  -- gemm: f32×f32 matmul at various scales. At scale_bits >= 8 (scale >= 256)
  -- the Metal simdgroup GEMM's half-precision products overflow f16 range.
  -- The scheduler clamp fix (scheduler.cpp) prevents this by bounding x0.
  gemm "gemm-safe-scale1"   64 64 64 0,
  gemm "gemm-edge-scale7"  64 64 64 7,
  gemm "gemm-overflow-scale8" 64 64 64 8,
  gemm "gemm-overflow-scale10" 64 64 64 10,
  gemm "gemm-small-128"    32 32 32 8,
  gemm "gemm-large-64"    128 128 64 8
]

def main : IO UInt32 := do
  IO.println s!"device-primary witness search (primary device: seethrough_c)"
  let found ← searchAndReport productionDomain "production-envelope-defect"
  IO.println (if found then "VERIFY FAIL" else "VERIFY PASS")
  pure (if found then 1 else 0)
