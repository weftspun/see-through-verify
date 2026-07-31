import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Model` — model graph descriptors for all pipeline models

Each structure captures the layer architecture of one model: input/output
shapes, channel counts per block, attention head counts, and block lists.
The runtime (harness) reads these to build the dispatch sequence and load
the correct Parquet weight tensors.

Reference: shitagaki-lab/see-through inference scripts.
-/

/-- CLIP text encoder (SDXL). -/
structure CLIPConfig where
  hidden : Nat := 768
  intermediate : Nat := 3072
  layers : Nat := 12
  heads : Nat := 12
  vocab : Nat := 49408
  maxPos : Nat := 77
deriving Inhabited

/-- CLIP text encoder 2 (SDXL, larger). -/
structure CLIP2Config where
  hidden : Nat := 1280
  intermediate : Nat := 5120
  layers : Nat := 24
  heads : Nat := 20
  vocab : Nat := 49408
  maxPos : Nat := 77
deriving Inhabited

/-- One UNet level: input/output channels + attention config. -/
structure UNetLevel where
  channels : Nat
  attentionLayers : Nat  -- how many transformer blocks at this level
  hasDownsample : Bool := true
deriving Inhabited

/-- LayerDiff UNet (3 levels: 320 → 640 → 1280). -/
structure LayerDiffConfig where
  baseChannels : Nat := 320
  levels : List UNetLevel := [
    { channels := 320, attentionLayers := 1, hasDownsample := true },
    { channels := 640, attentionLayers := 2, hasDownsample := true },
    { channels := 1280, attentionLayers := 10, hasDownsample := false },
  ]
  headDim : Nat := 64
  dualAttention : Bool := true  -- spatial + temporal attention
deriving Inhabited

/-- Marigold UNet (2 levels). -/
structure MarigoldConfig where
  baseChannels : Nat := 320
  levels : List UNetLevel := [
    { channels := 320, attentionLayers := 1, hasDownsample := true },
    { channels := 640, attentionLayers := 2, hasDownsample := true },
  ]
  headDim : Nat := 64
deriving Inhabited

/-- VAE config. -/
structure VAEConfig where
  inChannels : Nat := 3
  outChannels : Nat := 3  -- or 4 for latent space
  channels : List Nat     -- channel progression
  numResBlocks : Nat := 2
deriving Inhabited

/-- TransVAE (6-stage decoder). -/
def transVAE : VAEConfig :=
  { inChannels := 8, outChannels := 4, channels := [128, 256, 512, 512, 512, 512], numResBlocks := 2 }

/-- Marigold VAE (3-stage decoder). -/
def marigoldVAE : VAEConfig :=
  { inChannels := 4, outChannels := 3, channels := [128, 256, 512], numResBlocks := 2 }

/-- Standard SD VAE. -/
def sdVAE : VAEConfig :=
  { inChannels := 4, outChannels := 3, channels := [128, 256, 512, 512], numResBlocks := 2 }

/-- LaMa inpainting model config. -/
structure LaMaConfig where
  inChannels : Nat := 4
  outChannels : Nat := 3
  channels : List Nat := [32, 64, 128, 256, 512]
  numResBlocks : Nat := 2
deriving Inhabited

/-- Generate a human-readable summary of the model architecture. -/
def describeCLIP (cfg : CLIPConfig) : String :=
  s!"CLIP L/14: {cfg.hidden}h {cfg.intermediate}ff {cfg.layers}L {cfg.heads}H"

def describeUNet (cfg : LayerDiffConfig) : String :=
  let levelDesc := cfg.levels.map (fun l => s!"{l.channels}ch({l.attentionLayers}attn)")
  s!"LayerDiff UNet: {", ".join levelDesc} head_dim={cfg.headDim}"

def describeVAE (cfg : VAEConfig) : String :=
  s!"VAE: {cfg.inChannels}→{cfg.outChannels} ch=[{", ".join (cfg.channels.map toString)}]"

#eval describeCLIP (CLIPConfig.mk)
#eval describeUNet (LayerDiffConfig.mk)
#eval describeVAE transVAE