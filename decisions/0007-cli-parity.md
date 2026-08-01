# CLI Parity — Critical Path (PERT)

Mermaid flowchart source for the CLI-parity critical path. Renders natively on
GitHub. Colors chosen in OKHSL (perceptually uniform): every role shares the
same lightness (fill L=0.34, border L=0.60, text L=0.82) and differs only by
hue — green=done, red=critical path, blue=pending, amber=optional.

```mermaid
%%{init: {"theme": "base", "themeVariables": {
  "background": "#16213e",
  "fontFamily": "SF Mono, Menlo, Monaco, monospace"
}} }%%
flowchart LR
    %% Foundation (done)
    subgraph Foundation["FOUNDATION — 4/4 complete"]
        G0[GEMM shader<br/>Slang→Metal ✓]:::done
        V0[Vulkan compute<br/>max_err=0.0 ✓]:::done
        W0[Weight loader<br/>safetensors→BF16 ✓]:::done
        C0[CLI interface<br/>--demo ✓]:::done
    end

    %% Shader coverage — parallel
    A[<b>A · CLIP text encoder</b><br/>12× GEMM + attn + norm<br/>2 files · 1.5GB]:::onpath
    B[<b>B · LayerDiff UNet body</b><br/>3-level + transformer blocks<br/>2452 tensors · 7.6GB]:::onpath
    C[<b>C · Marigold depth</b><br/>2-level depth estimation<br/>990 tensors · 2.3GB]:::onpath
    D[<b>D · VAE decoder</b><br/>trans/sd/marigold-vae<br/>1052 tensors]:::offpath

    Foundation --> A
    Foundation --> B
    Foundation --> C
    Foundation --> D

    %% Critical path
    E[<b>E · Dispatch runner</b><br/>graph → shader dispatch]:::flow
    F[<b>F · Diffusion scheduler</b><br/>DDIM · 30 steps × 2 passes]:::flow
    G[<b>G · Layer postprocess</b><br/>depth → bbox → PSD]:::flow
    H[<b>H · PSD writer</b><br/>reuse psd_sdk]:::flow
    I[<b>I · Full CLI flags</b><br/>50+ args]:::offpath

    A --> E
    B --> E
    C --> E
    D -. optional .-> E

    E --> F --> G
    G --> H
    G -.-> I

    %% Milestones
    M1([<b>M1 · CLIP encode</b><br/>GEMM + attn validation · DONE]):::done
    M2([<b>M2 · UNet step</b><br/>one diffusion step on GPU]):::flow
    P([<b>CLI PARITY</b><br/>full pipeline ≈ reference PSD]):::flow

    E --> M1
    G --> M2
    H --> P
    I --> P

    classDef done fill:#00470f,stroke:#00b000,color:#00ed64
    classDef onpath fill:#68111f,stroke:#ff0017,color:#ff6d86
    classDef flow fill:#68111f,stroke:#ff0017,color:#ff6d86
    classDef offpath fill:#532f00,stroke:#dc4e00,color:#ffb000
```
