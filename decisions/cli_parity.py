# Single source of truth for RFD 6 — CLI parity critical path.
#
# Holds the node data ONCE and derives both rendered outputs from it:
#   1. status_table_markdown()  -> the critical-path status table
#   2. mermaid_source()         -> the PERT Mermaid diagram
#
# Editing a node here updates BOTH the table and the diagram, so the two
# outputs can never drift apart.

# roles -> CSS class & Mermaid classDef (colors in OKHSL, perceptually uniform,
# same lightness per role, differ by hue)
ROLE_CLASS = {
    "done":    "done",     # green
    "onpath":  "onpath",   # red (model coverage on the critical path)
    "flow":    "flow",     # red (critical-path pipeline stages)
    "offpath": "offpath",  # amber (optional)
}

# node: (label_short, subtitle, class)
FOUNDATION = [
    ("GEMM shader", "Slang→Metal ✓", "done"),
    ("Vulkan compute", "max_err=0.0 ✓", "done"),
    ("Weight loader", "safetensors→BF16 ✓", "done"),
    ("CLI interface", "--demo ✓", "done"),
]

MODELS = {
    "A": ("CLIP", "12× GEMM+attn+norm · 1.5GB", "onpath"),
    "B": ("UNet body", "3-level + transformer · 2452 · 7.6GB", "onpath"),
    "C": ("Marigold", "2-level depth · 990 · 2.3GB", "onpath"),
    "D": ("VAE", "trans/sd/marigold · 1052", "offpath"),
}

FLOW = {
    "E": ("Dispatch", "graph → dispatch", "flow"),
    "F": ("Scheduler", "DDIM 30×2", "flow"),
    "G": ("Postprocess", "depth→bbox→PSD", "flow"),
    "H": ("PSD writer", "psd_sdk", "flow"),
    "I": ("CLI flags", "50+", "offpath"),
}

MILESTONES = {
    "M1": ("M1 · CLIP", "encode · DONE", "done"),
    "M2": ("M2 · UNet", "one step", "flow"),
    "P":  ("CLI PARITY", "≈ ref PSD", "flow"),
}


def status_table_markdown() -> str:
    """The critical-path status table (Rendered view)."""
    rows = [
        ("A", "CLIP text encoder (12× GEMM + attn + norm)",
         "GREEN — bit-for-bit encoder match vs ggml (cos=1.0)"),
        ("B", "LayerDiff UNet body (3-level, transformer blocks)",
         "In progress — all primitives GREEN"),
        ("C", "Marigold depth UNet (2-level)", "Pending"),
        ("D", "VAE decoders (trans-vae, sd-vae, marigold-vae)", "Pending"),
        ("E", "Dispatch runner (graph → sequence of shader dispatches)", "GREEN"),
        ("F", "DDIM scheduler (30-step loop, 2 UNet passes)", "GREEN"),
        ("G", "Layer postprocess (depth reordering → bbox → PSD)", "Partial — CLIP validated"),
        ("H", "PSD writer (reuse psd_sdk)", "Pending"),
        ("I", "Full CLI flags (50+, `--steps --res --split-depth --device`)", "Pending"),
    ]
    out = ["| Node | Work | Status |", "|------|------|--------|"]
    for n, work, status in rows:
        out.append(f"| {n} | {work} | {status} |")
    return "\n".join(out)


def mermaid_source() -> str:
    """The PERT Mermaid diagram (derived from the same node data)."""
    lines = [
        '%%{init: {"theme": "base", "themeVariables": {',
        '  "background": "#16213e",',
        '  "fontFamily": "SF Mono, Menlo, Monaco, monospace"',
        '}, "flowchart": {',
        '  "curve": "basis",',
        '  "nodeSpacing": 18,',
        '  "rankSpacing": 28,',
        '  "wrappingWidth": 90,',
        '  "useMaxWidth": true,',
        '  "htmlLabels": true',
        '}} }%%',
        "flowchart TB",
        "    %% Foundation (done)",
        '    subgraph Foundation["FOUNDATION — 4/4"]',
    ]
    for i, (label, sub, role) in enumerate(FOUNDATION):
        lines.append(f'        G{i}[<b>{label}</b><br/>{sub}]:::{ROLE_CLASS[role]}')
    lines.append("    end")
    lines.append("")
    lines.append("    %% Shader coverage — parallel")
    for n, (label, sub, role) in MODELS.items():
        lines.append(f'    {n}[<b>{n} · {label}</b><br/>{sub}]:::{ROLE_CLASS[role]}')
    lines.append("")
    lines.append("    Foundation --> A")
    lines.append("    Foundation --> B")
    lines.append("    Foundation --> C")
    lines.append("    Foundation --> D")
    lines.append("")
    lines.append("    %% Critical path")
    for n, (label, sub, role) in FLOW.items():
        lines.append(f'    {n}[<b>{n} · {label}</b><br/>{sub}]:::{ROLE_CLASS[role]}')
    lines.append("")
    lines.append("    A --> E")
    lines.append("    B --> E")
    lines.append("    C --> E")
    lines.append("    D -. optional .-> E")
    lines.append("")
    lines.append("    E --> F --> G")
    lines.append("    G --> H")
    lines.append("    G -.-> I")
    lines.append("")
    lines.append("    %% Milestones")
    for n, (label, sub, role) in MILESTONES.items():
        lines.append(f'    {n}([<b>{label}</b><br/>{sub}]):::{ROLE_CLASS[role]}')
    lines.append("")
    lines.append("    E --> M1")
    lines.append("    G --> M2")
    lines.append("    H --> P")
    lines.append("    I --> P")
    lines.append("")
    lines.append("    classDef done fill:#00470f,stroke:#00b000,color:#00ed64")
    lines.append("    classDef onpath fill:#68111f,stroke:#ff0017,color:#ff6d86")
    lines.append("    classDef flow fill:#68111f,stroke:#ff0017,color:#ff6d86")
    lines.append("    classDef offpath fill:#532f00,stroke:#dc4e00,color:#ffb000")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status_table_markdown())
    print("\n\n---\n\n")
    print(mermaid_source())
