#!/usr/bin/env python3
"""Download see-through model weights from HuggingFace and convert to Parquet (zstd).

Usage:
  python tools/download-models.py                   # download all models
  python tools/download-models.py --convert-only     # convert cached safetensors to Parquet

Output: weights/<model>/<tensor_name>.parquet  (zstd-compressed, F32)

Models:
  - layerdifforg/seethroughv0.0.2_layerdiff3d  (LayerDiff 3D UNet + text encoders + VAE)
  - 24yearsold/seethroughv0.0.1_marigold        (Marigold depth UNet + VAE)
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Safetensor reader (no PyTorch dependency — pure Python)
# ---------------------------------------------------------------------------

def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Read safetensors header without loading tensor data.

    Returns (metadata dict, header_size_in_bytes).
    The metadata dict maps tensor_name -> {dtype, shape, data_offsets}.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    metadata: dict = json.loads(header_bytes)
    return metadata, 8 + header_len


def extract_tensor(path: str, tensor_name: str) -> bytes:
    """Extract raw tensor bytes from a safetensors file."""
    metadata, header_size = read_safetensors_header(path)
    info = metadata.get(tensor_name)
    if info is None:
        raise KeyError(f"tensor '{tensor_name}' not found in {path}")
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(header_size + start)
        return f.read(end - start)


DTYPE_TO_ELEM_BYTES = {
    "F16": 2,
    "F32": 4,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
}

# ---------------------------------------------------------------------------
# Weight → Parquet writer (minimal, no Arrow dependency)
# ---------------------------------------------------------------------------

def write_parquet_zstd(path: str, values: list[float]) -> None:
    """Write a single-column Parquet file with zstd compression.

    Schema: idx INT64, val FLOAT.
    Uses the Parquet API format directly (no library dependency).
    This is a minimal Parquet writer — for production use, use PyArrow.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"idx": range(len(values)), "val": pa.array(values, type=pa.float32())})
    pq.write_table(table, path, compression="zstd", compression_level=19)


def safetensors_to_parquet(safetensors_path: str, out_dir: str) -> None:
    """Convert all tensors in a safetensors file to individual Parquet files."""
    metadata, header_size = read_safetensors_header(safetensors_path)
    os.makedirs(out_dir, exist_ok=True)

    # Write metadata
    meta_path = os.path.join(out_dir, "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    for tensor_name, info in metadata.items():
        if tensor_name == "__metadata__":
            continue

        dtype = info["dtype"]
        shape = info["shape"]
        elem_bytes = DTYPE_TO_ELEM_BYTES.get(dtype, 4)

        raw = extract_tensor(safetensors_path, tensor_name)
        nelems = len(raw) // elem_bytes

        # Convert to F32
        if dtype == "F16":
            import struct as st
            values = []
            for i in range(nelems):
                u16 = st.unpack("<H", raw[i * 2 : i * 2 + 2])[0]
                # f16 → f32
                sign = (u16 >> 15) & 1
                exp = (u16 >> 10) & 0x1F
                mant = u16 & 0x3FF
                if exp == 0:
                    f32 = 0.0
                elif exp == 31:
                    f32 = float("inf") if mant == 0 else float("nan")
                else:
                    f32 = (1.0 if sign == 0 else -1.0) * (2 ** (exp - 15)) * (1.0 + mant / 1024.0)
                values.append(f32)
        elif dtype == "F32":
            import struct as st
            values = list(st.iter_unpack("<f", raw))
            values = [v[0] for v in values]
        elif dtype == "BF16":
            import struct as st
            values = []
            for i in range(nelems):
                u16 = st.unpack("<H", raw[i * 2 : i * 2 + 2])[0]
                # bf16 → f32 (zero-extend the upper 16 bits)
                u32 = u16 << 16
                f32 = st.unpack("<f", struct.pack("<I", u32))[0]
                values.append(f32)
        else:
            print(f"  skipping {tensor_name}: unsupported dtype {dtype}")
            continue

        # Write Parquet
        sanitized = tensor_name.replace(".", "_").replace("/", "_")
        parquet_path = os.path.join(out_dir, f"{sanitized}.parquet")
        print(f"  {tensor_name} {dtype} {shape} ({nelems} elems) → {parquet_path}")
        write_parquet_zstd(parquet_path, values)


# ---------------------------------------------------------------------------
# Download from HuggingFace
# ---------------------------------------------------------------------------

def download_hf_model(repo_id: str, cache_dir: str) -> str:
    """Download a HuggingFace model using huggingface_hub.

    Returns the path to the downloaded model directory.
    """
    from huggingface_hub import snapshot_download

    print(f"downloading {repo_id} to {cache_dir}...")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    return path


def find_safetensors(model_dir: str) -> list[str]:
    """Find all .safetensors files in a model directory."""
    result = []
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".safetensors"):
                result.append(os.path.join(root, f))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = [
    ("layerdifforg/seethroughv0.0.2_layerdiff3d", "layerdiff3d"),
    ("24yearsold/seethroughv0.0.1_marigold", "marigold"),
]


def main():
    parser = argparse.ArgumentParser(description="Download see-through models and convert to Parquet")
    parser.add_argument("--cache-dir", default="hf_cache", help="HuggingFace cache directory")
    parser.add_argument("--out-dir", default="weights", help="Output directory for Parquet files")
    parser.add_argument("--convert-only", action="store_true", help="Skip download, convert cached safetensors only")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for repo_id, model_name in MODELS:
        print(f"\n=== {repo_id} ({model_name}) ===")

        if not args.convert_only:
            model_dir = download_hf_model(repo_id, args.cache_dir)
        else:
            # Find cached model directory
            from huggingface_hub import scan_cache_dir

            cache = scan_cache_dir(args.cache_dir)
            revisions = cache.get_repo(repo_id)
            if not revisions:
                print(f"  {repo_id} not in cache, skipping")
                continue
            # Use the most recent revision
            rev = sorted(revisions, key=lambda r: r.last_accessed)[-1]
            model_dir = str(rev.snapshot_path)
            print(f"  cached at {model_dir}")

        safetensors_files = find_safetensors(model_dir)
        if not safetensors_files:
            print(f"  no safetensors found in {model_dir}")
            continue

        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        for sf_path in safetensors_files:
            rel_path = os.path.relpath(sf_path, model_dir)
            file_out = model_out / rel_path.replace(".safetensors", "")
            print(f"  converting {rel_path}...")
            safetensors_to_parquet(sf_path, str(file_out))

        print(f"  done → {model_out}/")

    print("\nAll done. Converted weights are in weights/")


if __name__ == "__main__":
    main()