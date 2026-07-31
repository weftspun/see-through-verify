#!/usr/bin/env python3
"""Download see-through model weights from HuggingFace and convert to Parquet (zstd).

Self-contained — uses only requests + pyarrow + Python stdlib.
Runs under `pixi run download-models`.

Usage:
  pixi run download-models

Output: weights/<model>/<tensor_name>.parquet  (zstd-compressed, F32)

Models:
  - layerdifforg/seethroughv0.0.2_layerdiff3d
  - 24yearsold/seethroughv0.0.1_marigold
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Safetensor reader (pure Python)
# ---------------------------------------------------------------------------

DTYPE_TO_BYTES = {"F16": 2, "F32": 4, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}


def f16_to_f32(u16: int) -> float:
    """Convert IEEE 754 f16 bit pattern to f32."""
    sign = (u16 >> 15) & 1
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x3FF
    if exp == 0:
        return 0.0
    if exp == 31:
        return float("inf") if mant == 0 else float("nan")
    return (1.0 if sign == 0 else -1.0) * (2 ** (exp - 15)) * (1.0 + mant / 1024.0)


def bf16_to_f32(u16: int) -> float:
    """Convert brain-float 16 bit pattern to f32 (zero-extend upper 16 bits)."""
    u32 = u16 << 16
    return struct.unpack("<f", struct.pack("<I", u32))[0]


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Read safetensors header. Returns (metadata dict, header_size)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len)), 8 + header_len


def safetensors_to_parquet(sf_path: str, out_dir: str) -> None:
    """Convert all tensors in a safetensors file to individual Parquet files."""
    metadata, header_size = read_safetensors_header(sf_path)
    os.makedirs(out_dir, exist_ok=True)

    # Write metadata sidecar
    meta_path = os.path.join(out_dir, "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    for tensor_name, info in metadata.items():
        if tensor_name == "__metadata__":
            continue

        dtype = info["dtype"]
        shape = info["shape"]
        elem_bytes = DTYPE_TO_BYTES.get(dtype, 4)
        start, end = info["data_offsets"]
        nelems = (end - start) // elem_bytes

        with open(sf_path, "rb") as f:
            f.seek(header_size + start)
            raw = f.read(end - start)

        # Convert to F32
        if dtype == "F16":
            values = [f16_to_f32(struct.unpack("<H", raw[i * 2 : i * 2 + 2])[0]) for i in range(nelems)]
        elif dtype == "F32":
            values = [struct.unpack("<f", raw[i * 4 : i * 4 + 4])[0] for i in range(nelems)]
        elif dtype == "BF16":
            values = [bf16_to_f32(struct.unpack("<H", raw[i * 2 : i * 2 + 2])[0]) for i in range(nelems)]
        else:
            print(f"  skipping {tensor_name}: unsupported dtype {dtype}")
            continue

        # Write Parquet with zstd
        sanitized = tensor_name.replace(".", "_").replace("/", "_")
        parquet_path = os.path.join(out_dir, f"{sanitized}.parquet")
        print(f"  {tensor_name}  {dtype} {shape}  ({nelems} elems)  → {sanitized}.parquet")
        table = pa.table({"idx": range(nelems), "val": pa.array(values, type=pa.float32())})
        pq.write_table(table, parquet_path, compression="zstd", compression_level=19)

    print(f"  metadata → {meta_path}")


# ---------------------------------------------------------------------------
# HuggingFace download (self-contained, uses requests)
# ---------------------------------------------------------------------------

HF_BASE = "https://huggingface.co"

MODEL_REPOS = [
    ("layerdifforg/seethroughv0.0.2_layerdiff3d", "layerdiff3d"),
    ("24yearsold/seethroughv0.0.1_marigold", "marigold"),
]


def list_hf_files(repo_id: str) -> list[str]:
    """List all files in a HuggingFace model repo via the API."""
    url = f"{HF_BASE}/api/models/{repo_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    return [s["rfilename"] for s in siblings]


def download_safetensors(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a single safetensors file from HuggingFace, return local path."""
    local_path = os.path.join(cache_dir, repo_id.replace("/", "_"), filename)
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"{HF_BASE}/{repo_id}/resolve/main/{filename}"

    print(f"  downloading {filename}...")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)", end="")
    elapsed = time.time() - t0
    mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"\n    done: {mb:.0f}MB in {elapsed:.0f}s ({mb / elapsed:.0f} MB/s)")
    return local_path


def download_model(repo_id: str, model_name: str, cache_dir: str, out_root: str) -> None:
    """Download all safetensors for a model and convert to Parquet."""
    print(f"\n=== {repo_id} ({model_name}) ===")
    files = list_hf_files(repo_id)
    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    if not safetensors_files:
        print(f"  no safetensors found")
        return

    model_out = os.path.join(out_root, model_name)
    os.makedirs(model_out, exist_ok=True)

    for sf_rel in safetensors_files:
        sf_path = download_safetensors(repo_id, sf_rel, cache_dir)
        # Output subdirectory matches the relative path within the model
        sub_out = os.path.join(model_out, os.path.dirname(sf_rel))
        safetensors_to_parquet(sf_path, sub_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download see-through models → Parquet (zstd)")
    parser.add_argument("--cache-dir", default="hf_cache", help="Download cache directory")
    parser.add_argument("--out-dir", default="weights", help="Output directory for Parquet files")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for repo_id, model_name in MODEL_REPOS:
        download_model(repo_id, model_name, args.cache_dir, str(out_root))

    print("\nDone. Converted weights are in weights/")


if __name__ == "__main__":
    main()