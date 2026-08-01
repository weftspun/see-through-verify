#!/usr/bin/env python3
"""Download see-through model weights from HuggingFace (safetensors, original format).

Usage:
  pixi run download-models

Output: hf_cache/<repo>/<path>.safetensors  (original dtype, no conversion)

The Lean Safetensors reader (Compute/Safetensors.lean) reads these directly.
"""

import argparse, json, os, time
from pathlib import Path
import requests

HF = "https://huggingface.co"

MODELS = [
    ("layerdifforg/seethroughv0.0.2_layerdiff3d", "layerdiff3d"),
    ("24yearsold/seethroughv0.0.1_marigold", "marigold"),
]


def hf_list(repo: str) -> list[str]:
    url = f"{HF}/api/models/{repo}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return [s["rfilename"] for s in r.json().get("siblings", [])]
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry listing {repo}: {e}")
            time.sleep(5)


def hf_download(repo: str, filename: str, cache: str) -> str:
    local = os.path.join(cache, repo.replace("/", "_"), filename)
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    url = f"{HF}/{repo}/resolve/main/{filename}"

    t0 = time.time()
    for attempt in range(3):
        try:
            print(f"  downloading {filename}...")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                down = 0
                with open(local, "wb") as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            down += len(chunk)
                            if total:
                                pct = down * 100 // total
                                print(f"\r    {down >> 20}MB / {total >> 20}MB ({pct}%)", end="")
                elapsed = time.time() - t0
                mb = os.path.getsize(local) / 1e6
                print(f"\n    done: {mb:.0f}MB in {elapsed:.0f}s ({mb / elapsed:.0f} MB/s)")
                return local
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {filename} ({attempt + 1}/3): {e}")
            time.sleep(10)
    raise RuntimeError(f"failed to download {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="hf_cache")
    args = parser.parse_args()

    for repo_id, _model_name in MODELS:
        print(f"\n=== {repo_id} ===")
        sf_list = [f for f in hf_list(repo_id) if f.endswith(".safetensors")]
        print(f"  safetensors files: {len(sf_list)}")
        total_mb = 0
        for sf_rel in sf_list:
            sf_path = hf_download(repo_id, sf_rel, args.cache_dir)
            total_mb += os.path.getsize(sf_path) / 1e6
        print(f"  total: {total_mb:.0f}MB in hf_cache/")

    print("\nDone. Lean reads these via Compute/Safetensors.lean.")


if __name__ == "__main__":
    main()