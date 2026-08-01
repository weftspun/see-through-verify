#!/bin/bash
# Package see-through as single binary + weights/ folder
set -euo pipefail

VERSION="0.1.0"
OUT="see-through-${VERSION}-macos-arm64.tar"

rm -rf /tmp/see-through-dist
mkdir -p /tmp/see-through-dist

echo "=== Building CLI ==="
clang++ -std=c++17 Runtime/see-through.cpp -o /tmp/see-through-dist/see-through -O2

echo "=== Copying weights ==="
if [ -d hf_cache ] && [ "$(find hf_cache -name '*.safetensors' 2>/dev/null | wc -l)" -gt 0 ]; then
  mkdir -p /tmp/see-through-dist/weights
  cp -r hf_cache/* /tmp/see-through-dist/weights/
fi

echo "=== Creating tarball (no compression for speed) ==="
cd /tmp/see-through-dist
tar cf "/tmp/$OUT" --zstd .
cd - >/dev/null
cp "/tmp/$OUT" .
echo "Done: $(ls -lh "$OUT" | awk '{print $5}') — $OUT"

echo ""
echo "Extract and run:"
echo "  tar xf $OUT"
echo "  cd see-through-${VERSION}-macos-arm64"
echo "  ./see-through -i image.png -o out.psd"