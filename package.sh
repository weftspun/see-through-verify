#!/bin/bash
# Package see-through CLI with fpm for macOS
set -euo pipefail

PROJECT="see-through"
VERSION="0.1.0"
DESCRIPTION="Single-image layer decomposition for anime characters"
PKG_DIR="/tmp/see-through-pkg"

# Build the CLI
echo "=== Building CLI ==="
clang++ -std=c++17 Runtime/see-through.cpp -o "$PKG_DIR/usr/local/bin/see-through" \
  -O2 -Wno-c++17-extensions 2>&1

# Copy weights
echo "=== Copying weights ==="
mkdir -p "$PKG_DIR/usr/local/share/see-through/hf_cache"
cp -r hf_cache/* "$PKG_DIR/usr/local/share/see-through/hf_cache/"

# Copy Lean compute shaders (as documentation)
echo "=== Copying shaders ==="
mkdir -p "$PKG_DIR/usr/local/share/see-through/shaders"
cp Compute/*.lean "$PKG_DIR/usr/local/share/see-through/shaders/" 2>/dev/null || true

# Package with fpm
echo "=== Packaging ==="
fpm -s dir -t osxpkg \
  --name "$PROJECT" \
  --version "$VERSION" \
  --description "$DESCRIPTION" \
  --license "Apache-2.0" \
  --vendor "weftspun" \
  --url "https://github.com/weftspun/see-through-verify" \
  -C "$PKG_DIR" \
  .

echo "=== Done ==="
ls -lh *.pkg 2>/dev/null || ls -lh *.tar.* 2>/dev/null || echo "package created"