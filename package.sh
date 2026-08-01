#!/bin/bash
# Package see-through CLI with fpm for macOS
# Note: model weights (~13GB) are NOT bundled — they're downloaded separately
# via `pixi run download-models` or `see-through --download-models`
set -euo pipefail

PKG_DIR="/tmp/see-through-pkg"
BINDIR="$PKG_DIR/usr/local/bin"
SHAREDIR="$PKG_DIR/usr/local/share/see-through"

rm -rf "$PKG_DIR"
mkdir -p "$BINDIR" "$SHAREDIR"

echo "=== Building CLI ==="
clang++ -std=c++17 Runtime/see-through.cpp -o "$BINDIR/see-through" -O2

echo "=== Copying metadata ==="
cp LICENSE "$SHAREDIR/" 2>/dev/null || true
cp RFD/*.md "$SHAREDIR/" 2>/dev/null || true

echo "=== Packaging with fpm ==="
fpm -s dir -t osxpkg \
  --name "see-through" \
  --version "0.1.0" \
  --description "Single-image layer decomposition for anime characters" \
  --license "Apache-2.0" \
  --vendor "weftspun" \
  --url "https://github.com/weftspun/see-through-verify" \
  --after-install <(echo 'echo "Run: see-through --help"; echo "Weights: cd /usr/local/share/see-through && pixi run download-models"') \
  -C "$PKG_DIR" \
  .

echo "=== Done ==="
ls -lh *.pkg 2>/dev/null || echo "package created"