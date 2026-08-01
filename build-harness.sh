#!/bin/bash
# Build the Vulkan compute dispatch harness and test the gemm shader
set -euo pipefail

echo "=== Building Vulkan compute dispatch ==="
clang++ -std=c++17 Runtime/harness.cpp -o /tmp/gemm-test \
  -I/opt/homebrew/include \
  -L/opt/homebrew/lib \
  -lvulkan -framework Cocoa -framework Metal -framework QuartzCore \
  -O2 2>&1

echo "=== Testing gemm shader ==="
# Create a small test input: load a weight tensor, run gemm
# For now, generate random data
dd if=/dev/urandom bs=64 count=64 of=/tmp/a_input.bin 2>/dev/null
dd if=/dev/urandom bs=64 count=64 of=/tmp/b_input.bin 2>/dev/null

/tmp/gemm-test /tmp/a_input.bin shaders/gemm.spv 64 64 64 2>&1 | head -5

echo "=== Done ==="