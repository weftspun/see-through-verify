#!/bin/bash
# Single no-arg build: compiles the full validation harness set + the shader-
# execution CLI from one command. CI/CD ideal: a fresh checkout builds
# everything here with no flags.
#
#   ./build.sh        # build everything (default, release -O3)
#   ./build.sh fast   # build everything (no -O3, quicker for iteration)
#
# CPU (Accelerate-BLAS) harnesses are VALIDATION ONLY; they are never linked
# into the release CLI (Runtime/see-through.cpp, built by package.sh). Only the
# shader execution path ships in release.
set -euo pipefail

OPT="-O3"
if [ "${1:-}" = "fast" ]; then OPT="-O0"; fi
OUT=/tmp/see-through-build
mkdir -p "$OUT"

CXX=clang++
STD="-std=c++17 $OPT"
BLAS="-framework Accelerate"

echo "=== Building CPU validation harnesses (Accelerate-BLAS, validation-only) ==="
# full suite: primitives + parity guards + chained reference harnesses

build() { # name  src...
    local name=$1; shift
    $CXX $STD "$@" -o "$OUT/$name" $BLAS 2> /tmp/${name}.log \
        && echo "  ok  $name" \
        || { echo "  FAIL $name"; cat /tmp/${name}.log; exit 1; }
}

build parity_mha             parity_mha.cpp
build parity_gemm            parity_gemm.cpp
build slang_mha_compare      slang_mha_compare.cpp
build slang_geglu_compare    slang_geglu_compare.cpp
build slang_gn_compare       slang_gn_compare.cpp
build slang_conv_compare     slang_conv_compare.cpp
build slang_prims_compare    slang_prims_compare.cpp
build unet_t3d_compare       unet_t3d_compare.cpp
build temporal_compare       temporal_compare.cpp
build btblock_compare        btblock_compare.cpp
build btblock_stage_compare  btblock_stage_compare.cpp

echo "=== Building shader-execution CLI (Vulkan/Metal, ships in release) ==="
$CXX $STD Runtime/harness.cpp -o "$OUT/harness" \
    -I/opt/homebrew/include -L/opt/homebrew/lib \
    -lvulkan -framework Cocoa -framework Metal -framework QuartzCore 2>&1 \
    && echo "  ok  harness (Vulkan/Metal dispatch)" \
    || echo "  (harness needs Vulkan SDK; skipped — see build-harness.sh)"

echo ""
echo "All binaries in $OUT"
