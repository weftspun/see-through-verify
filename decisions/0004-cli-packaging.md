# RFD 4 — CLI Packaging with fpm

**Status:** Draft  
**Author:** fire  
**Date:** 2026-07-31  

## Problem

The see-through CLI needs to be distributable as a single installable package for macOS users. Currently it exists as source files in the repo.

## Proposal

Use `fpm` (Effing Package Management) to create a macOS `.pkg` installer that bundles:

- CLI binary (`see-through`)
- Model weights (8 safetensors files, ~13GB)
- Lean compute shaders (as documentation)

The package installs to `/usr/local` with a standard layout.

## Implementation

### Package contents

```
/usr/local/bin/see-through          # CLI binary
/usr/local/share/see-through/hf_cache/  # model weights
/usr/local/share/see-through/shaders/   # Lean compute shader source
```

### CLI binary

The `Runtime/see-through.cpp` is a self-contained C++ program with:
- A minimal safetensors reader (no DuckDB dependency)
- CLI flags matching the original see-through-cpp
- `--demo` flag to inspect tensor metadata from any safetensors file

### Build

```bash
clang++ -std=c++17 Runtime/see-through.cpp -o see-through -O2
```

### Package

```bash
bash package.sh
# → see-through-0.1.0-1.osx.pkg
```

## Alternatives considered

- **Homebrew tap**: More complex setup, requires a formula. Preferred for final release but fpm is simpler for the initial packaging.
- **Docker image**: Not suitable for native macOS distribution.
- **Raw binary tarball**: Simpler but no standard install path.

## Next steps

1. Build the CLI binary
2. Run `--demo` to verify safetensors reading
3. Run `package.sh` to produce the .pkg
4. Distribute the .pkg to users