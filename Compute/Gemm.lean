import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Gemm` — f32×f32 GEMM compute shader

A direct (non-tiled) matrix multiply: `C[M, N] = A[M, K] × B[K, N]`.
Each thread computes one element of C. The loop reduction over K is
the innermost loop.

This is the foundation for every matmul in the see-through pipeline:
conv2d (im2col → GEMM), attention (QK^T, PV), and linear layers
(ffn, proj). Tiling, groupshared memory, and shared-memory blocking
will be added in subsequent iterations.
-/

/-- Build a direct GEMM compute shader module.

Parameters `M`, `N`, `K` are the matrix dimensions passed as
constants so the shader can be specialized at compile time.
Dispatcher dimensions: `ceil(M / BM) × ceil(N / BN) × 1` where
`BM` and `BN` are the threadgroup blocking factors.
-/
def gemmShader (BM BN : Nat) (M N K : SlangExpr) : SlangShaderModule :=
  { globals := [
      { name := "A", type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
    , { name := "B", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
    , { name := "C", type := .rwBuf (.scalar .float), binding := some 2, space := some 0 }
    , { name := "M", type := .scalar .uint, binding := some 3, space := some 0 }
    , { name := "N", type := .scalar .uint, binding := some 4, space := some 0 }
    , { name := "K", type := .scalar .uint, binding := some 5, space := some 0 }
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads BM BN 1]
      , name := "gemm"
      , params := [
          ⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩
        ]
      , body := [
          .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
        , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
        , -- for (uint k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
          .forCount "k" (.litUint 0) (.var "K")
            [ .assign (.var "sum")
                (.bin "+" (.var "sum")
                  (.bin "*"
                    (.index (.var "A") (.bin "+" (.bin "*" (.var "row") (.var "K")) (.var "k")))
                    (.index (.var "B") (.bin "+" (.bin "*" (.var "k") (.var "N")) (.var "col"))))) ]
        , -- C[row * N + col] = sum;
          .assign
            (.index (.var "C") (.bin "+" (.var "row") (.bin "*" (.var "col") (.var "N"))))
            (.var "sum")
        ]
      }
    ]
  }

/-- Emit the GEMM shader as Slang source text. -/
def emitGemmShader (BM BN : Nat) (M N K : SlangExpr) : String :=
  LeanSlang.emit (gemmShader BM BN M N K)

/-- Emit with literal dimensions (for testing / codegen). -/
def emitGemmShaderLit (BM BN M N K : Nat) : String :=
  emitGemmShader BM BN (.litUint M) (.litUint N) (.litUint K)