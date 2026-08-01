import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Gemm` — tiled f32×f32 GEMM compute shader

Tiled matrix multiply: `C[M, N] = A[M, K] × B[K, N]`. Each threadgroup
computes a `BM × BN` tile of C, loading `BM × BK` of A and `BK × BN`
of B into groupshared memory with barriers between phases.

BM = BN = BK = 32 matches Apple M2 Pro's 1024-thread limit and gives
full thread utilization for tiled loads.
-/

def BM : Nat := 32
def BN : Nat := 32
def BK : Nat := 32

def gemmShader (M N K : SlangExpr) : SlangShaderModule :=
  { globals := [
      { name := "A", type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
    , { name := "B", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
    , { name := "C", type := .rwBuf (.scalar .float), binding := some 2, space := some 0 }
    , { name := "M", type := .scalar .uint, binding := some 3, space := some 0 }
    , { name := "N", type := .scalar .uint, binding := some 4, space := some 0 }
    , { name := "K", type := .scalar .uint, binding := some 5, space := some 0 }
    ]
  , groupShared := [
      { name := "sA", elemType := .scalar .float, dims := [BM * BK] }
    , { name := "sB", elemType := .scalar .float, dims := [BK * BN] }
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads BM BN 1]
      , name := "gemm"
      , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      , body := [
          .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
        , .declare (.scalar .uint) "tx" (some (.bin "%" (.var "row") (.litUint BM)))
        , .declare (.scalar .uint) "ty" (some (.bin "%" (.var "col") (.litUint BN)))
        , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
        , .declare (.scalar .uint) "numTiles" (some (.bin "/" (.bin "+" (.var "K") (.litUint (BK - 1))) (.litUint BK)))
        , .forCount "kkTile" (.litUint 0) (.var "numTiles")
            [ .declare (.scalar .uint) "kk" (some (.bin "*" (.var "kkTile") (.litUint BK)))
            , -- load A tile: sA[tx * BK + ty] = A[row * K + kk + ty]
              .assign
                (.index (.var "sA") (.bin "+" (.bin "*" (.var "tx") (.litUint BK)) (.var "ty")))
                (.index (.var "A") (.bin "+" (.bin "*" (.var "row") (.var "K")) (.bin "+" (.var "kk") (.var "ty"))))
            , -- load B tile: sB[tx * BN + ty] = B[(kk + tx) * N + col]
              .assign
                (.index (.var "sB") (.bin "+" (.bin "*" (.var "tx") (.litUint BN)) (.var "ty")))
                (.index (.var "B") (.bin "+" (.bin "*" (.bin "+" (.var "kk") (.var "tx")) (.var "N")) (.var "col")))
            , -- barrier: ensure all loads complete
              .expr (.call "GroupMemoryBarrierWithGroupSync" [])
            , -- accumulate over BK elements
              .forCount "k" (.litUint 0) (.litUint BK)
                [ .assign (.var "sum")
                    (.bin "+" (.var "sum")
                      (.bin "*"
                        (.index (.var "sA") (.bin "+" (.bin "*" (.var "tx") (.litUint BK)) (.var "k")))
                        (.index (.var "sB") (.bin "+" (.bin "*" (.var "k") (.litUint BN)) (.var "ty"))))) ]
            , -- barrier: ensure all reads done before next tile overwrites sA/sB
              .expr (.call "GroupMemoryBarrierWithGroupSync" [])
            ]
        , -- C[row * N + col] = sum
          .assign
            (.index (.var "C") (.bin "+" (.var "row") (.bin "*" (.var "col") (.var "N"))))
            (.var "sum")
        ]
      }
    ]
  }

def emitGemmShader (M N K : SlangExpr) : String :=
  LeanSlang.emit (gemmShader M N K)

def emitGemmShaderLit (M N K : Nat) : String :=
  emitGemmShader (.litUint M) (.litUint N) (.litUint K)