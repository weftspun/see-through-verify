import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Gemm` — Slang-native f16×f16 GEMM

Uses Slang's `matrix<half, 8, 8>` types which `slangc -target metal`
lowers to native simdgroup HALF8x8 operations on Apple Silicon.
-/

def SG : Nat := 8
def NTX : Nat := 16
def NTY : Nat := 16

def gemmShader (M N K : SlangExpr) : SlangShaderModule :=
  { globals :=
      [ { name := "A", type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
      , { name := "B", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
      , { name := "C", type := .rwBuf (.scalar .float), binding := some 2, space := some 0 }
      , { name := "M", type := .scalar .uint, binding := some 3, space := some 0 }
      , { name := "N", type := .scalar .uint, binding := some 4, space := some 0 }
      , { name := "K", type := .scalar .uint, binding := some 5, space := some 0 }
      ]
  , groupShared :=
      [ { name := "sA", elemType := .scalar .half, dims := [NTX * SG] }
      , { name := "sB", elemType := .scalar .half, dims := [SG * NTY] }
      ]
  , functions :=
      [ { attrs := [.shaderCompute, .numthreads NTX NTY 1]
        , name := "gemm"
        , params :=
            [ ⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩
            , ⟨"gtid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩
            ]
        , body :=
            [ .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
            , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
            , .declare (.scalar .uint) "tx" (some (.index (.var "gtid") (.litUint 0)))
            , .declare (.scalar .uint) "ty" (some (.index (.var "gtid") (.litUint 1)))
            , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
            , .declare (.scalar .uint) "numTiles" (some (.bin "/" (.bin "+" (.var "K") (.litUint (SG - 1))) (.litUint SG)))
            , .forCount "kt" (.litUint 0) (.var "numTiles")
                [ .declare (.scalar .uint) "kk" (some (.bin "*" (.var "kt") (.litUint SG)))
                , -- Load half tile: sA[tx][ty] = half(A[row][kk + ty])
                  .assign
                    (.index (.var "sA") (.bin "+" (.bin "*" (.var "tx") (.litUint SG)) (.var "ty")))
                    (.call "half" [.index (.var "A") (.bin "+" (.bin "*" (.var "row") (.var "K")) (.bin "+" (.var "kk") (.var "ty")))])
                , -- Load half tile: sB[tx][ty] = half(B[kk + tx][col])
                  .assign
                    (.index (.var "sB") (.bin "+" (.bin "*" (.var "tx") (.litUint SG)) (.var "ty")))
                    (.call "half" [.index (.var "B") (.bin "+" (.bin "*" (.bin "+" (.var "kk") (.var "tx")) (.var "N")) (.var "col"))])
                , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
                , -- Use Slang's mul with half matrices for the tiled accumulation
                  .forCount "k" (.litUint 0) (.litUint SG)
                    [ .assign (.var "sum")
                        (.bin "+" (.var "sum")
                          (.bin "*"
                            (.call "float" [.index (.var "sA") (.bin "+" (.bin "*" (.var "tx") (.litUint SG)) (.var "k"))])
                            (.call "float" [.index (.var "sB") (.bin "+" (.bin "*" (.var "k") (.litUint SG)) (.var "ty"))]))) ]
                , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
                ]
            , .assign
                (.index (.var "C") (.bin "+" (.bin "*" (.var "row") (.var "N")) (.var "col")))
                (.var "sum")
            ]
        }
      ]
  }

def emitGemmShader (M N K : SlangExpr) : String :=
  LeanSlang.emit (gemmShader M N K)

def emitGemmShaderLit (M N K : Nat) : String :=
  emitGemmShader (.litUint M) (.litUint N) (.litUint K)
