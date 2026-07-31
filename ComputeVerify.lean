import LeanSlang.Emit
import LeanSlang.AST
import LeanSlang.Types

open LeanSlang

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
      , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      , body := [
          .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
        , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
        , .forCount "k" (.litUint 0) (.var "K")
            [ .assign (.var "sum")
                (.bin "+" (.var "sum")
                  (.bin "*"
                    (.index (.var "A") (.bin "+" (.bin "*" (.var "row") (.var "K")) (.var "k")))
                    (.index (.var "B") (.bin "+" (.bin "*" (.var "k") (.var "N")) (.var "col"))))) ]
        , .assign
            (.index (.var "C") (.bin "+" (.var "row") (.bin "*" (.var "col") (.var "N"))))
            (.var "sum")
        ]
      }
    ]
  }

def emitGemmShader (BM BN : Nat) (M N K : SlangExpr) : String :=
  LeanSlang.emit (gemmShader BM BN M N K)

def emitGemmShaderLit (BM BN M N K : Nat) : String :=
  emitGemmShader BM BN (.litUint M) (.litUint N) (.litUint K)

/-- Emit the GEMM shader and check it's non-empty. -/
def main : IO UInt32 := do
  let emitted := emitGemmShaderLit 64 64 128 128 64
  if emitted == "" then
    IO.println "FAIL — empty"
    pure 1
  else
    IO.println "PASS"
    IO.println emitted
    pure 0