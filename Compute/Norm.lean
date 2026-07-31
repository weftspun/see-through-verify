import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Norm` — normalization compute shaders (layer norm, group norm)
-/

/-- Layer normalization: y = (x - μ) / σ * γ + β -/
def layerNormShader (BM : Nat) (C : SlangExpr) : SlangShaderModule :=
  { globals := [
      { name := "x", type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
    , { name := "gamma", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
    , { name := "beta", type := .roBuf (.scalar .float), binding := some 2, space := some 0 }
    , { name := "y", type := .rwBuf (.scalar .float), binding := some 3, space := some 0 }
    , { name := "C", type := .scalar .uint, binding := some 4, space := some 0 }
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads BM 1 1]
      , name := "layer_norm"
      , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      , body := [
          .declare (.scalar .uint) "tid" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .float) "mean" (some (.litFloat 0.0))
        , .forCount "i" (.litUint 0) (.var "C")
            [ .assign (.var "mean") (.bin "+" (.var "mean") (.index (.var "x") (.bin "+" (.bin "*" (.var "tid") (.var "C")) (.var "i")))) ]
        , .assign (.var "mean") (.bin "/" (.var "mean") (.var "C"))
        , .declare (.scalar .float) "variance" (some (.litFloat 0.0))
        , .forCount "i" (.litUint 0) (.var "C")
            [ .assign (.var "variance")
                (.bin "+" (.var "variance")
                  (.bin "*"
                    (.bin "-" (.index (.var "x") (.bin "+" (.bin "*" (.var "tid") (.var "C")) (.var "i"))) (.var "mean"))
                    (.bin "-" (.index (.var "x") (.bin "+" (.bin "*" (.var "tid") (.var "C")) (.var "i"))) (.var "mean")))) ]
        , .assign (.var "variance") (.bin "/" (.var "variance") (.var "C"))
        , .forCount "i" (.litUint 0) (.var "C")
            [ .assign
                (.index (.var "y") (.bin "+" (.bin "*" (.var "tid") (.var "C")) (.var "i")))
                (.bin "+"
                  (.bin "*"
                    (.bin "/"
                      (.bin "-" (.index (.var "x") (.bin "+" (.bin "*" (.var "tid") (.var "C")) (.var "i"))) (.var "mean"))
                      (.call "sqrt" [.bin "+" (.var "variance") (.litFloat 1e-5)]))
                    (.index (.var "gamma") (.var "i")))
                  (.index (.var "beta") (.var "i"))) ]
        ]
      }
    ]
  }

/-- SiLU activation: x * sigmoid(x) -/
def siluShader (BM : Nat) : SlangShaderModule :=
  { globals := [
      { name := "x", type := .rwBuf (.scalar .float), binding := some 0, space := some 0 }
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads BM 1 1]
      , name := "silu"
      , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      , body := [
          .declare (.scalar .uint) "tid" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .float) "v" (some (.index (.var "x") (.var "tid")))
        , .assign (.index (.var "x") (.var "tid"))
            (.bin "*" (.var "v") (.bin "/" (.litFloat 1.0) (.bin "+" (.litFloat 1.0) (.call "exp" [.bin "-" (.var "v")]))))
        ]
      }
    ]
  }