import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Conv2d` — f32×f32 conv2d compute shader
-/

/-- Direct conv2d: one thread per output element.
    out[n, oc, oh, ow] = Σ_c Σ_kh Σ_kw in[n, c, oh*stride + kh, ow*stride + kw] * w[c, oc, kh, kw] -/
def conv2dShader (BM BN : Nat) (OC OH OW C : SlangExpr) (KH KW stride pad : Nat) : SlangShaderModule :=
  { globals :=
      [ { name := "input",  type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
      , { name := "weight", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
      , { name := "output", type := .rwBuf (.scalar .float), binding := some 2, space := some 0 }
      , { name := "OC", type := .scalar .uint, binding := some 3, space := some 0 }
      , { name := "OH", type := .scalar .uint, binding := some 4, space := some 0 }
      , { name := "OW", type := .scalar .uint, binding := some 5, space := some 0 }
      , { name := "C",  type := .scalar .uint, binding := some 6, space := some 0 }
      ]
  , functions :=
      [ { attrs := [.shaderCompute, .numthreads BM BN 1]
        , name := "conv2d"
        , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
        , body :=
            [ .declare (.scalar .uint) "oc" (some (.index (.var "dtid") (.litUint 0)))
            , .declare (.scalar .uint) "oh" (some (.index (.var "dtid") (.litUint 1)))
            , .declare (.scalar .uint) "ow" (some (.index (.var "dtid") (.litUint 2)))
            , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
            , .forCount "c" (.litUint 0) (.var "C")
                [ .forCount "kh" (.litUint 0) (.litUint KH)
                    [ .forCount "kw" (.litUint 0) (.litUint KW)
                        [ .assign (.var "sum")
                            (.bin "+" (.var "sum")
                              (.bin "*"
                                (.index (.var "input")
                                  (.bin "+"
                                    (.bin "*" (.var "c") (.bin "*" (.var "OH") (.var "OW")))
                                    (.bin "+"
                                      (.bin "*" (.bin "+" (.bin "*" (.var "oh") (.litUint stride)) (.var "kh")) (.var "OW"))
                                      (.bin "+" (.bin "*" (.var "ow") (.litUint stride)) (.var "kw")))))
                                (.index (.var "weight")
                                  (.bin "+"
                                    (.bin "*" (.var "c") (.bin "*" (.bin "*" (.var "OC") (.litUint KH)) (.litUint KW)))
                                    (.bin "+"
                                      (.bin "*" (.var "oc") (.bin "*" (.litUint KH) (.litUint KW)))
                                      (.bin "+" (.bin "*" (.var "kh") (.litUint KW)) (.var "kw"))))))) ] ] ]
            , .assign (.index (.var "output")
                (.bin "+" (.var "oc") (.bin "*" (.var "oh") (.bin "*" (.var "OW") (.var "OC")))))
                (.var "sum")
            ]
        }
      ]
  }

def emitConv2dShader (BM BN : Nat) (OC OH OW C : SlangExpr) (KH KW stride pad : Nat) : String :=
  LeanSlang.emit (conv2dShader BM BN OC OH OW C KH KW stride pad)