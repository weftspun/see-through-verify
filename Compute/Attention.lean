import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Attention` — multi-head self-attention compute shader

Q, K, V are f32 buffers with shape [heads, T, d].
Output is the attended result.

For each head, computes:
  attn = softmax(Q × K^T / √d)
  out = attn × V

Uses shared memory for the attention matrix to avoid O(T²) global
writes. Only the output is written back.
-/

/-- Build a single-head attention shader.

Parameters:
  T — number of tokens (spatial positions)
  d — head dimension (e.g. 64)
  BM, BN — threadgroup blocking factors for the QK^T matmul
-/
def attnShader (BM BN : Nat) (T d : SlangExpr) : SlangShaderModule :=
  { globals := [
      { name := "Q", type := .roBuf (.scalar .float), binding := some 0, space := some 0 }
    , { name := "K", type := .roBuf (.scalar .float), binding := some 1, space := some 0 }
    , { name := "V", type := .roBuf (.scalar .float), binding := some 2, space := some 0 }
    , { name := "O", type := .rwBuf (.scalar .float), binding := some 3, space := some 0 }
    , { name := "T", type := .scalar .uint, binding := some 4, space := some 0 }
    , { name := "d", type := .scalar .uint, binding := some 5, space := some 0 }
    ]
  , groupShared := [
      { name := "sQ", elemType := .scalar .float, dims := [BM * 64] }
    , { name := "sK", elemType := .scalar .float, dims := [64 * BN] }
    , { name := "sAttn", elemType := .scalar .float, dims := [BM * BN] }
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads BM BN 1]
      , name := "attention"
      , params := [⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      , body := [
          .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
        , .declare (.scalar .float) "sum" (some (.litFloat 0.0))
        , -- QK^T: sum over d dimension
          .forCount "k" (.litUint 0) (.var "d")
            [ .assign (.var "sum")
                (.bin "+" (.var "sum")
                  (.bin "*"
                    (.index (.var "Q") (.bin "+" (.bin "*" (.var "row") (.var "d")) (.var "k")))
                    (.index (.var "K") (.bin "+" (.bin "*" (.var "col") (.var "d")) (.var "k"))))) ]
        , -- Store raw attention score, scaled by 1/√d
          .assign (.index (.var "sAttn") (.bin "+" (.var "row") (.bin "*" (.var "col") (.litUint BM))))
                   (.bin "/" (.var "sum") (.call "sqrt" [.var "d"]))
        , -- Barrier: ensure all scores are written
          .expr (.call "GroupMemoryBarrierWithGroupSync" [])
        , -- Softmax in-place over rows (each row is a query)
          -- Find max
          .declare (.scalar .float) "rowMax" (some (.index (.var "sAttn") (.var "row")))
        , .forCount "j" (.litUint 1) (.var "T")
            [ .assign (.var "rowMax")
                (.call "max" [.var "rowMax", .index (.var "sAttn") (.bin "+" (.var "row") (.bin "*" (.var "j") (.litUint BM)))]) ]
        , -- Subtract max and exponentiate
          .declare (.scalar .float) "rowSum" (some (.litFloat 0.0))
        , .forCount "j" (.litUint 0) (.var "T")
            [ let idx := .bin "+" (.var "row") (.bin "*" (.var "j") (.litUint BM))
            , .assign (.index (.var "sAttn") idx)
                (.call "exp" [.bin "-" (.index (.var "sAttn") idx) (.var "rowMax")])
            , .assign (.var "rowSum") (.bin "+" (.var "rowSum") (.index (.var "sAttn") idx)) ]
        , -- Normalize
          .forCount "j" (.litUint 0) (.var "T")
            [ .assign (.index (.var "sAttn") (.bin "+" (.var "row") (.bin "*" (.var "j") (.litUint BM))))
                (.bin "/" (.index (.var "sAttn") (.bin "+" (.var "row") (.bin "*" (.var "j") (.litUint BM)))) (.var "rowSum")) ]
        , -- Barrier: ensure softmax is done
          .expr (.call "GroupMemoryBarrierWithGroupSync" [])
        , -- PV: weighted sum over T
          .assign (.var "sum") (.litFloat 0.0)
        , .forCount "j" (.litUint 0) (.var "T")
            [ .assign (.var "sum")
                (.bin "+" (.var "sum")
                  (.bin "*"
                    (.index (.var "sAttn") (.bin "+" (.var "row") (.bin "*" (.var "j") (.litUint BM))))
                    (.call "V" [.bin "+" (.bin "*" (.var "j") (.var "d")) (.var "col")]))) ]
        , -- Write output
          .assign (.index (.var "O") (.bin "+" (.var "row") (.bin "*" (.var "T") (.var "d")) (.bin "+" (.bin "*" (.var "col") (.var "d")) (.var "row"))))
                   (.var "sum")
        ]
      }
    ]
  }