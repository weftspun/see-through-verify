import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Gemm` — simdgroup f16×f16 GEMM compute shader

Uses Metal simdgroup HALF8x8 matrix multiply for native hardware
throughput. Each simdgroup (32 threads) collaboratively loads an 8×8
tile of A and B into `simdgroup_half8x8` registers, then issues
a single `simdgroup_multiply_accumulate` instruction that executes
8×8×8 = 512 multiply-adds in one cycle.

Threadgroup: 2×2 simdgroups = 64 threads, each handling a 8×8 tile
→ output tile of 16×16 per threadgroup.
-/

def SIMDGROUP_SIZE : Nat := 8  -- simdgroup_half8x8
def SIMDGROUPS_X : Nat := 2
def SIMDGROUPS_Y : Nat := 2
def BM : Nat := SIMDGROUP_SIZE * SIMDGROUPS_X  -- 16
def BN : Nat := SIMDGROUP_SIZE * SIMDGROUPS_Y  -- 16
def BK : Nat := SIMDGROUP_SIZE                 -- 8 (one tile column)

-- Parameters for the threadgroup
def NTHREADS_X : Nat := SIMDGROUP_SIZE * SIMDGROUPS_X  -- 16
def NTHREADS_Y : Nat := SIMDGROUP_SIZE * SIMDGROUPS_Y  -- 16

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
      { name := "sA", elemType := .scalar .half, dims := [BM * BK] }   -- 16*8 = 128 halfs
    , { name := "sB", elemType := .scalar .half, dims := [BK * BN] }   -- 8*16 = 128 halfs
    ]
  , functions := [
      { attrs := [.shaderCompute, .numthreads NTHREADS_X NTHREADS_Y 1]
      , name := "gemm"
      , params := [
          ⟨"dtid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩
        , ⟨"gtid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩
        , ⟨"sgtg", .scalar .uint, .svGroupId, none, none, .qIn⟩
        ]
      , body := [
          .declare (.scalar .uint) "row" (some (.index (.var "dtid") (.litUint 0)))
        , .declare (.scalar .uint) "col" (some (.index (.var "dtid") (.litUint 1)))
        , .declare (.scalar .uint) "tx" (some (.index (.var "gtid") (.litUint 0)))
        , .declare (.scalar .uint) "ty" (some (.index (.var "gtid") (.litUint 1)))
        , .declare (.scalar .uint) "laneId" (some (.bin "+" (.bin "*" (.bin "/" (.var "tx") (.litUint SIMDGROUP_SIZE)) (.litUint SIMDGROUP_SIZE)) (.bin "/" (.var "ty") (.litUint SIMDGROUP_SIZE))))
        , .declare (.scalar .uint) "laneInSG" (some (.bin "+" (.bin "*" (.bin "%" (.var "tx") (.litUint SIMDGROUP_SIZE)) (.litUint SIMDGROUP_SIZE)) (.bin "%" (.var "ty") (.litUint SIMDGROUP_SIZE))))
        , .declare (.simdgroup .half SIMDGROUP_SIZE SIMDGROUP_SIZE) "mAcc" none
        , .declare (.simdgroup .half SIMDGROUP_SIZE SIMDGROUP_SIZE) "mA" none
        , .declare (.simdgroup .half SIMDGROUP_SIZE SIMDGROUP_SIZE) "mB" none
        , .expr (.call "simdgroup_load" [.var "mAcc", .litFloat 0.0, .litUint 0])
        , .declare (.scalar .uint) "numTiles" (some (.bin "/" (.bin "+" (.var "K") (.litUint (BK - 1))) (.litUint BK)))
        , .forCount "kkTile" (.litUint 0) (.var "numTiles")
            [ .declare (.scalar .uint) "kk" (some (.bin "*" (.var "kkTile") (.litUint BK)))
            , -- Load A tile into shared: sA[tx * BK + ty] = half(A[row * K + kk + ty])
              .assign
                (.index (.var "sA") (.bin "+" (.bin "*" (.var "tx") (.litUint BK)) (.var "ty")))
                (.call "half" [.index (.var "A") (.bin "+" (.bin "*" (.var "row") (.var "K")) (.bin "+" (.var "kk") (.var "ty")))])
            , -- Load B tile into shared: sB[tx * BN + ty] = half(B[(kk + tx) * N + col])
              .assign
                (.index (.var "sB") (.bin "+" (.bin "*" (.var "tx") (.litUint BN)) (.var "ty")))
                (.call "half" [.index (.var "B") (.bin "+" (.bin "*" (.bin "+" (.var "kk") (.var "tx")) (.var "N")) (.var "col"))])
            , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
            , -- Load from shared into simdgroup matrix
              .simdgroupLoad (.var "mA") (.index (.var "sA") (.var "laneInSG")) SIMDGROUP_SIZE
            , .simdgroupLoad (.var "mB") (.index (.var "sB") (.var "laneInSG")) SIMDGROUP_SIZE
            , -- Multiply-accumulate
              .expr (.simdgroupMulAdd (.var "mAcc") (.var "mA") (.var "mB"))
            , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
            ]
        , -- Store result: C[row * N + col] = mAcc
          .simdgroupStore (.index (.var "C") (.bin "+" (.var "row") (.bin "*" (.var "col") (.var "N")))) (.var "mAcc") 1
        ]
      }
    ]
  }

def emitGemmShader (M N K : SlangExpr) : String :=
  LeanSlang.emit (gemmShader M N K)

def emitGemmShaderLit (M N K : Nat) : String :=
  emitGemmShader (.litUint M) (.litUint N) (.litUint K)