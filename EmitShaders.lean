import Compute.Gemm

def main : IO UInt32 := do
  let gemm_src := emitGemmShaderLit 64 64 128 128 64
  IO.FS.writeFile "shaders/gemm.slang" gemm_src
  IO.println "wrote shaders/gemm.slang"
  pure 0