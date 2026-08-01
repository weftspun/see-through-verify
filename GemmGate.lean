import Case

/-!
# `GemmGate.lean` — TDD: GEMM with real model weights

RED: The Slang GEMM shader has not been validated against real model
weights. This test loads a weight tensor from the safetensors file and
verifies the GEMM output matches a CPU reference.

GREEN: Wire the runtime to dispatch the Slang shader and compare
against CPU.
-/

/-- RED: GEMM validation fails because the Slang shader dispatch
is not yet wired into the runtime. -/
def main : IO UInt32 := do
  IO.println ""
  IO.println "═══ TDD: GemmGate — RED ═══"
  IO.println "Test: Slang GEMM with real layerdiff3d text_encoder weights"
  IO.println "Status: dispatch not wired — test fails"
  IO.println ""
  IO.println "Next: implement GPU dispatch and compare against CPU"
  pure 1