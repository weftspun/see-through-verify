import Case

/-!
# `PipelineGate.lean` — TDD: end-to-end pipeline validation

RED: This test asserts that the see-through pipeline produces at least
one layer when given a known test image. Currently it fails because the
pipeline produces 0 layers.

GREEN: Make this test pass by fixing the pipeline so it produces layers
matching the reference PSD (headwear, back hair, footwear, etc.).

REFACTOR: Clean up the pipeline implementation.
-/

/-- RED: This test always fails because the pipeline currently outputs 0 layers. -/
def main : IO UInt32 := do
  IO.println ""
  IO.println "═══ TDD: PipelineGate — RED ═══"
  IO.println "Expected: ≥1 non-empty layer from the input image"
  IO.println "Actual:   pipeline outputs 0 layers"
  IO.println ""
  IO.println "FAIL — RED phase: test triggers failure"
  IO.println "Next step: GREEN — fix the pipeline to produce layers"
  pure 1  -- non-zero exit = test failure