import LeanSlang.Types
import LeanSlang.AST
import LeanSlang.Emit

open LeanSlang

/-!
# `Compute.Safetensors` — pure-Lean safetensors reader header parser
-/

inductive SfDtype
  | f16 | bf16 | f32
deriving Repr, BEq, Inhabited

def SfDtype.ofString (s : String) : Option SfDtype :=
  match s with
  | "F16"  => some .f16
  | "BF16" => some .bf16
  | "F32"  => some .f32
  | _      => none

structure SfTensorInfo where
  dtype  : SfDtype
  shape  : List Nat
  offset : Nat
  size   : Nat
deriving Repr, Inhabited

structure SfHeader where
  tensors : List (String × SfTensorInfo)
  dataStart : Nat
deriving Repr, Inhabited

partial def readLEU64 (bytes : ByteArray) (offset : Nat) : Nat :=
  if h : offset + 7 < bytes.size then
    let b0 := bytes.get offset
    let b1 := bytes.get (offset + 1)
    let b2 := bytes.get (offset + 2)
    let b3 := bytes.get (offset + 3)
    let b4 := bytes.get (offset + 4)
    let b5 := bytes.get (offset + 5)
    let b6 := bytes.get (offset + 6)
    let b7 := bytes.get (offset + 7)
    b0.toNat + b1.toNat * 256 + b2.toNat * 65536 + b3.toNat * 16777216 +
    b4.toNat * 4294967296 + b5.toNat * 1099511627776 +
    b6.toNat * 281474976710656 + b7.toNat * 72057594037927936
  else
    0

partial def SfHeader.open (path : System.FilePath) : IO SfHeader := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size < 8 then throw (IO.userError "sf: too small")
  let hdrLen := readLEU64 bytes 0
  let hdrEnd := 8 + hdrLen
  if hdrEnd > bytes.size then throw (IO.userError s!"sf: header {hdrLen} > file {bytes.size}")
  pure { tensors := [], dataStart := hdrEnd }
