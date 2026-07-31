import Lake
open Lake DSL

package verify where
  -- Witness-search quality gates + lean-slang GPU compute shader codegen

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "v0.0.6"

require «lean-duckdb» from git
  "https://github.com/v-sekai-multiplayer-fabric/lean-duckdb.git" @ "main"

-- import library for the seethrough_c DLL; override with
--   lake build -Kseethrough_c_lib=<path>
def seethroughCLib :=
  #[((get_config? seethrough_c_lib).getD
      (__dir__ / ".." / "build-vulkan" / "seethrough_c.lib").toString)]

lean_lib Case
lean_lib Compute

@[default_target] lean_exe kernel_gate where
  root := `KernelGate
  moreLinkArgs := seethroughCLib

lean_exe quant_design where
  root := `QuantDesign
  moreLinkArgs := seethroughCLib

lean_exe compute_verify where
  root := `ComputeVerify
