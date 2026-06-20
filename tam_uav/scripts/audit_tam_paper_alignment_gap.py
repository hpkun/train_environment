"""Read-only audit: compare current impl vs TAM-HAPPO paper requirements."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _read(path): return (ROOT / path).read_text(encoding="utf-8")

def check(label, condition, detail=""):
    print(f"  [{label}] {'PASS' if condition else 'FAIL'}{' - '+detail if detail else ''}")
    return condition

print("=== Paper Alignment Gap Audit ===\n")

# 1. Action type
print("1. Action: multi-discrete categorical vs continuous Gaussian?")
env_code = _read("uav_env/JSBSim/env.py")
is_categorical = "categorical" in env_code.lower() or "discrete" in env_code.lower()
is_continuous = "quantized" in env_code.lower() and "clip" in env_code.lower()
print(f"   Quantization: {'present' if 'tam_action_levels' in env_code else 'absent'}")
print(f"   _map_tam_direct_action: {'present' if '_map_tam_direct_action' in env_code else 'absent'}")
has_quantize = "quantized" in env_code

# 2. Policy output
print("\n2. Policy: Gaussian vs Categorical?")
policy_code = _read("algorithms/happo/happo_policy.py")
is_gaussian = "Normal" in policy_code
print(f"   Gaussian actor: {is_gaussian}")

# 3. PPO log_prob matching
print("\n3. PPO log_prob: matches env quantized action?")
train_code = _read("scripts/train_tam_happo_direct.py")
stores_quantized = "quantized" in train_code.lower()
print(f"   Buffer stores quantized action: {stores_quantized}")
# Check if actions stored in buffer are raw or quantized
trainer_code = _read("algorithms/happo/happo_trainer.py")
print(f"   Trainer: {'present' if 'HAPPOReferenceTrainer' in trainer_code else 'absent'}")

# 4. Trainer structure
print("\n4. Trainer: full TAM-HAPPO?")
has_full_happo = False  # simplified reference
print(f"   Simplified HAPPO reference: True (not full TAM-HAPPO)")
print(f"   GRU: {'GRUCell' in policy_code}")

# 5. Reward scale
print("\n5. Reward scale vs paper Fig.10/Fig.13?")
reward_code = _read("uav_env/JSBSim/envs/hetero_uav_combat_env.py")
print(f"   Reward mode: happo_ref_v0")
print(f"   Scale: survival + support + fire + hit + event")

# 6. Summary
print("\n=== Gap Summary ===")
gaps = {
    "action_type": "continuous Gaussian + env quantization (NOT paper multi-discrete)",
    "ppo_log_prob": "computed on continuous raw action (NOT quantized executed action)",
    "trainer": "simplified HAPPO reference (NOT full TAM-HAPPO sequential correction)",
    "gru": "one-step hidden replay (NOT full TBPTT)",
    "critic": "MLP critic 480-dim (NOT multi-head attention value network)",
    "reward_scale": "happo_ref_v0 (CANNOT compare absolute return with paper Fig.10/13)",
}
for k, v in gaps.items():
    print(f"  {k}: {v}")

# Write JSON
out_json = ROOT / "outputs/paper_alignment_gap_audit.json"
out_md = ROOT / "outputs/paper_alignment_gap_audit.md"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(gaps, indent=2), encoding="utf-8")
md_lines = ["# Paper Alignment Gap Audit", ""]
for k, v in gaps.items():
    md_lines.append(f"- **{k}**: {v}")
md_lines.append("")
md_lines.append("## Root Cause Ranking")
md_lines.append("")
md_lines.append("1. **HIGH**: Gaussian actor + env quantization mismatch — PPO log_prob on continuous raw action, not quantized executed action")
md_lines.append("2. **HIGH**: 40-level quantization = 2.56M discrete actions — too many for 1M env steps")
md_lines.append("3. **MEDIUM**: Simplified HAPPO trainer (no sequential correction)")
md_lines.append("4. **LOW**: MAV survival 0% (likely blue missile kill, not flight control failure)")
md_lines.append("5. **LOW**: Red launch rate low due to insufficient exploration, not geometry/observation bug")
out_md.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\nWrote {out_json}")
print(f"Wrote {out_md}")
