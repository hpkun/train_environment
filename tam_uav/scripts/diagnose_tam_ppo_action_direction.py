"""Diagnose categorical PPO action-gradient direction.

Runs synthetic sign tests and reports results as md/json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "tam_ppo_action_direction"


def diagnostic_1_synthetic(device: torch.device) -> dict:
    """Synthetic PPO sign test: adv<0 decreases prob, adv>0 increases prob."""
    # Re-use the well-tested helper from the pytest module
    from tests.test_tam_categorical_ppo_sign import (
        _build_policy, _build_mini_buffer,
        _run_update, _get_mav_selected_probs,
    )

    test_cases = [
        ([38, 20, 22, 21], -1.0, "should_decrease"),
        ([38, 20, 22, 21], +1.0, "should_increase"),
        ([39, 10, 10, 10], -1.0, "should_decrease"),
        ([39, 10, 10, 10], +1.0, "should_increase"),
        ([20, 20, 20, 20], -1.0, "should_decrease"),
        ([20, 20, 20, 20], +1.0, "should_increase"),
    ]

    results = []
    for actions, adv_val, expected in test_cases:
        policy = _build_policy().to(device)
        buf = _build_mini_buffer(policy, device, actions)
        joint_before, per_axis_before = _get_mav_selected_probs(policy, buf, device)
        _run_update(policy, buf, device, advantage_val=adv_val)
        joint_after, per_axis_after = _get_mav_selected_probs(policy, buf, device)
        delta = joint_after - joint_before
        direction_ok = (adv_val < 0 and delta < 0) or (adv_val > 0 and delta > 0)

        results.append({
            "actions": list(actions),
            "advantage": adv_val,
            "expected": expected,
            "joint_before": joint_before,
            "joint_after": joint_after,
            "joint_delta": delta,
            "per_axis_before": per_axis_before.tolist(),
            "per_axis_after": per_axis_after.tolist(),
            "direction_correct": bool(direction_ok),
        })

    return {
        "test": "synthetic_ppo_sign",
        "all_correct": all(r["direction_correct"] for r in results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Diagnostic 1: Synthetic PPO sign test ===", flush=True)
    synth = diagnostic_1_synthetic(device)

    report = {"diagnostic_1_synthetic": synth}
    (out_dir / "ppo_action_direction.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# PPO Action-Gradient Direction Diagnostic", "",
             "## Synthetic Categorical PPO Sign Test", ""]
    if synth["all_correct"]:
        lines.append("**RESULT: ALL SIGN TESTS PASSED** [PASS]")
    else:
        lines.append("**RESULT: SOME SIGN TESTS FAILED** [FAIL]")
    lines.append("")

    for case in synth["cases"]:
        status = "[PASS]" if case["direction_correct"] else "[FAIL]"
        lines.append(
            f"- actions={case['actions']} adv={case['advantage']:+.1f} "
            f"joint: {case['joint_before']:.6f} → {case['joint_after']:.6f} "
            f"(Δ={case['joint_delta']:+.6f}) {status}")
        for i in range(len(case["per_axis_before"])):
            pb_val = case["per_axis_before"][i]
            pa_val = case["per_axis_after"][i]
            if isinstance(pb_val, list):
                pb_val = pb_val[0] if pb_val else 0.0
            if isinstance(pa_val, list):
                pa_val = pa_val[0] if pa_val else 0.0
            lines.append("  axis %d: %.6f -> %.6f (d=%+.6f)" % (i, float(pb_val), float(pa_val), float(pa_val)-float(pb_val)))

    (out_dir / "ppo_action_direction.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nReports written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
