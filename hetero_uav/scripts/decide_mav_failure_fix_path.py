"""Decide next MAV failure fix path from diagnostic outputs."""
from __future__ import annotations

import argparse

from happo_mav_audit_common import (
    DEFAULT_EXPERIMENT_DIR,
    read_json,
    rel,
    write_json,
    write_md,
)


def build_decision(args) -> dict:
    control = read_json(args.control_json, {})
    ablation = read_json(args.ablation_json, {})
    blue = read_json(args.blue_target_json, {})
    failure = read_json(args.mav_failure_json, {})
    readiness = read_json(args.readiness_json, {})

    control_summary = control.get("summary", {})
    ablation_summary = ablation.get("summary", {})
    best_survival = float(ablation_summary.get("best_realistic_mav_survival_rate", 0.0) or 0.0)
    safe_survival = bool(ablation_summary.get("safe_mav_can_survive", False))
    f16_better = bool(ablation_summary.get("f16_surrogate_improves", False) or control_summary.get("f16_surrogate_more_stable", False))
    f22_stable = bool(control_summary.get("f22_stable", False))
    mav_target_fraction = float(blue.get("mav_missile_target_fraction", blue.get("mav_target_fraction", 0.0)) or 0.0)
    unavailable = set(blue.get("unavailable_fields", []))
    existing_1m = bool(readiness.get("run_1m_recommended", False))

    blocking: list[str] = []
    if best_survival <= 0.5:
        blocking.append("no realistic ablation case reaches MAV survival > 0.5")
    if not safe_survival:
        blocking.append("safe fixed MAV action does not yet prove survival")
    if not existing_1m:
        blocking.append("previous HAPPO 1M readiness decision is false")

    if not f22_stable and f16_better:
        hypothesis = "f22_control_or_dynamics_instability"
        action = "C. use F-16 MAV surrogate for algorithm validation, then return to F-22"
    elif not f22_stable:
        hypothesis = "f22_control_or_dynamics_instability"
        action = "A. fix environment or control stability"
    elif mav_target_fraction > 0.5:
        hypothesis = "blue_target_preference"
        action = "B. fix happo_ref_v0 MAV survival reward or action regularization"
    elif "blue_lock_target_counts" in unavailable and "blue_selected_target_counts" in unavailable:
        hypothesis = "instrumentation_insufficient"
        action = "A. fix environment or control stability"
    elif best_survival <= 0.5:
        hypothesis = "reward_insufficient_for_mav_survival"
        action = "B. fix happo_ref_v0 MAV survival reward or action regularization"
    else:
        hypothesis = "mixed"
        action = "B. fix happo_ref_v0 MAV survival reward or action regularization"

    run_1m_allowed = best_survival > 0.5 and not blocking
    command = None
    if run_1m_allowed:
        command = "python scripts/run_happo_3v2_reference_1m_fast.py"

    return {
        "primary_failure_hypothesis": hypothesis,
        "recommended_next_action": action,
        "run_1m_allowed": bool(run_1m_allowed),
        "blocking_issues": blocking[:3],
        "if_allowed_command": command,
        "evidence": {
            "f22_stable": f22_stable,
            "f16_surrogate_more_stable": f16_better,
            "best_realistic_mav_survival_rate": best_survival,
            "safe_mav_can_survive": safe_survival,
            "mav_missile_target_fraction": mav_target_fraction,
            "blue_unavailable_fields": sorted(unavailable),
            "previous_1m_readiness": existing_1m,
            "mav_failure_audit_present": bool(failure),
        },
    }


def write_report(data: dict, output_md: str) -> None:
    lines = [
        "# MAV Failure Fix Path Decision",
        "",
        f"- primary_failure_hypothesis: {data['primary_failure_hypothesis']}",
        f"- recommended_next_action: {data['recommended_next_action']}",
        f"- run_1m_allowed: {data['run_1m_allowed']}",
        "",
        "## Blocking Issues",
    ]
    if data["blocking_issues"]:
        lines.extend(f"- {item}" for item in data["blocking_issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Evidence", f"- {data['evidence']}"])
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide MAV failure fix path")
    default = DEFAULT_EXPERIMENT_DIR
    parser.add_argument("--control-json", default="outputs/environment_audit/mav_control_stability_sweep.json")
    parser.add_argument("--ablation-json", default=f"{default}/mav_survival_ablation/happo_mav_survival_ablation.json")
    parser.add_argument("--blue-target-json", default=f"{default}/blue_target_audit/blue_target_preference_against_mav.json")
    parser.add_argument("--mav-failure-json", default=f"{default}/mav_failure_audit/happo_mav_failure_modes.json")
    parser.add_argument("--readiness-json", default=f"{default}/readiness/happo_1m_readiness_decision.json")
    parser.add_argument("--output-json", default=f"{default}/mav_failure_decision/mav_failure_fix_path.json")
    parser.add_argument("--output-md", default=f"{default}/mav_failure_decision/mav_failure_fix_path.md")
    args = parser.parse_args()
    data = build_decision(args)
    out_json = write_json(args.output_json, data)
    write_report(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"primary_failure_hypothesis: {data['primary_failure_hypothesis']}")
    print(f"run_1m_allowed: {data['run_1m_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
