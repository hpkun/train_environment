"""Decide whether HAPPO reference v0 is ready for a 1M run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _find(records: list[dict], checkpoint: str, token: str = "hetero_mav_shared_geo_3v2") -> dict:
    for record in records:
        if record.get("checkpoint") == checkpoint and token in str(record.get("config", "")):
            return record
    return {}


def _mode_record(payload: dict, checkpoint: str, mode: str, token: str = "hetero_mav_shared_geo_3v2") -> dict:
    for record in payload.get("records", []):
        if record.get("checkpoint") == checkpoint and record.get("mode") == mode and token in str(record.get("config", "")):
            return record
    return {}


def _failure_summary(payload: dict, checkpoint: str) -> dict:
    for record in payload.get("records", []):
        summary = record.get("summary", {})
        if summary.get("checkpoint") == checkpoint:
            return summary
    return {}


def _build_decision(args) -> dict:
    exp_dir = _rel(args.experiment_dir)
    consistency = _read_json(_rel(args.consistency_json), {})
    policy_modes = _read_json(_rel(args.policy_modes_json), {})
    mav_failures = _read_json(_rel(args.mav_failure_json), {})
    checkpoints = _read_json(_rel(args.checkpoint_eval_json), [])

    best = _find(checkpoints, "best")
    latest = _find(checkpoints, "latest")
    best_det = _mode_record(policy_modes, "best", "deterministic")
    best_stoch = _mode_record(policy_modes, "best", "stochastic")
    latest_det = _mode_record(policy_modes, "latest", "deterministic")
    best_mav_failure = _failure_summary(mav_failures, "best")

    blocking: list[str] = []
    warnings: list[str] = []
    required_fixes: list[str] = []

    useful_checkpoint_signal = (
        float(best.get("blue_dead_mean", 0.0)) > 0.0
        or float(best.get("red_missile_hits_mean", 0.0)) > 0.0
        or float(best_stoch.get("blue_dead_mean", 0.0)) > 0.0
        or float(best_stoch.get("red_missile_hits_mean", 0.0)) > 0.0
    )
    if not useful_checkpoint_signal:
        blocking.append("no checkpoint shows red missile hits or blue deaths")

    if float(latest.get("blue_win_rate", 0.0)) >= 0.9 and float(latest.get("mav_survival_rate", 1.0)) <= 0.1:
        warnings.append("latest deterministic checkpoint collapses to blue wins and MAV death")

    if float(best.get("mav_survival_rate", 1.0)) <= 0.1:
        warnings.append("best deterministic checkpoint still has near-zero MAV survival")

    if consistency.get("consistency_status") == "inconsistent":
        warnings.append("training-window and deterministic checkpoint evaluation are inconsistent")

    if any("metadata mismatch" in item for item in consistency.get("blocking_issues", [])):
        blocking.append("reward/config/opponent metadata mismatch detected")

    if best_det and best_stoch:
        det_blue = float(best_det.get("blue_win_rate", 0.0))
        stoch_blue = float(best_stoch.get("blue_win_rate", 0.0))
        if abs(det_blue - stoch_blue) >= 0.3:
            warnings.append("deterministic and stochastic policy modes diverge materially")
    elif policy_modes:
        warnings.append("policy mode audit missing best deterministic or stochastic 3v2 record")

    if float(best_mav_failure.get("mav_death_rate", 0.0)) >= 0.9:
        blocking.append("MAV failure is systematic in audited best checkpoint")

    if blocking:
        required_fixes = [
            "fix MAV survival/failure mode before extending training",
            "stabilize deterministic checkpoint evaluation against stochastic rollout mismatch",
            "re-run a short 100k-200k pilot after fixes before any 1M run",
        ]
    elif warnings:
        required_fixes = [
            "treat 1M as conditional only after validating stochastic and deterministic eval",
            "monitor MAV survival and blue death metrics during online eval",
        ]

    run_1m_recommended = bool(useful_checkpoint_signal and not blocking and not warnings)
    next_command = None
    if run_1m_recommended:
        next_command = (
            "python scripts/run_happo_3v2_reference_1m_fast.py "
            "--total-env-steps 1000000 --rollout-length 256 --device cpu"
        )

    return {
        "experiment_dir": str(exp_dir),
        "run_1m_recommended": run_1m_recommended,
        "blocking_issues": blocking,
        "warnings": warnings,
        "required_fixes_before_1m": required_fixes[:3],
        "evidence": {
            "best_checkpoint_3v2": best,
            "latest_checkpoint_3v2": latest,
            "best_deterministic_3v2": best_det,
            "best_stochastic_3v2": best_stoch,
            "latest_deterministic_3v2": latest_det,
            "best_mav_failure_summary": best_mav_failure,
            "consistency_status": consistency.get("consistency_status"),
        },
        "next_command": next_command,
    }


def _write_md(path: Path, decision: dict) -> None:
    lines = [
        "# HAPPO 1M Readiness Decision",
        "",
        f"- run_1m_recommended: {decision['run_1m_recommended']}",
        "",
        "## Blocking Issues",
    ]
    lines.extend(f"- {item}" for item in decision["blocking_issues"])
    if not decision["blocking_issues"]:
        lines.append("- none")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {item}" for item in decision["warnings"])
    if not decision["warnings"]:
        lines.append("- none")
    lines.extend(["", "## Required Fixes Before 1M"])
    lines.extend(f"- {item}" for item in decision["required_fixes_before_1m"])
    if not decision["required_fixes_before_1m"]:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide HAPPO 1M readiness from audit JSON files")
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--consistency-json", default=f"{DEFAULT_DIR}/consistency_audit/happo_train_eval_consistency.json")
    parser.add_argument("--policy-modes-json", "--policy-mode-json", dest="policy_modes_json", default=f"{DEFAULT_DIR}/policy_mode_eval/happo_policy_mode_eval.json")
    parser.add_argument("--mav-failure-json", default=f"{DEFAULT_DIR}/mav_failure_audit/happo_mav_failure_modes.json")
    parser.add_argument("--checkpoint-eval-json", "--checkpoint-json", dest="checkpoint_eval_json", default=f"{DEFAULT_DIR}/checkpoint_eval/happo_3v2_checkpoint_eval.json")
    parser.add_argument("--summary-json", default=f"{DEFAULT_DIR}/happo_3v2_training_summary.json")
    parser.add_argument("--output-json", default=f"{DEFAULT_DIR}/readiness/happo_1m_readiness_decision.json")
    parser.add_argument("--output-md", default=f"{DEFAULT_DIR}/readiness/happo_1m_readiness_decision.md")
    args = parser.parse_args()

    decision = _build_decision(args)
    out_json = _rel(args.output_json)
    out_md = _rel(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _write_md(out_md, decision)
    print(f"run_1m_recommended: {decision['run_1m_recommended']}")
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
