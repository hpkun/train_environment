"""Audit HAPPO reference train/eval consistency without running rollouts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
THREE_V_TWO = "hetero_mav_shared_geo_3v2"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _last_eval_group(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    max_steps = max(_float(row, "total_steps", -1.0) for row in rows)
    return [row for row in rows if _float(row, "total_steps", -2.0) == max_steps]


def _find_checkpoint(records: list[dict], checkpoint: str, config_token: str = THREE_V_TWO) -> dict:
    for record in records:
        if record.get("checkpoint") == checkpoint and config_token in str(record.get("config", "")):
            return record
    return {}


def _meta(path: Path) -> dict:
    return _read_json(path, {})


def _build_audit(exp_dir: Path) -> dict:
    train_rows = _read_csv(exp_dir / "train_log.csv")
    eval_rows = _read_csv(exp_dir / "eval_log.csv")
    checkpoint_records = _read_json(
        exp_dir / "checkpoint_eval" / "happo_3v2_checkpoint_eval.json", []
    )
    latest_meta = _meta(exp_dir / "latest" / "meta.json")
    best_meta = _meta(exp_dir / "best" / "meta.json")

    latest_train = train_rows[-1] if train_rows else {}
    last_eval = _last_eval_group(eval_rows)
    best_3v2 = _find_checkpoint(checkpoint_records, "best")
    latest_3v2 = _find_checkpoint(checkpoint_records, "latest")

    likely_causes: list[str] = []
    blocking_issues: list[str] = []
    warnings: list[str] = []

    train_red_win = _float(latest_train, "red_win")
    train_timeout = _float(latest_train, "timeout")
    train_mav_survival = _float(latest_train, "mav_survival")
    latest_eval_blue_win = _float(latest_3v2, "blue_win_rate")
    latest_eval_mav_survival = _float(latest_3v2, "mav_survival_rate")

    inconsistent = (
        train_red_win >= 0.9
        and train_timeout >= 0.9
        and train_mav_survival >= 0.9
        and latest_eval_blue_win >= 0.9
        and latest_eval_mav_survival <= 0.1
    )
    if inconsistent:
        likely_causes.extend([
            "train_log_latest_is_recent_on_policy_stochastic_window_not_checkpoint_deterministic_eval",
            "deterministic_mean_action_eval_can_collapse_even_when_sampled_training_rollouts_timeout",
            "latest_checkpoint_eval_uses_saved_policy_after_training_not_the_exact_episode_window_row",
        ])
        warnings.append("latest train row and deterministic latest checkpoint eval disagree strongly")

    meta_mismatch = []
    for key in ("reward_mode", "opponent_policy", "obs_adapter_version"):
        values = {str(latest_meta.get(key, "")), str(best_meta.get(key, ""))}
        if len(values - {""}) > 1:
            meta_mismatch.append(key)
    if meta_mismatch:
        blocking_issues.append(f"checkpoint metadata mismatch: {meta_mismatch}")
    else:
        likely_causes.append("no obvious checkpoint metadata mismatch found")

    consistency_status = "inconsistent" if inconsistent else "not_proven_inconsistent"
    if latest_eval_blue_win >= 0.9 and latest_eval_mav_survival <= 0.1:
        blocking_issues.append("latest deterministic checkpoint is not a usable baseline")

    return {
        "experiment_dir": str(exp_dir),
        "consistency_status": consistency_status,
        "latest_train_row": latest_train,
        "last_online_eval_rows": last_eval,
        "best_3v2_checkpoint_eval": best_3v2,
        "latest_3v2_checkpoint_eval": latest_3v2,
        "metadata": {"best": best_meta, "latest": latest_meta},
        "likely_causes": likely_causes,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "recommended_next_steps": [
            "compare deterministic and stochastic checkpoint evaluation",
            "audit MAV death/failure modes",
            "decide 1M readiness only after those diagnostics",
        ],
    }


def _write_md(path: Path, audit: dict) -> None:
    lines = [
        "# HAPPO Train/Eval Consistency Audit",
        "",
        f"- experiment_dir: {audit['experiment_dir']}",
        f"- consistency_status: {audit['consistency_status']}",
        "",
        "## Latest Train Row",
    ]
    for key in ("total_steps", "avg_return", "red_win", "blue_win", "timeout",
                "mav_survival", "red_alive_final", "blue_alive_final",
                "red_missiles_fired", "missile_hits"):
        lines.append(f"- {key}: {audit['latest_train_row'].get(key)}")
    lines.extend(["", "## Latest Deterministic 3v2 Eval"])
    latest = audit.get("latest_3v2_checkpoint_eval", {})
    for key in ("avg_return", "red_win_rate", "blue_win_rate", "timeout_rate",
                "mav_survival_rate", "blue_dead_mean", "red_missile_hits_mean"):
        lines.append(f"- {key}: {latest.get(key)}")
    lines.extend(["", "## Likely Causes"])
    lines.extend(f"- {item}" for item in audit["likely_causes"])
    lines.extend(["", "## Blocking Issues"])
    lines.extend(f"- {item}" for item in audit["blocking_issues"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HAPPO train/eval consistency")
    parser.add_argument("--experiment-dir", "--output-dir", dest="experiment_dir", default=DEFAULT_DIR)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    exp_dir = _rel(args.experiment_dir)
    if not exp_dir.exists():
        print(f"experiment directory does not exist: {exp_dir}", file=sys.stderr)
        return 2
    audit = _build_audit(exp_dir)
    out_dir = exp_dir / "consistency_audit"
    out_json = _rel(args.output_json) if args.output_json else out_dir / "happo_train_eval_consistency.json"
    out_md = _rel(args.output_md) if args.output_md else out_dir / "happo_train_eval_consistency.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_md(out_md, audit)
    print(f"consistency_status: {audit['consistency_status']}")
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
