"""Analyze a short entity-attention HAPPO sanity run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _range(rows: list[dict], key: str) -> dict:
    values = [_float(row, key, math.nan) for row in rows]
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {"min": None, "max": None, "final": None}
    return {"min": min(values), "max": max(values), "final": values[-1]}


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def analyze(output_dir: Path) -> dict:
    train_rows = _read_csv(output_dir / "train_log.csv")
    eval_rows = _read_csv(output_dir / "eval_log.csv")
    latest_meta = _load_json(output_dir / "latest" / "meta.json")
    best_meta = _load_json(output_dir / "best" / "meta.json")
    final_row = train_rows[-1] if train_rows else {}
    nan_detected = bool(latest_meta.get("nan_detected", False)) or any(
        int(_float(row, "nan_detected", 0.0)) != 0 for row in train_rows
    )
    report = {
        "output_dir": str(output_dir),
        "training_completed": bool(train_rows and (output_dir / "latest" / "model.pt").exists()),
        "final_steps": int(latest_meta.get("total_env_steps_actual") or _float(final_row, "total_steps", 0.0)),
        "policy_arch": latest_meta.get("policy_arch", "unknown"),
        "nan_detected": nan_detected,
        "latest_checkpoint_exists": (output_dir / "latest" / "model.pt").exists(),
        "best_checkpoint_exists": (output_dir / "best" / "model.pt").exists(),
        "latest_meta_exists": bool(latest_meta),
        "best_meta_exists": bool(best_meta),
        "eval_success": bool(eval_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "loss_ranges": {
            "actor_loss_mav": _range(train_rows, "actor_loss_mav"),
            "actor_loss_uav": _range(train_rows, "actor_loss_uav"),
            "critic_loss": _range(train_rows, "critic_loss"),
        },
        "entropy_ranges": {
            "entropy_mav": _range(train_rows, "entropy_mav"),
            "entropy_uav": _range(train_rows, "entropy_uav"),
        },
        "log_std_ranges": {
            "action_log_std_mav_min": _range(train_rows, "action_log_std_mav_min"),
            "action_log_std_mav_max": _range(train_rows, "action_log_std_mav_max"),
            "action_log_std_mav_mean": _range(train_rows, "action_log_std_mav_mean"),
            "action_log_std_uav_min": _range(train_rows, "action_log_std_uav_min"),
            "action_log_std_uav_max": _range(train_rows, "action_log_std_uav_max"),
            "action_log_std_uav_mean": _range(train_rows, "action_log_std_uav_mean"),
        },
        "action_saturation_ranges": {
            "mav_action_saturation_rate": _range(train_rows, "mav_action_saturation_rate"),
            "uav_action_saturation_rate": _range(train_rows, "uav_action_saturation_rate"),
        },
        "event_totals": {
            "red_missiles_fired": sum(_float(row, "red_missiles_fired") for row in train_rows),
            "missile_hits": sum(_float(row, "missile_hits") for row in train_rows),
        },
    }
    report["recommend_enter_p3_gru"] = bool(
        report["training_completed"]
        and report["eval_success"]
        and not report["nan_detected"]
        and report["policy_arch"] == "entity_attention"
    )
    if not report["recommend_enter_p3_gru"]:
        report["p3_blocking_reason"] = "entity_attention sanity run did not complete cleanly"
    else:
        report["p3_blocking_reason"] = ""
    return report


def _write_md(path: Path, report: dict) -> None:
    lines = [
        "# Entity Attention Sanity Report",
        "",
        f"- output_dir: `{report['output_dir']}`",
        f"- training_completed: {report['training_completed']}",
        f"- final_steps: {report['final_steps']}",
        f"- policy_arch: {report['policy_arch']}",
        f"- nan_detected: {report['nan_detected']}",
        f"- eval_success: {report['eval_success']}",
        f"- latest_checkpoint_exists: {report['latest_checkpoint_exists']}",
        f"- best_checkpoint_exists: {report['best_checkpoint_exists']}",
        f"- recommend_enter_p3_gru: {report['recommend_enter_p3_gru']}",
        "",
        "## Loss Ranges",
        "",
    ]
    for key, value in report["loss_ranges"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Entropy Ranges", ""])
    for key, value in report["entropy_ranges"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Log Std Ranges", ""])
    for key, value in report["log_std_ranges"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Events", ""])
    for key, value in report["event_totals"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/debug_entity_attention_10k_sanity")
    args = parser.parse_args()
    out_dir = _resolve(args.output_dir)
    report = analyze(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entity_sanity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out_dir / "entity_sanity_report.md", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
