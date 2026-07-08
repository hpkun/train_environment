"""Offline pure-HAPPO staged degradation audit.

Reads an existing training output directory and writes a reproducible summary
under ``<output-dir>/degradation_audit``.  The audit is descriptive only: it
does not train, evaluate, or modify checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_FIELDS = [
    "avg_return", "red_win", "blue_win", "draw", "timeout",
    "mav_survival", "red_alive_final", "blue_alive_final",
    "red_missiles_fired", "red_missile_hits",
    "blue_missiles_fired", "blue_missile_hits",
    "critic_loss", "entropy_mav", "entropy_uav",
    "approx_kl_mav", "approx_kl_uav",
    "mav_active_sample_count", "uav_active_sample_count",
    "action_log_std_mav_mean", "action_log_std_uav_mean",
    "mav_action_saturation_rate", "uav_action_saturation_rate",
]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _first_existing(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            return _num(row, key, default)
    return default


def _mean(rows: list[dict], *keys: str) -> float:
    vals = [_first_existing(row, *keys) for row in rows]
    return mean(vals) if vals else 0.0


def _split_stages(rows: list[dict], windows: int) -> list[tuple[str, list[dict]]]:
    if not rows:
        return []
    windows = max(1, int(windows))
    size = max(1, math.ceil(len(rows) / windows))
    out = []
    for idx in range(0, len(rows), size):
        part = rows[idx:idx + size]
        first_step = int(_num(part[0], "total_steps"))
        last_step = int(_num(part[-1], "total_steps"))
        out.append((f"{first_step}-{last_step}", part))
    return out


def _anomaly_flags(part: list[dict]) -> list[str]:
    flags: list[str] = []
    if not part:
        return flags
    if _mean(part, "approx_kl_mav") > 0.2 or _mean(part, "approx_kl_uav") > 0.2:
        flags.append("high_approx_kl")
    if _mean(part, "critic_loss") > 1.0e4:
        flags.append("critic_loss_large")
    if _mean(part, "entropy_mav") <= 1.0e-6 and _mean(part, "mav_active_sample_count") > 0:
        flags.append("mav_entropy_zero_with_active_samples")
    if _mean(part, "entropy_uav") <= 1.0e-6 and _mean(part, "uav_active_sample_count") > 0:
        flags.append("uav_entropy_zero_with_active_samples")
    if _mean(part, "action_log_std_mav_mean") > 2.0 or _mean(part, "action_log_std_uav_mean") > 2.0:
        flags.append("log_std_high")
    if _mean(part, "action_log_std_mav_mean") < -5.0 or _mean(part, "action_log_std_uav_mean") < -5.0:
        flags.append("log_std_low")
    if _mean(part, "mav_action_saturation_rate") > 0.5 or _mean(part, "uav_action_saturation_rate") > 0.5:
        flags.append("action_saturation_high")
    return flags


def build_degradation_audit(output_dir: str | Path, windows: int = 6) -> dict:
    run_dir = Path(output_dir)
    out_dir = run_dir / "degradation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _read_rows(run_dir / "train_log.csv")
    eval_rows = _read_rows(run_dir / "eval_log.csv")
    reward_rows = _read_rows(run_dir / "rich_logs" / "episode_reward_components.csv")

    stage_rows: list[dict] = []
    previous: dict | None = None
    for idx, (stage, part) in enumerate(_split_stages(train_rows, windows)):
        red_fire = _mean(part, "red_episode_missiles_fired_mean", "red_missiles_fired")
        red_hit = _mean(part, "red_episode_missile_hits_mean", "red_missile_hits")
        blue_alive = _mean(part, "blue_alive_final")
        row = {
            "stage_index": idx,
            "stage_steps": stage,
            "rows": len(part),
            "avg_return_mean": _mean(part, "avg_return"),
            "avg_return_final": _num(part[-1], "avg_return"),
            "red_win_mean": _mean(part, "red_win"),
            "blue_win_mean": _mean(part, "blue_win"),
            "draw_mean": _mean(part, "draw"),
            "timeout_mean": _mean(part, "timeout"),
            "mav_survival_mean": _mean(part, "mav_survival"),
            "red_alive_final_mean": _mean(part, "red_alive_final"),
            "blue_alive_final_mean": blue_alive,
            "red_missiles_fired_mean": red_fire,
            "red_missile_hits_mean": red_hit,
            "blue_missiles_fired_mean": _mean(part, "blue_episode_missiles_fired_mean", "blue_missiles_fired"),
            "blue_missile_hits_mean": _mean(part, "blue_episode_missile_hits_mean", "blue_missile_hits"),
            "critic_loss_mean": _mean(part, "critic_loss"),
            "approx_kl_mav_mean": _mean(part, "approx_kl_mav"),
            "approx_kl_uav_mean": _mean(part, "approx_kl_uav"),
            "entropy_mav_mean": _mean(part, "entropy_mav"),
            "entropy_uav_mean": _mean(part, "entropy_uav"),
            "mav_active_sample_count_mean": _mean(part, "mav_active_sample_count"),
            "uav_active_sample_count_mean": _mean(part, "uav_active_sample_count"),
            "action_log_std_mav_mean": _mean(part, "action_log_std_mav_mean"),
            "action_log_std_uav_mean": _mean(part, "action_log_std_uav_mean"),
            "mav_action_saturation_rate_mean": _mean(part, "mav_action_saturation_rate"),
            "uav_action_saturation_rate_mean": _mean(part, "uav_action_saturation_rate"),
        }
        suspicious = (
            row["red_win_mean"] > 0.25
            and row["timeout_mean"] > 0.25
            and red_hit <= 0.05
            and blue_alive >= 1.0
        )
        attack_effective = red_fire > 0.25 and (red_hit > 0.05 or blue_alive < 1.5)
        regression = False
        if previous is not None:
            regression = (
                row["avg_return_mean"] < previous["avg_return_mean"] - 5.0
                or row["red_win_mean"] < previous["red_win_mean"] - 0.2
                or red_fire < previous["red_missiles_fired_mean"] * 0.5
            )
        row["suspicious_survival_win"] = int(suspicious)
        row["attack_effective"] = int(attack_effective)
        row["regression_from_previous_stage"] = int(regression)
        row["anomaly_flags"] = ";".join(_anomaly_flags(part))
        stage_rows.append(row)
        previous = row

    _write_rows(out_dir / "degradation_summary.csv", stage_rows)
    summary = {
        "output_dir": str(run_dir),
        "train_log_rows": len(train_rows),
        "eval_log_rows": len(eval_rows),
        "reward_component_rows": len(reward_rows),
        "stage_count": len(stage_rows),
        "latest_train": train_rows[-1] if train_rows else {},
        "stage_rows": stage_rows,
        "notes": [
            "train_log fields are recent-window training summaries, not formal evaluation.",
            "suspicious_survival_win flags red wins that appear timeout/survival based rather than attack-effective.",
            "attack_effective is heuristic and should be checked against eval and missile event logs.",
        ],
    }
    (out_dir / "degradation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir / "degradation_summary.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Pure-HAPPO Degradation Audit",
        "",
        f"- Output dir: `{summary['output_dir']}`",
        f"- train_log rows: {summary['train_log_rows']}",
        f"- eval_log rows: {summary['eval_log_rows']}",
        f"- reward component rows: {summary['reward_component_rows']}",
        "",
        "This report is offline-only. `train_log.csv` is treated as recent-window training telemetry, not formal eval.",
        "",
        "## Stage Summary",
        "",
        "| stage | return | R win | B win | timeout | red fire | red hit | blue alive | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["stage_rows"]:
        flags = ",".join([
            name for name in [
                "suspicious_survival_win" if row["suspicious_survival_win"] else "",
                "attack_effective" if row["attack_effective"] else "",
                "regression" if row["regression_from_previous_stage"] else "",
                row.get("anomaly_flags", ""),
            ] if name
        ])
        lines.append(
            f"| {row['stage_steps']} | {row['avg_return_mean']:.3f} | "
            f"{row['red_win_mean']:.3f} | {row['blue_win_mean']:.3f} | "
            f"{row['timeout_mean']:.3f} | {row['red_missiles_fired_mean']:.3f} | "
            f"{row['red_missile_hits_mean']:.3f} | {row['blue_alive_final_mean']:.3f} | {flags} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=6)
    args = parser.parse_args()
    summary = build_degradation_audit(args.output_dir, windows=args.windows)
    print(f"wrote {Path(args.output_dir) / 'degradation_audit'}")
    print(f"stage_count={summary['stage_count']} train_rows={summary['train_log_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
