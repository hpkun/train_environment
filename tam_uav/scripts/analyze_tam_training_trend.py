"""Summarize TAM training direction without comparing incompatible reward scales."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ALIASES = {
    "total_steps": ("total_steps", "total_env_steps_actual", "train_steps"),
    "avg_return": ("avg_return", "avg_episode_return"),
    "red_win": ("red_win", "red_win_rate"),
    "blue_win": ("blue_win", "blue_win_rate"),
    "mav_survival": ("mav_survival", "mav_survival_rate"),
    "red_missiles_fired": ("red_missiles_fired", "red_missiles_fired_mean"),
    "red_missile_hits": ("red_missile_hits", "red_missile_hits_mean", "missile_hits"),
    "blue_alive_final": ("blue_alive_final", "blue_alive_final_mean"),
    "entropy_mav": ("entropy_mav",),
    "entropy_uav": ("entropy_uav",),
    "action_bin_usage_mav": ("action_bin_usage_mav",),
    "action_bin_usage_uav": ("action_bin_usage_uav",),
    "correction_factor_mean": ("correction_factor_mean",),
    "approx_kl_mav": ("approx_kl_mav",),
    "approx_kl_uav": ("approx_kl_uav",),
    "nan_detected": ("nan_detected",),
}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _value(row, field):
    for name in ALIASES.get(field, (field,)):
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def staged_values(rows, field):
    values = [_value(row, field) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {key: None for key in ("start", "25%", "50%", "75%", "end")}
    last = len(values) - 1
    indices = [0, round(last * 0.25), round(last * 0.5), round(last * 0.75), last]
    return dict(zip(("start", "25%", "50%", "75%", "end"), [values[index] for index in indices]))


def summarize_missile_event_rows(rows):
    red_rows = [row for row in rows if row.get("owner_team") == "red"]
    max_step = max((_number(row.get("step")) or 0.0 for row in rows), default=0.0)
    labels = ("start", "25%", "50%", "75%", "end")
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    fired, hits, rates = {}, {}, {}
    for label, fraction in zip(labels, fractions):
        threshold = max_step * fraction
        selected = [row for row in red_rows if (_number(row.get("step")) or 0.0) <= threshold]
        launch_count = sum(row.get("event_type") == "launch" for row in selected)
        hit_count = sum(
            row.get("event_type") == "hit" or str(row.get("hit_success", "")).lower() in ("1", "true")
            for row in selected
        )
        fired[label] = float(launch_count)
        hits[label] = float(hit_count)
        rates[label] = hit_count / launch_count if launch_count else 0.0
    return {"red_fired": fired, "red_hits": hits, "red_hit_rate": rates}


def _slope(rows, field, tail_fraction=0.25):
    points = [
        (_value(row, "total_steps"), _value(row, field)) for row in rows
    ]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if len(points) < 2:
        return None
    tail_count = max(2, int(math.ceil(len(points) * tail_fraction)))
    x = [point[0] for point in points[-tail_count:]]
    y = [point[1] for point in points[-tail_count:]]
    slope = _linear_slope(x, y)
    if slope is None:
        return 0.0
    return float(slope * 10000.0)


def _linear_slope(x, y):
    if len(x) < 2 or len(x) != len(y):
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 0:
        return None
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y)
    ) / denominator


def classify_stage(summary):
    audit = summary.get("environment_audit") or {}
    if audit and not audit.get("reset_contract_passed", False):
        return "D"
    eval_survival_collapsed = any(
        trend.get("mav_survival", {}).get("start", 0.0) >= 0.5
        and trend.get("mav_survival", {}).get("end", 1.0) <= 0.1
        for trend in (summary.get("eval_trend") or {}).values()
    )
    policy_survival_collapsed = (
        audit.get("passed", False)
        and summary.get("mav_survival", {}).get("end") == 0.0
        and eval_survival_collapsed
    )
    correction = summary.get("correction_factor", {}).get("end")
    kl_values = [
        summary.get("approx_kl_mav", {}).get("end"),
        summary.get("approx_kl_uav", {}).get("end"),
    ]
    if (
        summary.get("collapse_detected")
        or policy_survival_collapsed
        or correction is not None and not 0.1 <= correction <= 10.0
        or any(value is not None and abs(value) > 1.0 for value in kl_values)
    ):
        return "C"
    if (
        audit and not audit.get("passed", False)
        and audit.get("reset_contract_passed", False)
        and not audit.get("f22_speed_at_60s_passed", False)
    ):
        return "B"
    death = summary.get("mav_death_step") or {}
    if (
        summary.get("mav_survival", {}).get("end") == 0.0
        and death.get("end") is not None
        and (death.get("slope") or 0.0) <= 0.0
    ):
        return "B"
    improvements = [
        (summary.get("rolling_return_slope_per_10k_steps") or 0.0) > 0.0,
        (summary.get("red_fired", {}).get("end") or 0.0) > (summary.get("red_fired", {}).get("start") or 0.0),
        (summary.get("red_hits", {}).get("end") or 0.0) > (summary.get("red_hits", {}).get("start") or 0.0),
        (death.get("slope") or 0.0) > 0.0,
    ]
    return "A" if any(improvements) else "D"


def summarize_training_rows(rows, mav_death_step=None):
    returns = staged_values(rows, "avg_return")
    fired = staged_values(rows, "red_missiles_fired")
    hits = staged_values(rows, "red_missile_hits")
    hit_rate = {}
    for key in returns:
        denominator = fired[key]
        hit_rate[key] = (
            hits[key] / denominator if denominator not in (None, 0.0) and hits[key] is not None else 0.0
        )
    metric_fields = {
        "red_win": "red_win", "blue_win": "blue_win",
        "mav_survival": "mav_survival", "blue_alive": "blue_alive_final",
        "entropy_mav": "entropy_mav", "entropy_uav": "entropy_uav",
        "action_bin_usage_mav": "action_bin_usage_mav",
        "action_bin_usage_uav": "action_bin_usage_uav",
        "correction_factor": "correction_factor_mean",
        "approx_kl_mav": "approx_kl_mav", "approx_kl_uav": "approx_kl_uav",
    }
    summary = {
        "rows": len(rows),
        "return_stages": returns,
        "final_return": returns["end"],
        "rolling_return_slope_per_10k_steps": _slope(rows, "avg_return"),
        "red_fired": fired,
        "red_hits": hits,
        "red_hit_rate": hit_rate,
        "mav_death_step": mav_death_step,
    }
    summary.update({name: staged_values(rows, field) for name, field in metric_fields.items()})
    nan_detected = any((_value(row, "nan_detected") or 0.0) != 0.0 for row in rows)
    low_entropy = any(
        summary[name]["end"] is not None and summary[name]["end"] < 0.05
        for name in ("entropy_mav", "entropy_uav")
    )
    collapsed_usage = any(
        summary[name]["end"] is not None and summary[name]["end"] < 0.025
        for name in ("action_bin_usage_mav", "action_bin_usage_uav")
    )
    summary["collapse_detected"] = bool(nan_detected or low_entropy or collapsed_usage)
    summary["stage_decision"] = classify_stage(summary)
    return summary


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _mav_death_trend(path):
    rows = _read_csv(path)
    episodes = {}
    for row in rows:
        if row.get("agent_id") != "red_0":
            continue
        key = (row.get("scenario"), row.get("episode_id"))
        episodes.setdefault(key, []).append(row)
    death_steps = []
    for episode_rows in episodes.values():
        dead = [row for row in episode_rows if str(row.get("alive", "")).lower() in ("0", "false")]
        if dead:
            value = _number(dead[0].get("step"))
            if value is not None:
                death_steps.append(value)
    if not death_steps:
        return None
    slope = _linear_slope(list(range(len(death_steps))), death_steps) or 0.0
    return {"start": death_steps[0], "end": death_steps[-1], "slope": slope}


def _eval_summary(rows):
    by_config = {}
    for row in rows:
        config = row.get("config", "unknown")
        by_config.setdefault(config, []).append(row)
    return {
        config: {
            "red_win": staged_values(config_rows, "red_win_rate"),
            "blue_win": staged_values(config_rows, "blue_win_rate"),
            "mav_survival": staged_values(config_rows, "mav_survival_rate"),
        }
        for config, config_rows in by_config.items()
    }


def analyze_run(run_dir, baseline_50k=None, paper_reference_note=None):
    run_dir = Path(run_dir)
    train_rows = _read_csv(run_dir / "train_log.csv")
    rich_rows = _read_csv(run_dir / "rich_logs" / "train_metrics.csv")
    rows = train_rows if train_rows else rich_rows
    death = _mav_death_trend(run_dir / "rich_logs" / "aircraft_timeseries.csv")
    summary = summarize_training_rows(rows, mav_death_step=death)
    summary.update({
        "run_dir": str(run_dir),
        "eval_trend": _eval_summary(_read_csv(run_dir / "eval_log.csv")),
        "paper_reference_note": paper_reference_note,
        "reward_scale_comparison": "absolute reward scale may differ; within-run direction is comparable",
    })
    audit_path = run_dir.parent / "environment_audit" / "tam_airborne_initialization.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        summary["environment_audit"] = {
            "passed": bool(audit.get("passed")),
            "reset_contract_passed": bool(
                audit.get("reset_contract", {}).get("passed_reset_contract")
            ),
            "f22_speed_at_60s_passed": bool(
                audit.get("fixed_neutral_120s", {}).get("f22_speed_at_60s_passed")
            ),
        }
    missile_rows = _read_csv(run_dir / "rich_logs" / "missile_events.csv")
    if missile_rows:
        summary.update(summarize_missile_event_rows(missile_rows))
    if baseline_50k:
        baseline_dir = Path(baseline_50k)
        baseline_rows = _read_csv(baseline_dir / "train_log.csv")
        summary["baseline_50k"] = summarize_training_rows(baseline_rows)
    summary["stage_decision"] = classify_stage(summary)
    (run_dir / "trend_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# TAM Training Trend Summary", "",
        f"- Stage decision: **{summary['stage_decision']}**",
        f"- Final return: `{summary['final_return']}`",
        f"- Rolling return slope / 10k steps: `{summary['rolling_return_slope_per_10k_steps']}`",
        f"- Return stages: `{summary['return_stages']}`",
        f"- Red fired: `{summary['red_fired']}`",
        f"- Red hits: `{summary['red_hits']}`",
        f"- MAV survival: `{summary['mav_survival']}`",
        f"- MAV death step: `{summary['mav_death_step']}`",
        f"- Blue alive: `{summary['blue_alive']}`",
        f"- Entropy MAV/UAV: `{summary['entropy_mav']}` / `{summary['entropy_uav']}`",
        f"- Action bin usage MAV/UAV: `{summary['action_bin_usage_mav']}` / `{summary['action_bin_usage_uav']}`",
        f"- Correction factor: `{summary['correction_factor']}`",
        f"- Approx KL MAV/UAV: `{summary['approx_kl_mav']}` / `{summary['approx_kl_uav']}`",
        f"- Collapse detected: `{summary['collapse_detected']}`", "",
        f"Paper reference note: {paper_reference_note or 'not provided'}", "",
    ]
    (run_dir / "trend_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-50k")
    parser.add_argument("--paper-reference-note")
    args = parser.parse_args()
    summary = analyze_run(
        args.run_dir, baseline_50k=args.baseline_50k,
        paper_reference_note=args.paper_reference_note,
    )
    print(json.dumps({
        "stage_decision": summary["stage_decision"],
        "final_return": summary["final_return"],
        "rolling_return_slope_per_10k_steps": summary["rolling_return_slope_per_10k_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
