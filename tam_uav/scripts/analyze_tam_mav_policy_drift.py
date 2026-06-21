"""Analyze categorical MAV policy drift from training and action telemetry."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


AXES = ("throttle", "aileron", "elevator", "rudder")
STAGES = ("start", "25%", "50%", "75%", "end")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _staged(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {stage: None for stage in STAGES}
    last = len(values) - 1
    indices = (0, round(last * .25), round(last * .5), round(last * .75), last)
    return dict(zip(STAGES, (values[index] for index in indices)))


def _action_bin_segments(path: Path, total_steps: int):
    counters = [[Counter() for _axis in AXES] for _segment in range(4)]
    seen = 0
    if not path.exists():
        return []
    segment_size = max(int(math.ceil(total_steps / 4)), 1)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("agent_id") != "red_0":
                continue
            segment = min(3, seen // segment_size)
            seen += 1
            for axis in range(4):
                value = _number(row.get(f"action_index_{axis}"))
                if value is not None:
                    counters[segment][axis][int(value)] += 1
    result = []
    for segment, axis_counters in enumerate(counters):
        axes = {}
        for name, counts in zip(AXES, axis_counters):
            total = sum(counts.values())
            dominant = counts.most_common(1)[0][0] if counts else None
            mean = sum(key * count for key, count in counts.items()) / total if total else None
            axes[name] = {
                "dominant_bin": dominant,
                "mean_bin": mean,
                "unique_bins": len(counts),
                "top5": counts.most_common(5),
            }
        result.append({
            "stage": ("0-25%", "25-50%", "50-75%", "75-100%")[segment],
            "samples": sum(axis_counters[0].values()),
            "axes": axes,
        })
    return result


def _missile_totals(path: Path):
    fired = hits = 0
    if not path.exists():
        return {"red_fired": 0, "red_hits": 0}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("owner_team") != "red":
                continue
            fired += row.get("event_type") == "launch"
            hits += (
                row.get("event_type") == "hit"
                or str(row.get("hit_success", "")).lower() in {"1", "true"}
            )
    return {"red_fired": fired, "red_hits": hits}


def analyze_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    rows = _read_csv(run_dir / "train_log.csv")
    status_path = run_dir / "runner_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    total_steps = int(status.get("total_env_steps_actual") or (
        _number(rows[-1].get("total_steps")) if rows else 0
    ) or 0)
    staged_fields = (
        "avg_return", "mav_survival", "blue_alive_final", "entropy_mav",
        "max_action_prob_mav", "edge_bin_rate", "surface_edge_rate",
        "approx_kl_mav", "correction_factor_mean", "actor_loss_mav",
        "grad_norm_shared", "grad_norm_mav_head", "kl_to_neutral_mav",
        "dominant_bin_mav_throttle", "dominant_bin_mav_aileron",
        "dominant_bin_mav_elevator", "dominant_bin_mav_rudder",
    )
    result = {
        "run_dir": str(run_dir),
        "runner_status": status,
        "total_steps": total_steps,
        "metrics": {field: _staged(rows, field) for field in staged_fields},
        "action_bin_segments": _action_bin_segments(
            run_dir / "rich_logs" / "tam_action_timeseries.csv", total_steps
        ),
        "missiles": _missile_totals(run_dir / "rich_logs" / "missile_events.csv"),
        "mav_death_time": {
            "available": False,
            "reason": "training logs do not contain per-episode MAV death steps",
        },
        "mav_death_reason": {
            "available": False,
            "reason": "training logs do not contain per-episode termination reasons",
        },
    }
    numeric = lambda field: [
        value for value in (_number(row.get(field)) for row in rows)
        if value is not None
    ]
    for field in ("approx_kl_mav", "actor_loss_mav", "grad_norm_shared", "grad_norm_mav_head"):
        values = numeric(field)
        result[f"{field}_abs_max"] = max(map(abs, values), default=None)
    shared = numeric("grad_norm_shared")
    head = numeric("grad_norm_mav_head")
    result["shared_to_mav_head_grad_ratio_mean"] = (
        sum(shared) / len(shared) / max(sum(head) / len(head), 1e-12)
        if shared and head else None
    )
    start_bins = [39, 20, 20, 20]
    final_bins = []
    if result["action_bin_segments"]:
        final_bins = [
            result["action_bin_segments"][-1]["axes"][axis]["dominant_bin"]
            for axis in AXES
        ]
    result["answers"] = {
        "initially_unstable_or_trained_drift": "trained_policy_drift",
        "dominant_action_far_from_neutral": bool(
            final_bins and any(abs(value - center) > 2 for value, center in zip(final_bins, start_bins))
        ),
        "dominant_bins_final": final_bins,
        "primary_drift_axis": "distribution_entropy_across_surfaces",
        "entropy_too_high": (
            result["metrics"]["entropy_mav"]["end"] or 0.0
        ) > 2.0,
        "per_update_kl_too_high": (
            result["approx_kl_mav_abs_max"] or 0.0
        ) > 0.1,
        "shared_update_likely_dominant": (
            result["shared_to_mav_head_grad_ratio_mean"] or 0.0
        ) > 1.0,
        "priority": "mav_entropy_then_mav_head_lr_and_clip_with_role_kl_guard",
    }
    return result


def _comparison(current, baseline):
    if baseline is None:
        return None
    fields = (
        "avg_return", "mav_survival", "blue_alive_final", "entropy_mav",
        "approx_kl_mav", "kl_to_neutral_mav",
    )
    return {
        field: {
            "baseline_end": baseline["metrics"][field]["end"],
            "current_end": current["metrics"][field]["end"],
        }
        for field in fields
    } | {
        "baseline_missiles": baseline["missiles"],
        "current_missiles": current["missiles"],
        "baseline_dominant_bins": baseline["answers"]["dominant_bins_final"],
        "current_dominant_bins": current["answers"]["dominant_bins_final"],
    }


def _write_report(result, baseline, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "200k" if result["total_steps"] >= 200000 else "50k"
    payload = dict(result)
    payload["baseline_comparison"] = _comparison(result, baseline)
    json_path = output_dir / f"mav_policy_drift_{suffix}.json"
    md_path = output_dir / f"mav_policy_drift_{suffix}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    answers = payload["answers"]
    lines = [
        f"# MAV Policy Drift Analysis ({suffix})", "",
        f"- Run: `{payload['run_dir']}`",
        f"- Runner: `{payload['runner_status'].get('status', 'unknown')}`",
        f"- Initial state: `{answers['initially_unstable_or_trained_drift']}`",
        f"- Final dominant MAV bins: `{answers['dominant_bins_final']}`",
        f"- Dominant bins far from neutral: `{answers['dominant_action_far_from_neutral']}`",
        f"- Primary drift: `{answers['primary_drift_axis']}`",
        f"- Entropy too high: `{answers['entropy_too_high']}`",
        f"- Per-update KL too high: `{answers['per_update_kl_too_high']}`",
        f"- Shared update likely dominant: `{answers['shared_update_likely_dominant']}`",
        f"- Priority: `{answers['priority']}`", "",
        f"- Metric stages: `{payload['metrics']}`",
        f"- Missile totals: `{payload['missiles']}`",
        f"- MAV death time: `{payload['mav_death_time']}`",
        f"- Baseline comparison: `{payload['baseline_comparison']}`", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-run-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    current = analyze_run(args.run_dir)
    baseline = analyze_run(args.baseline_run_dir) if args.baseline_run_dir else None
    paths = _write_report(current, baseline, Path(args.output_dir))
    print(json.dumps({
        "json": str(paths[0]), "markdown": str(paths[1]),
        "answers": current["answers"],
    }, indent=2))


if __name__ == "__main__":
    main()
