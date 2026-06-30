"""Analyze blue safe-pursuit rollout diagnostics.

The input is the diagnostics directory written by the ACMI exporters.  This
script intentionally reads the CSV diagnostics instead of parsing Tacview ACMI;
it is a lightweight behavior audit, not a trajectory renderer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BATTLEFIELD_HALF_SIZE = 40000.0
BATTLEFIELD_ALTITUDE_MAX = 10000.0


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _long_segments(mask: list[bool], min_len: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    start = None
    for i, value in enumerate(mask + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length >= min_len:
                out.append({"start_index": start, "end_index": i - 1, "length": length})
            start = None
    return out


def _increasing_segments(values: list[float], min_len: int) -> list[dict[str, int]]:
    mask = []
    prev = None
    for value in values:
        mask.append(prev is not None and math.isfinite(value) and math.isfinite(prev) and value > prev)
        prev = value
    return _long_segments(mask, min_len)


def _ned_to_body_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
    ], dtype=np.float64)


def _geo_heading_from_body_row(row: dict[str, str]) -> float:
    x = _float(row.get("selected_rel_body_x_norm")) * BATTLEFIELD_HALF_SIZE
    y = _float(row.get("selected_rel_body_y_norm")) * BATTLEFIELD_HALF_SIZE
    up = _float(row.get("selected_rel_body_up_norm")) * BATTLEFIELD_ALTITUDE_MAX
    if not all(math.isfinite(v) for v in (x, y, up)):
        return float("nan")
    body = np.array([x, y, -up], dtype=np.float64)
    roll = math.radians(_float(row.get("roll_deg"), 0.0))
    pitch = math.radians(_float(row.get("pitch_deg"), 0.0))
    yaw = math.radians(_float(row.get("yaw_deg", row.get("heading_deg")), 0.0))
    ned = _ned_to_body_matrix(roll, pitch, yaw).T @ body
    return math.atan2(float(ned[1]), float(ned[0]))


def _summary_for_agent(rows: list[dict[str, str]]) -> dict[str, Any]:
    speeds = [_float(r.get("speed_mps")) for r in rows if math.isfinite(_float(r.get("speed_mps")))]
    alts = [_float(r.get("alt_m", r.get("altitude_m"))) for r in rows if math.isfinite(_float(r.get("alt_m", r.get("altitude_m"))))]
    rolls = [abs(_float(r.get("roll_deg"))) for r in rows if math.isfinite(_float(r.get("roll_deg")))]
    pitches = [abs(_float(r.get("pitch_deg"))) for r in rows if math.isfinite(_float(r.get("pitch_deg")))]
    ranges = [_float(r.get("nearest_enemy_range_m")) for r in rows]
    ranges_finite = [v for v in ranges if math.isfinite(v)]
    sources = [str(r.get("desired_heading_source", "")) for r in rows]
    frames = len(rows)

    def count_source(name: str) -> int:
        return sum(1 for s in sources if s == name)

    center_dist = [
        math.hypot(_float(r.get("north_m")), _float(r.get("east_m")))
        for r in rows
        if math.isfinite(_float(r.get("north_m"))) and math.isfinite(_float(r.get("east_m")))
    ]
    low_speed_mask = [str(r.get("desired_heading_source", "")) == "low_speed_recovery" for r in rows]
    high_roll_low_speed = [
        abs(_float(r.get("roll_deg"))) > 75.0 and _float(r.get("speed_mps")) < 220.0
        for r in rows
    ]
    current_target_not_closing = [
        i > 0
        and str(rows[i].get("desired_heading_source", "")) == "current_target"
        and math.isfinite(ranges[i])
        and math.isfinite(ranges[i - 1])
        and ranges[i] > ranges[i - 1]
        for i in range(len(rows))
    ]

    diffs = []
    high_roll_diffs = []
    low_speed_pre_diffs = []
    current_target_range_increase_diffs = []
    for i, row in enumerate(rows):
        old_heading = _float(row.get("action_heading_abs_rad"))
        geo_heading = _geo_heading_from_body_row(row)
        if not (math.isfinite(old_heading) and math.isfinite(geo_heading)):
            continue
        diff = abs(_wrap_pi(old_heading - geo_heading))
        diffs.append(diff)
        if abs(_float(row.get("roll_deg"))) > 75.0:
            high_roll_diffs.append(diff)
        if any(low_speed_mask[j] for j in range(i + 1, min(len(rows), i + 11))):
            low_speed_pre_diffs.append(diff)
        if i > 0 and str(row.get("desired_heading_source", "")) == "current_target" and math.isfinite(ranges[i]) and math.isfinite(ranges[i - 1]) and ranges[i] > ranges[i - 1]:
            current_target_range_increase_diffs.append(diff)

    return {
        "frames": frames,
        "alive_final": int(_float(rows[-1].get("alive"), 0.0)) if rows else 0,
        "death_reason": str(rows[-1].get("death_reason", "")) if rows else "",
        "speed_min": float(np.min(speeds)) if speeds else None,
        "speed_p5": float(np.percentile(speeds, 5)) if speeds else None,
        "speed_mean": float(np.mean(speeds)) if speeds else None,
        "speed_max": float(np.max(speeds)) if speeds else None,
        "alt_min": float(np.min(alts)) if alts else None,
        "alt_p5": float(np.percentile(alts, 5)) if alts else None,
        "alt_mean": float(np.mean(alts)) if alts else None,
        "roll_abs_max": float(np.max(rolls)) if rolls else None,
        "roll_abs_p95": float(np.percentile(rolls, 95)) if rolls else None,
        "roll_abs_gt_75_rate": float(np.mean([r > 75.0 for r in rolls])) if rolls else None,
        "pitch_abs_max": float(np.max(pitches)) if pitches else None,
        "pitch_abs_p95": float(np.percentile(pitches, 95)) if pitches else None,
        "nearest_enemy_range_start": ranges_finite[0] if ranges_finite else None,
        "nearest_enemy_range_end": ranges_finite[-1] if ranges_finite else None,
        "nearest_enemy_range_min": float(np.min(ranges_finite)) if ranges_finite else None,
        "nearest_enemy_range_delta": (ranges_finite[-1] - ranges_finite[0]) if len(ranges_finite) >= 2 else None,
        "distance_to_center_start": center_dist[0] if center_dist else None,
        "distance_to_center_end": center_dist[-1] if center_dist else None,
        "distance_to_center_max": float(np.max(center_dist)) if center_dist else None,
        "current_target_count": count_source("current_target"),
        "current_target_rate": count_source("current_target") / frames if frames else 0.0,
        "low_speed_recovery_count": count_source("low_speed_recovery"),
        "low_speed_recovery_rate": count_source("low_speed_recovery") / frames if frames else 0.0,
        "boundary_safety_count": count_source("safety"),
        "boundary_safety_rate": count_source("safety") / frames if frames else 0.0,
        "reacquire_last_seen_count": count_source("reacquire_last_seen"),
        "reacquire_last_seen_rate": count_source("reacquire_last_seen") / frames if frames else 0.0,
        "center_cruise_count": count_source("center_cruise"),
        "center_cruise_rate": count_source("center_cruise") / frames if frames else 0.0,
        "hold_heading_count": count_source("hold_heading"),
        "hold_heading_rate": count_source("hold_heading") / frames if frames else 0.0,
        "long_low_speed_recovery_segments": _long_segments(low_speed_mask, 30),
        "fly_away_segments": _increasing_segments(ranges, 60),
        "vertical_climb_segments": _increasing_segments(alts, 60),
        "high_roll_low_speed_overlap_rate": float(np.mean(high_roll_low_speed)) if high_roll_low_speed else 0.0,
        "target_visible_but_not_closing_segments": _long_segments(current_target_not_closing, 30),
        "old_geo_heading_diff_mean": float(np.mean(diffs)) if diffs else None,
        "old_geo_heading_diff_p50": float(np.percentile(diffs, 50)) if diffs else None,
        "old_geo_heading_diff_p95": float(np.percentile(diffs, 95)) if diffs else None,
        "old_geo_heading_diff_max": float(np.max(diffs)) if diffs else None,
        "high_roll_old_geo_heading_diff_mean": float(np.mean(high_roll_diffs)) if high_roll_diffs else None,
        "high_roll_old_geo_heading_diff_p95": float(np.percentile(high_roll_diffs, 95)) if high_roll_diffs else None,
        "pre_low_speed_old_geo_heading_diff_mean": float(np.mean(low_speed_pre_diffs)) if low_speed_pre_diffs else None,
        "pre_low_speed_old_geo_heading_diff_p95": float(np.percentile(low_speed_pre_diffs, 95)) if low_speed_pre_diffs else None,
        "current_target_range_increase_old_geo_heading_diff_mean": float(np.mean(current_target_range_increase_diffs)) if current_target_range_increase_diffs else None,
    }


def _natural_summary(agent_summaries: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for aid, s in agent_summaries.items():
        closing = s.get("nearest_enemy_range_delta")
        pursuing = closing is not None and closing < 0
        low_speed_long = bool(s.get("long_low_speed_recovery_segments"))
        fly_away = bool(s.get("fly_away_segments"))
        vertical = bool(s.get("vertical_climb_segments"))
        high_roll = (s.get("roll_abs_gt_75_rate") or 0.0) > 0.25
        cause = []
        if low_speed_long:
            cause.append("long low-speed recovery")
        if high_roll:
            cause.append("large roll excursions")
        if fly_away:
            cause.append("sustained range increase")
        if vertical:
            cause.append("sustained vertical climb")
        if not cause:
            cause.append("no dominant trajectory anomaly")
        lines.append(
            f"- {aid}: pursuing_red={pursuing}, fly_away={fly_away}, "
            f"long_low_speed_recovery={low_speed_long}, vertical_climb={vertical}, "
            f"high_roll={high_roll}. Main issue: {', '.join(cause)}."
        )
    return lines


def analyze(diagnostics_dir: Path) -> dict[str, Any]:
    blue_rows = _read_csv(diagnostics_dir / "blue_behavior_timeseries.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in blue_rows:
        grouped.setdefault(str(row.get("agent_id", "unknown")), []).append(row)
    agent_summaries = {aid: _summary_for_agent(rows) for aid, rows in grouped.items()}
    summary_path = diagnostics_dir.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "diagnostics_dir": str(diagnostics_dir),
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "episode_summary": summary,
        "blue_agents": agent_summaries,
        "natural_summary": _natural_summary(agent_summaries),
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Blue Safe-Pursuit Track Analysis",
        "",
        f"diagnostics_dir: `{report['diagnostics_dir']}`",
        "",
        "## Natural Summary",
        *report["natural_summary"],
        "",
        "## Per-Agent Metrics",
    ]
    for aid, s in report["blue_agents"].items():
        lines.extend([
            f"### {aid}",
            f"- alive_final: {s.get('alive_final')}",
            f"- speed_min/p5/mean/max: {s.get('speed_min')}, {s.get('speed_p5')}, {s.get('speed_mean')}, {s.get('speed_max')}",
            f"- alt_min/p5/mean: {s.get('alt_min')}, {s.get('alt_p5')}, {s.get('alt_mean')}",
            f"- roll_abs_max/p95/gt75_rate: {s.get('roll_abs_max')}, {s.get('roll_abs_p95')}, {s.get('roll_abs_gt_75_rate')}",
            f"- nearest_range_start/end/min/delta: {s.get('nearest_enemy_range_start')}, {s.get('nearest_enemy_range_end')}, {s.get('nearest_enemy_range_min')}, {s.get('nearest_enemy_range_delta')}",
            f"- current_target_count/rate: {s.get('current_target_count')}, {s.get('current_target_rate')}",
            f"- low_speed_recovery_count/rate: {s.get('low_speed_recovery_count')}, {s.get('low_speed_recovery_rate')}",
            f"- reacquire/center/hold counts: {s.get('reacquire_last_seen_count')}, {s.get('center_cruise_count')}, {s.get('hold_heading_count')}",
            f"- old_vs_geo_heading mean/p50/p95/max: {s.get('old_geo_heading_diff_mean')}, {s.get('old_geo_heading_diff_p50')}, {s.get('old_geo_heading_diff_p95')}, {s.get('old_geo_heading_diff_max')}",
            f"- high_roll old_vs_geo mean/p95: {s.get('high_roll_old_geo_heading_diff_mean')}, {s.get('high_roll_old_geo_heading_diff_p95')}",
            f"- pre_low_speed old_vs_geo mean/p95: {s.get('pre_low_speed_old_geo_heading_diff_mean')}, {s.get('pre_low_speed_old_geo_heading_diff_p95')}",
            f"- long_low_speed_recovery_segments: {s.get('long_low_speed_recovery_segments')}",
            f"- fly_away_segments: {s.get('fly_away_segments')}",
            f"- vertical_climb_segments: {s.get('vertical_climb_segments')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    report = analyze(Path(args.diagnostics_dir))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out_md, report)
    print(f"output_md: {out_md}")
    print(f"output_json: {out_json}")


if __name__ == "__main__":
    main()
