"""Utilities for offline TAM-BRMA v1 + pure-HAPPO audits.

These helpers are intentionally read-only.  They summarize existing logs and
audit traces; they do not instantiate environments or modify training logic.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def explained_variance(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = [safe_float(v) for v in y_true]
    yp = [safe_float(v) for v in y_pred]
    if len(yt) != len(yp) or not yt:
        return 0.0
    mean_y = sum(yt) / len(yt)
    var_y = sum((v - mean_y) ** 2 for v in yt) / len(yt)
    if var_y <= 1e-12:
        return 0.0
    err = [a - b for a, b in zip(yt, yp)]
    mean_err = sum(err) / len(err)
    var_err = sum((v - mean_err) ** 2 for v in err) / len(err)
    return 1.0 - var_err / var_y


def action_clamp_stats(rows: list[dict]) -> dict:
    """Summarize action saturation fields already logged by training/eval."""
    if not rows:
        return {}
    mav = [safe_float(r.get("mav_action_saturation_rate")) for r in rows]
    uav = [safe_float(r.get("uav_action_saturation_rate")) for r in rows]
    return {
        "samples": len(rows),
        "mav_saturation_mean": sum(mav) / len(mav),
        "mav_saturation_max": max(mav),
        "uav_saturation_mean": sum(uav) / len(uav),
        "uav_saturation_max": max(uav),
        "mav_log_std_final": safe_float(rows[-1].get("action_log_std_mav_mean")),
        "uav_log_std_final": safe_float(rows[-1].get("action_log_std_uav_mean")),
    }


def phase_summary(rows: list[dict], phase_size: int = 100_000) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        step = int(safe_float(row.get("total_steps")))
        bucket = max(0, (step - 1) // phase_size)
        buckets[bucket].append(row)
    out = []
    for bucket in sorted(buckets):
        part = buckets[bucket]
        label = f"{bucket * phase_size}-{(bucket + 1) * phase_size}"
        out.append({
            "phase_steps": label,
            "rows": len(part),
            "avg_return_mean": mean_field(part, "avg_return"),
            "avg_return_final": safe_float(part[-1].get("avg_return")),
            "red_win_mean": mean_field(part, "red_win"),
            "blue_win_mean": mean_field(part, "blue_win"),
            "timeout_mean": mean_field(part, "timeout"),
            "mav_survival_mean": mean_field(part, "mav_survival"),
            "red_fire_mean": mean_field(part, "red_episode_missiles_fired_mean", "red_missiles_fired"),
            "red_hit_mean": mean_field(part, "red_episode_missile_hits_mean", "red_missile_hits"),
            "blue_fire_mean": mean_field(part, "blue_episode_missiles_fired_mean", "blue_missiles_fired"),
            "blue_hit_mean": mean_field(part, "blue_episode_missile_hits_mean", "blue_missile_hits"),
            "critic_loss_mean": mean_field(part, "critic_loss"),
            "approx_kl_mav_mean": mean_field(part, "approx_kl_mav"),
            "approx_kl_uav_mean": mean_field(part, "approx_kl_uav"),
            "mav_active_sample_count_mean": mean_field(part, "mav_active_sample_count"),
            "uav_active_sample_count_mean": mean_field(part, "uav_active_sample_count"),
        })
    return out


def mean_field(rows: list[dict], *names: str) -> float:
    for name in names:
        vals = [safe_float(r.get(name)) for r in rows if name in r]
        if vals:
            return sum(vals) / len(vals)
    return 0.0


def select_checkpoints_from_train_log(rows: list[dict]) -> list[dict]:
    """Select candidate rows without requiring checkpoint files."""
    if not rows:
        return []
    selectors = {
        "best_return": ("avg_return", max),
        "red_fire_peak": ("red_episode_missiles_fired_mean", max),
        "red_hit_peak": ("red_episode_missile_hits_mean", max),
        "mav_survival_peak": ("mav_survival", max),
        "red_win_peak": ("red_win", max),
        "latest": ("total_steps", max),
    }
    selected = []
    seen_steps = set()
    for name, (field, fn) in selectors.items():
        candidates = rows if field in rows[0] else rows
        if not candidates:
            continue
        if fn is max:
            row = max(candidates, key=lambda r: safe_float(r.get(field)))
        else:
            row = candidates[-1]
        step = int(safe_float(row.get("total_steps")))
        selected.append({
            "selector": name,
            "total_steps": step,
            "avg_return": safe_float(row.get("avg_return")),
            "red_win": safe_float(row.get("red_win")),
            "blue_win": safe_float(row.get("blue_win")),
            "timeout": safe_float(row.get("timeout")),
            "mav_survival": safe_float(row.get("mav_survival")),
            "red_fire": safe_float(row.get("red_episode_missiles_fired_mean"), safe_float(row.get("red_missiles_fired"))),
            "red_hit": safe_float(row.get("red_episode_missile_hits_mean"), safe_float(row.get("red_missile_hits"))),
            "blue_fire": safe_float(row.get("blue_episode_missiles_fired_mean"), safe_float(row.get("blue_missiles_fired"))),
            "blue_hit": safe_float(row.get("blue_episode_missile_hits_mean"), safe_float(row.get("blue_missile_hits"))),
            "duplicate_step": int(step in seen_steps),
        })
        seen_steps.add(step)
    return selected


def reward_component_stats(step_rows: list[dict]) -> list[dict]:
    if not step_rows:
        return []
    component_keys = [
        "tam_brma_v1_flight",
        "tam_brma_v1_uav_gate_sit",
        "tam_brma_v1_uav_event",
        "tam_brma_v1_mav_safe",
        "tam_brma_v1_mav_support",
        "tam_brma_v1_mav_aware",
        "tam_brma_v1_mav_event",
        "tam_brma_v1_team_terminal",
        "reward_total",
    ]
    out = []
    for role in sorted({r.get("role", "") for r in step_rows}):
        subset = [r for r in step_rows if r.get("role", "") == role]
        for key in component_keys:
            vals = [safe_float(r.get(key)) for r in subset if key in r]
            if vals:
                out.append({
                    "role": role,
                    "component": key,
                    "samples": len(vals),
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "positive_rate": sum(v > 0 for v in vals) / len(vals),
                    "negative_rate": sum(v < 0 for v in vals) / len(vals),
                })
    return out


def gate_mismatch_stats(gate_rows: list[dict]) -> dict:
    if not gate_rows:
        return {}
    n = len(gate_rows)
    reward_pos = [r for r in gate_rows if safe_float(r.get("reward_g_own")) > 0.01]
    geom_ok = [r for r in gate_rows if int(safe_float(r.get("launch_geometry_ok_3d"))) == 1]
    track_ok = [r for r in gate_rows if int(safe_float(r.get("has_track"))) == 1]
    boresight_ok = [r for r in gate_rows if int(safe_float(r.get("boresight_ok_3d"))) == 1]
    counter = Counter(r.get("mismatch_type", "unknown") for r in gate_rows)
    ao_diffs = [
        abs(safe_float(r.get("AO_2d_rad")) - safe_float(r.get("ATA_3d_rad")))
        for r in gate_rows
    ]
    ta_diffs = [
        abs(safe_float(r.get("TA_2d_rad")) - safe_float(r.get("TA_3d_rad")))
        for r in gate_rows
    ]
    return {
        "total_pairs": n,
        "track_ok_rate": len(track_ok) / n,
        "reward_g_own_positive_rate": len(reward_pos) / n,
        "launch_geometry_ok_3d_rate": len(geom_ok) / n,
        "boresight_ok_3d_rate": len(boresight_ok) / n,
        "reward_positive_given_geometry_ok": (
            sum(safe_float(r.get("reward_g_own")) > 0.01 for r in geom_ok) / max(len(geom_ok), 1)
        ),
        "geometry_ok_given_reward_positive": (
            sum(int(safe_float(r.get("launch_geometry_ok_3d"))) == 1 for r in reward_pos) / max(len(reward_pos), 1)
        ),
        "ao2d_ata3d_abs_diff_mean_rad": sum(ao_diffs) / len(ao_diffs),
        "ao2d_ata3d_abs_diff_p90_rad": percentile(ao_diffs, 90),
        "ta2d_ta3d_abs_diff_mean_rad": sum(ta_diffs) / len(ta_diffs),
        "ta2d_ta3d_abs_diff_p90_rad": percentile(ta_diffs, 90),
        **{f"mismatch_{k}": v for k, v in counter.items()},
    }


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def pearson_corr(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = [safe_float(v) for v in xs]
    y = [safe_float(v) for v in ys]
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (vx * vy) ** 0.5


def classify_launch_first_failed_gate(row: dict) -> str:
    gates = [
        ("track", "has_track"),
        ("range", "range_ok_3d"),
        ("ata", "ata_ok_3d"),
        ("ta", "ta_ok_3d"),
        ("boresight", "boresight_ok_3d"),
        ("geometry", "launch_geometry_ok_3d"),
    ]
    for label, key in gates:
        if int(safe_float(row.get(key))) != 1:
            return label
    if int(safe_float(row.get("lock_mature", 0))) != 1:
        return "lock_mature"
    if int(safe_float(row.get("actual_launch", 0))) != 1:
        return "launch"
    return "passed"


def summarize_first_failed_gate(rows: list[dict]) -> list[dict]:
    counts = Counter(classify_launch_first_failed_gate(r) for r in rows)
    total = max(len(rows), 1)
    return [
        {"first_failed_gate": gate, "count": count, "rate": count / total}
        for gate, count in sorted(counts.items())
    ]


def source_line_hits(path: Path, patterns: dict[str, str]) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for name, pattern in patterns.items():
        for idx, line in enumerate(lines, start=1):
            if pattern in line:
                out.append({
                    "file": str(path),
                    "line": idx,
                    "symbol": name,
                    "snippet": line.strip(),
                })
                break
    return out
