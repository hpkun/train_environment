from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = (
    "range_m",
    "AO_deg",
    "TA_deg",
    "closing_speed_mps",
    "altitude_diff_m",
)


def _to_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _is_hit(record: dict) -> bool:
    return (
        str(record.get("is_success", "")).lower() == "true"
        or record.get("termination_reason") == "hit"
    )


def _values(records: list[dict], field: str, abs_value: bool = False) -> list[float]:
    vals = []
    for record in records:
        value = _to_float(record.get(field))
        if math.isnan(value):
            continue
        vals.append(abs(value) if abs_value else value)
    return vals


def percentile_summary(values: list[float]) -> dict:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "mean": float(np.mean(clean)),
        "p25": float(np.percentile(clean, 25)),
        "p50": float(np.percentile(clean, 50)),
        "p75": float(np.percentile(clean, 75)),
    }


def summarize_group(records: list[dict]) -> dict:
    out = {
        "launch_count": len(records),
        "hit_count": sum(1 for r in records if _is_hit(r)),
    }
    out["hit_rate"] = out["hit_count"] / out["launch_count"] if out["launch_count"] else 0.0
    out["range"] = percentile_summary(_values(records, "range_m"))
    out["ao_deg"] = percentile_summary(_values(records, "AO_deg"))
    out["ta_deg"] = percentile_summary(_values(records, "TA_deg"))
    out["closing_speed"] = percentile_summary(_values(records, "closing_speed_mps"))
    out["altitude_diff_abs_mean"] = percentile_summary(
        _values(records, "altitude_diff_m", abs_value=True))["mean"]
    return out


def read_launch_quality_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _table_row(label: str, summary: dict) -> str:
    return (
        f"| {label} | {summary['launch_count']} | {summary['hit_count']} | "
        f"{_fmt(summary['hit_rate'])} | {_fmt(summary['range']['mean'])} | "
        f"{_fmt(summary['range']['p25'])} | {_fmt(summary['range']['p50'])} | "
        f"{_fmt(summary['range']['p75'])} | {_fmt(summary['ao_deg']['mean'])} | "
        f"{_fmt(summary['ao_deg']['p50'])} | {_fmt(summary['ta_deg']['mean'])} | "
        f"{_fmt(summary['ta_deg']['p50'])} | {_fmt(summary['closing_speed']['mean'])} | "
        f"{_fmt(summary['altitude_diff_abs_mean'])} |"
    )


def _diagnose(red: list[dict], blue: list[dict]) -> list[str]:
    red_hits = [r for r in red if _is_hit(r)]
    red_misses = [r for r in red if not _is_hit(r)]
    blue_hits = [r for r in blue if _is_hit(r)]
    blue_misses = [r for r in blue if not _is_hit(r)]
    red_hit_s = summarize_group(red_hits)
    red_miss_s = summarize_group(red_misses)
    blue_s = summarize_group(blue)
    red_s = summarize_group(red)
    red_hit_ao_edge = percentile_summary([abs(v - 45.0) for v in _values(red_hits, "AO_deg")])["mean"]
    red_miss_ao_edge = percentile_summary([abs(v - 45.0) for v in _values(red_misses, "AO_deg")])["mean"]
    red_hit_ta_edge = percentile_summary([abs(v - 90.0) for v in _values(red_hits, "TA_deg")])["mean"]
    red_miss_ta_edge = percentile_summary([abs(v - 90.0) for v in _values(red_misses, "TA_deg")])["mean"]

    lines = []
    if red_miss_s["range"]["mean"] > red_hit_s["range"]["mean"]:
        lines.append("- Red misses are launched farther than Red hits on average.")
    else:
        lines.append("- Red misses are not farther than Red hits on average.")
    if red_miss_s["ao_deg"]["mean"] > red_hit_s["ao_deg"]["mean"]:
        lines.append("- Red misses have worse AO than Red hits on average.")
    else:
        lines.append("- Red misses do not have worse AO than Red hits on average.")
    if red_miss_ao_edge < red_hit_ao_edge:
        lines.append("- Red misses are closer to the 45 deg AO launch boundary than Red hits.")
    else:
        lines.append("- Red misses are not closer to the 45 deg AO launch boundary than Red hits.")
    if red_miss_s["ta_deg"]["mean"] < red_hit_s["ta_deg"]["mean"]:
        lines.append("- Red misses have weaker rear-aspect TA than Red hits on average.")
    else:
        lines.append("- Red misses do not show weaker TA than Red hits on average.")
    if red_miss_ta_edge < red_hit_ta_edge:
        lines.append("- Red misses are closer to the 90 deg TA launch boundary than Red hits.")
    else:
        lines.append("- Red misses are not closer to the 90 deg TA launch boundary than Red hits.")
    if red_miss_s["closing_speed"]["mean"] < red_hit_s["closing_speed"]["mean"]:
        lines.append("- Red misses have worse closing speed than Red hits on average.")
    else:
        lines.append("- Red misses do not have worse closing speed than Red hits on average.")
    if blue_s["hit_rate"] > red_s["hit_rate"]:
        lines.append("- Blue launch quality is better by realized hit rate.")
    else:
        lines.append("- Blue launch quality is not better by realized hit rate.")
    if not blue_hits and not blue_misses:
        lines.append("- Blue has no launch-quality rows in this file.")
    return lines


def build_markdown(rows: list[dict], source: str) -> str:
    red = [r for r in rows if r.get("team") == "red"]
    blue = [r for r in rows if r.get("team") == "blue"]
    groups = {
        "Red all": summarize_group(red),
        "Blue all": summarize_group(blue),
        "Red hit": summarize_group([r for r in red if _is_hit(r)]),
        "Red miss": summarize_group([r for r in red if not _is_hit(r)]),
        "Blue hit": summarize_group([r for r in blue if _is_hit(r)]),
        "Blue miss": summarize_group([r for r in blue if not _is_hit(r)]),
    }
    lines = [
        "# Launch quality summary",
        "",
        f"Source: `{source}`",
        "",
        "| Group | Launches | Hits | Hit rate | Range mean | Range p25 | Range p50 | Range p75 | AO mean | AO p50 | TA mean | TA p50 | Closing mean | Abs altitude diff mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in groups.items():
        lines.append(_table_row(label, summary))
    lines.extend(["", "## Diagnosis", ""])
    lines.extend(_diagnose(red, blue))
    lines.append("")
    return "\n".join(lines)


def write_summary(launch_quality_file: str | Path, output: str | Path) -> None:
    rows = read_launch_quality_csv(launch_quality_file)
    markdown = build_markdown(rows, str(launch_quality_file))
    output = Path(output)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-quality-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_summary(args.launch_quality_file, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
