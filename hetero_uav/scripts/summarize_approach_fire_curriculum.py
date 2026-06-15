"""Summarize approach-and-fire curriculum diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FIELDS = [
    "experiment",
    "policy_arch",
    "total_env_steps",
    "red_missiles_fired",
    "missile_hits",
    "blue_dead_mean",
    "range_ok_rate",
    "ao_ok_rate",
    "ta_ok_rate",
    "lock_ready_rate",
    "launch_allowed_rate",
    "dominant_block_reason",
    "action_saturation_rate",
    "improved_attack_chain",
]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def summarize_run(name: str, output_dir: Path, policy_arch: str, total_env_steps: int) -> dict[str, Any]:
    diag = _load_json(output_dir / "launch_diagnostics_3v2" / "summary.json")
    red_fired = _num(diag, "red_missiles_fired")
    hits = _num(diag, "missile_hits")
    improved = (
        _num(diag, "range_ok_rate") > 0.0
        and _num(diag, "ao_ok_rate") > 0.0
        and _num(diag, "ta_ok_rate") > 0.0
        and red_fired > 0.0
        and hits > 0.0
    )
    return {
        "experiment": name,
        "policy_arch": policy_arch,
        "total_env_steps": total_env_steps,
        "red_missiles_fired": red_fired,
        "missile_hits": hits,
        "blue_dead_mean": _num(diag, "blue_dead_mean"),
        "range_ok_rate": _num(diag, "range_ok_rate"),
        "ao_ok_rate": _num(diag, "ao_ok_rate"),
        "ta_ok_rate": _num(diag, "ta_ok_rate"),
        "lock_ready_rate": _num(diag, "lock_ready_rate"),
        "launch_allowed_rate": _num(diag, "launch_allowed_rate"),
        "dominant_block_reason": diag.get("dominant_block_reason", ""),
        "action_saturation_rate": _num(diag, "action_saturation_rate"),
        "improved_attack_chain": improved,
    }


def write_summary(rows: list[dict[str, Any]], output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    lines = [
        "# Approach-and-Fire Curriculum Summary",
        "",
        "| experiment | arch | steps | red fired | hits | blue dead | range ok | AO ok | TA ok | block | improved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {policy_arch} | {total_env_steps} | {red_missiles_fired} | "
            "{missile_hits} | {blue_dead_mean} | {range_ok_rate} | {ao_ok_rate} | "
            "{ta_ok_rate} | {dominant_block_reason} | {improved_attack_chain} |".format(**row)
        )
    lines.extend([
        "",
        "## Decision Rule",
        "",
        "Continue to a longer curriculum only if range/AO/TA rates improve and eval produces non-zero fire and hits.",
        "If out_of_range remains dominant, inspect imitation data/action decoding before adding GRU or masks.",
    ])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize approach-fire curriculum diagnostics")
    parser.add_argument("--base-dir", default="outputs/approach_fire_curriculum_50k")
    parser.add_argument("--flat-dir", default=None)
    parser.add_argument("--entity-dir", default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output-csv", default="outputs/approach_fire_curriculum_summary.csv")
    parser.add_argument("--output-md", default="outputs/approach_fire_curriculum_summary.md")
    args = parser.parse_args()

    base = _rel(args.base_dir)
    flat_dir = _rel(args.flat_dir) if args.flat_dir else base / "flat_easy_imitation"
    entity_dir = _rel(args.entity_dir) if args.entity_dir else base / "entity_easy_imitation"
    rows = [
        summarize_run("flat_easy_imitation", flat_dir, "flat", args.steps),
        summarize_run("entity_easy_imitation", entity_dir, "entity_attention", args.steps),
    ]
    write_summary(rows, _rel(args.output_csv), _rel(args.output_md))
    print(f"output_csv: {_rel(args.output_csv)}", flush=True)
    print(f"output_md: {_rel(args.output_md)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
