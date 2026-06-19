"""Summarize TAM training and evaluation CSV outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _number(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def _series(rows: list[dict], key: str) -> list[float]:
    return [value for row in rows if (value := _number(row, key)) is not None]


def _stats(rows: list[dict], key: str) -> dict:
    values = _series(rows, key)
    if not values:
        return {"count": 0, "first": None, "last": None, "mean": None, "delta": None}
    return {
        "count": len(values), "first": values[0], "last": values[-1],
        "mean": sum(values) / len(values), "delta": values[-1] - values[0],
    }


def summarize_rows(rows: list[dict], eval_rows: list[dict] | None = None) -> dict:
    eval_rows = eval_rows or []
    return {
        "avg_return": _stats(rows, "avg_return"),
        "outcomes": {
            "red_win_mean": _stats(rows, "red_win")["mean"],
            "blue_win_mean": _stats(rows, "blue_win")["mean"],
            "timeout_mean": _stats(rows, "timeout")["mean"],
        },
        "self_control": {
            key: _stats(rows, key) for key in (
                "mav_active_sample_count", "uav_active_sample_count",
                "mav_action_saturation_rate", "uav_action_saturation_rate",
                "entropy", "entropy_mav", "entropy_uav",
                "action_log_std_mav_mean", "action_log_std_uav_mean",
                "mav_death_time", "mav_death_reason",
            )
        },
        "weapon_chain": {
            key: _stats(rows, key) for key in (
                "red_missiles_fired", "red_missile_hits", "red_missile_hit_rate",
                "red_missile_low_speed", "shooter_speed",
            )
        },
        "evaluation": {
            "3v2": [row for row in eval_rows if "3v2" in row.get("config", "")],
            "5v4": [row for row in eval_rows if "5v4" in row.get("config", "")],
        },
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    train_rows = _read_csv(output_dir / "train_log.csv")
    eval_rows = _read_csv(output_dir / "eval_log.csv")
    summary = summarize_rows(train_rows, eval_rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# TAM Training Curve Analysis", "",
        "2k and 50k runs are diagnostic: evaluate self-control first, weapon-chain second, combat effectiveness last.", "",
        f"- avg_return: `{summary['avg_return']}`",
        f"- outcomes: `{summary['outcomes']}`",
        f"- self_control: `{summary['self_control']}`",
        f"- weapon_chain: `{summary['weapon_chain']}`",
        f"- 3v2 eval records: {len(summary['evaluation']['3v2'])}",
        f"- 5v4 eval records: {len(summary['evaluation']['5v4'])}",
    ]
    (output_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
