from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_checkpoint_selection import best_metric_name


def _load_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_from_meta(meta: dict) -> dict:
    metrics = meta.get("metrics", {})
    scores = meta.get("scores", {})
    row = {
        "step": meta.get("step"),
        "iteration": meta.get("iteration"),
        "score_3v2": scores.get("score_3v2", 0.0),
        "score_5v4": scores.get("score_5v4", 0.0),
        "score_combined": scores.get("score_combined", 0.0),
    }
    for label in ("3v2", "5v4"):
        record = metrics.get(label, {})
        for key in (
            "red_win_rate",
            "blue_win_rate",
            "draw_rate",
            "timeout_rate",
            "mav_survival_rate",
            "red_missile_hits_mean",
            "blue_dead_mean",
        ):
            row[f"{label}_{key}"] = record.get(key, "")
    return row


def _best(rows: list[dict], name: str) -> dict | None:
    if not rows:
        return None
    metric = best_metric_name(name)
    return max(rows, key=lambda row: _as_float(row.get(metric)))


def summarize(output_dir: Path) -> tuple[Path, Path]:
    eval_dir = output_dir / "eval_checkpoints"
    metas = sorted(eval_dir.glob("step_*/meta.json"))
    rows = []
    for meta_path in metas:
        meta = _load_meta(meta_path)
        if meta:
            rows.append(_row_from_meta(meta))

    csv_path = output_dir / "eval_checkpoint_summary.csv"
    md_path = output_dir / "eval_checkpoint_summary.md"
    fieldnames = [
        "step",
        "iteration",
        "score_3v2",
        "score_5v4",
        "score_combined",
        "3v2_red_win_rate",
        "3v2_blue_win_rate",
        "3v2_draw_rate",
        "3v2_timeout_rate",
        "3v2_mav_survival_rate",
        "3v2_red_missile_hits_mean",
        "3v2_blue_dead_mean",
        "5v4_red_win_rate",
        "5v4_blue_win_rate",
        "5v4_draw_rate",
        "5v4_timeout_rate",
        "5v4_mav_survival_rate",
        "5v4_red_missile_hits_mean",
        "5v4_blue_dead_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    latest = _load_meta(output_dir / "latest" / "meta.json") or {}
    best_lines = []
    for name in ("best_3v2", "best_5v4", "best_combined"):
        row = _best(rows, name)
        if row is None:
            best_lines.append(f"- {name}: not available")
        else:
            metric = best_metric_name(name)
            best_lines.append(f"- {name}: step {row.get('step')} ({metric}={_as_float(row.get(metric)):.3f})")
    latest_step = latest.get("total_env_steps_actual", "unknown")
    last_row = rows[-1] if rows else None
    regression = "unknown"
    if rows and _best(rows, "best_combined") is not None:
        regression = str(_best(rows, "best_combined").get("step") != last_row.get("step"))

    md = [
        "# Eval Checkpoint Summary",
        "",
        f"- output_dir: `{output_dir}`",
        f"- eval checkpoints: {len(rows)}",
        f"- latest step: {latest_step}",
        "",
        "## Best Checkpoints",
        *best_lines,
        "",
        "## Latest vs Best",
        f"- last eval step: {last_row.get('step') if last_row else 'not available'}",
        f"- best_combined differs from last eval: {regression}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    csv_path, md_path = summarize(ROOT / args.output_dir)
    print(f"output_csv: {csv_path}", flush=True)
    print(f"output_md: {md_path}", flush=True)


if __name__ == "__main__":
    main()

