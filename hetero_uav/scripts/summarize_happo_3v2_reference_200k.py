"""Summarize HAPPO 3v2 reference 200k train/eval logs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _trend(rows: list[dict], key: str) -> dict:
    vals = [_num(r, key) for r in rows if key in r]
    if not vals:
        return {"first": None, "last": None, "min": None, "max": None, "mean": None}
    return {
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _best_eval(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def score(r: dict) -> float:
        return (
            _num(r, "red_win_rate")
            + 0.1 * _num(r, "mav_survival_rate")
            + 0.05 * _num(r, "blue_dead_mean")
            + 0.05 * _num(r, "red_missile_hits_mean")
        )
    best = max(rows, key=score)
    out = dict(best)
    out["score"] = score(best)
    return out


def summarize(output_dir: Path) -> dict:
    if not output_dir.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")
    train = _read_csv(output_dir / "train_log.csv")
    eval_rows = _read_csv(output_dir / "eval_log.csv")
    latest_meta_path = output_dir / "latest" / "meta.json"
    best_meta_path = output_dir / "best" / "meta.json"
    latest_meta = json.loads(latest_meta_path.read_text(encoding="utf-8")) if latest_meta_path.exists() else {}
    best_meta = json.loads(best_meta_path.read_text(encoding="utf-8")) if best_meta_path.exists() else {}
    last_train = train[-1] if train else {}
    best_eval = _best_eval(eval_rows)
    judgment = {
        "still_timeout_survival": _num(last_train, "timeout") > 0.8 and _num(last_train, "red_win") > 0.0,
        "blue_death_seen_in_train": any(_num(r, "blue_alive_final") < 2.0 for r in train),
        "red_missile_hit_seen_in_train": any(_num(r, "missile_hits") > 0 for r in train),
        "non_timeout_outcome_seen_in_train": any(_num(r, "timeout") < 1.0 for r in train),
        "best_obviously_better_than_latest": bool(best_meta and best_meta.get("best_score", 0) > 0),
        "strategy_collapse_possible": _num(last_train, "blue_win") > 0.9 and _num(last_train, "red_alive_final") <= 0.1,
    }
    return {
        "output_dir": str(output_dir),
        "train_log_exists": bool(train),
        "eval_log_exists": bool(eval_rows),
        "latest_model_exists": (output_dir / "latest" / "model.pt").exists(),
        "best_model_exists": (output_dir / "best" / "model.pt").exists(),
        "latest_meta": latest_meta,
        "best_meta": best_meta,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "last_train": last_train,
        "train_trends": {
            key: _trend(train, key)
            for key in [
                "avg_return", "red_win", "blue_win", "draw", "timeout",
                "mav_survival", "red_alive_final", "blue_alive_final",
                "red_missiles_fired", "blue_missiles_fired", "missile_hits",
                "entropy_mav", "entropy_uav", "actor_loss_mav",
                "actor_loss_uav", "critic_loss", "mav_action_saturation_rate",
                "uav_action_saturation_rate",
            ]
        },
        "best_eval": best_eval,
        "eval_by_config_latest_step": [
            r for r in eval_rows if r.get("total_steps") == (eval_rows[-1].get("total_steps") if eval_rows else None)
        ],
        "judgment": judgment,
    }


def _write_md(path: Path, data: dict) -> None:
    j = data["judgment"]
    lines = [
        "# HAPPO 3v2 Reference 200k Training Summary",
        "",
        f"- train_rows: {data['train_rows']}",
        f"- eval_rows: {data['eval_rows']}",
        f"- latest_model_exists: {data['latest_model_exists']}",
        f"- best_model_exists: {data['best_model_exists']}",
        "",
        "## Last Train Row",
        "```json",
        json.dumps(data["last_train"], indent=2),
        "```",
        "",
        "## Best Eval Row",
        "```json",
        json.dumps(data["best_eval"], indent=2),
        "```",
        "",
        "## Judgment",
        *(f"- {k}: {v}" for k, v in j.items()),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize HAPPO 3v2 reference 200k")
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    try:
        out_dir = _rel(args.output_dir)
        data = summarize(out_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_json = _rel(args.output_json) if args.output_json else out_dir / "happo_3v2_training_summary.json"
    out_md = _rel(args.output_md) if args.output_md else out_dir / "happo_3v2_training_summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_md(out_md, data)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    print(f"latest_model_exists: {data['latest_model_exists']}")
    print(f"best_model_exists: {data['best_model_exists']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
