"""Check whether rich logs cover BRMA-MAPPO/TAM-HAPPO plot needs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return set(next(reader, []))


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _status(input_dir: Path, required_files: list[str], required_columns: dict[str, list[str]] | None = None) -> str:
    missing_files = [name for name in required_files if not _exists(input_dir / name)]
    if len(missing_files) == len(required_files):
        return "missing"
    required_columns = required_columns or {}
    missing_cols = []
    for name, cols in required_columns.items():
        have = _columns(input_dir / name)
        missing_cols.extend([f"{name}:{col}" for col in cols if col not in have])
    if missing_files or missing_cols:
        return "partially_available"
    return "available"


def build_report(input_dir: Path) -> dict:
    has_eval_summary = _exists(input_dir / "eval_summary_metrics.csv")
    has_perturb = _exists(input_dir / "perturbation_eval_summary.csv")
    has_train = _exists(input_dir / "train_metrics.csv")
    return {
        "BRMA-MAPPO": {
            "reward_curves": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["avg_episode_return"]}),
            "win_rate_curves": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["red_win_rate", "blue_win_rate"]}),
            "scale_transfer": "requires_full_experiment" if has_eval_summary else "missing",
            "RWR": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["relative_win_ratio"]}),
            "KD": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["kill_death_ratio"]}),
            "training_efficiency": "available" if _exists(input_dir / "training_efficiency.json") else "missing",
            "ablation_reward_win_curves": "requires_multiple_runs" if has_train else "missing",
            "attention_heatmap_metrics": "not_implemented_by_current_algorithm",
        },
        "TAM-HAPPO": {
            "reward_curve": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["avg_episode_return"]}),
            "trajectory": _status(input_dir, ["aircraft_timeseries.csv"], {"aircraft_timeseries.csv": ["lon", "lat"]}),
            "attitude_curves": _status(input_dir, ["aircraft_timeseries.csv"], {"aircraft_timeseries.csv": ["altitude", "speed", "yaw", "pitch"]}),
            "heterogeneous_reward_components": _status(input_dir, ["reward_components.csv"]),
            "loss_policy_gradient": _status(input_dir, ["train_metrics.csv"], {"train_metrics.csv": ["critic_loss", "policy_gradient_norm"]}),
            "ablation_curves": "requires_multiple_runs" if has_train else "missing",
            "perturbation_generalization": "schema_only" if has_perturb else "missing",
        },
    }


def _write_md(path: Path, report: dict) -> None:
    lines = ["# Paper Plot Coverage Report", ""]
    for section, items in report.items():
        lines.append(f"## {section}")
        for name, status in items.items():
            lines.append(f"- {name}: `{status}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check paper plot coverage from rich logs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = _rel(args.input_dir)
    output_dir = _rel(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(input_dir)
    (output_dir / "plot_coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(output_dir / "plot_coverage_report.md", report)
    print(f"coverage_json: {output_dir / 'plot_coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
