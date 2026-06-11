"""Evaluate HAPPO 3v2 reference checkpoints best/latest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _checkpoint_paths(exp_dir: Path, mode: str) -> list[tuple[str, Path]]:
    names = ["best", "latest"] if mode == "all" else [mode.replace("_only", "")]
    return [(name, exp_dir / name / "model.pt") for name in names]


def _run_eval(name: str, model: Path, args, out_dir: Path) -> list[dict]:
    if not model.exists():
        raise FileNotFoundError(f"checkpoint not found: {model}")
    tmp_json = out_dir / f"{name}_eval_raw.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "eval_happo_reference.py"),
        "--model",
        str(model),
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--opponent-policy",
        args.opponent_policy,
        "--summary-json",
        str(tmp_json.relative_to(ROOT)),
        "--configs",
        *args.configs,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    records = json.loads(tmp_json.read_text(encoding="utf-8"))
    for record in records:
        record["checkpoint"] = name
        record["model_path"] = str(model)
    return records


def _write_md(path: Path, records: list[dict]) -> None:
    lines = ["# HAPPO Checkpoint Evaluation", ""]
    for record in records:
        lines.extend([
            f"## {record['checkpoint']} - {record['config']}",
            f"- avg_return: {record.get('avg_return')}",
            f"- red_win_rate: {record.get('red_win_rate')}",
            f"- blue_win_rate: {record.get('blue_win_rate')}",
            f"- timeout_rate: {record.get('timeout_rate')}",
            f"- mav_survival_rate: {record.get('mav_survival_rate')}",
            f"- blue_dead_mean: {record.get('blue_dead_mean')}",
            f"- red_missile_hits_mean: {record.get('red_missile_hits_mean')}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HAPPO 3v2 reference checkpoints")
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--checkpoint-mode", choices=["best_only", "latest_only", "all"], default="all")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    exp_dir = _rel(args.experiment_dir)
    if not exp_dir.exists():
        print(f"experiment directory does not exist: {exp_dir}", file=sys.stderr)
        return 2
    out_dir = exp_dir / "checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    try:
        for name, model in _checkpoint_paths(exp_dir, args.checkpoint_mode):
            records.extend(_run_eval(name, model, args, out_dir))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_json = _rel(args.output_json) if args.output_json else out_dir / "happo_3v2_checkpoint_eval.json"
    out_md = _rel(args.output_md) if args.output_md else out_dir / "happo_3v2_checkpoint_eval.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _write_md(out_md, records)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    print(f"records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
