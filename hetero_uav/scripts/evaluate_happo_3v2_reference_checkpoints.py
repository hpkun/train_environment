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
ZERO_SHOT_5V4_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _checkpoint_paths(exp_dir: Path, mode: str) -> list[tuple[str, Path]]:
    names = ["best", "latest"] if mode == "all" else [mode.replace("_only", "")]
    return [(name, exp_dir / name / "model.pt") for name in names]


def _load_meta(model: Path) -> dict:
    meta = model.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _default_configs_for(exp_dir: Path) -> list[str]:
    for name in ("best", "latest"):
        cfg = _load_meta(exp_dir / name / "model.pt").get("config")
        if cfg:
            return [cfg, ZERO_SHOT_5V4_CONFIG]
    return DEFAULT_CONFIGS


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


def _write_final_decision(exp_dir: Path, records: list[dict]) -> None:
    def find(checkpoint: str, needle: str) -> dict:
        for record in records:
            if record.get("checkpoint") == checkpoint and needle in record.get("config", ""):
                return record
        return {}

    best3 = find("best", "3v2")
    best5 = find("best", "5v4")
    latest3 = find("latest", "3v2")
    latest5 = find("latest", "5v4")
    criteria = {
        "3v2_red_missiles_fired_mean_gt_0_3": float(best3.get("red_missiles_fired_mean", 0.0) or 0.0) > 0.3,
        "3v2_hit_or_blue_dead_gt_0_1": (
            float(best3.get("red_missile_hits_mean", 0.0) or 0.0) > 0.1
            or float(best3.get("blue_dead_mean", 0.0) or 0.0) > 0.1
        ),
        "3v2_mav_survival_rate_ge_0_3": float(best3.get("mav_survival_rate", 0.0) or 0.0) >= 0.3,
        "3v2_blue_win_rate_lt_0_9": float(best3.get("blue_win_rate", 1.0) or 1.0) < 0.9,
        "5v4_red_missiles_fired_mean_gt_0": float(best5.get("red_missiles_fired_mean", 0.0) or 0.0) > 0.0,
        "5v4_not_complete_collapse": float(best5.get("blue_win_rate", 1.0) or 1.0) < 1.0,
    }
    usable = bool(all(criteria.values()))
    decision = {
        "usable_as_combat_pilot": usable,
        "recommend_1m": usable,
        "next_step": (
            "run 1M oracle-pretrain fine-tune"
            if usable else
            "build easy combat task by shortening initial distance and adjusting initial heading"
        ),
        "criteria": criteria,
        "best_3v2": best3,
        "latest_3v2": latest3,
        "best_5v4": best5,
        "latest_5v4": latest5,
    }
    out_json = exp_dir / "final_decision.json"
    out_md = exp_dir / "final_decision.md"
    out_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# Oracle-Pretrain Fine-Tune Final Decision",
        "",
        f"- usable_as_combat_pilot: {usable}",
        f"- recommend_1m: {usable}",
        f"- next_step: {decision['next_step']}",
        "",
        "## Criteria",
    ]
    lines.extend(f"- {key}: {value}" for key, value in criteria.items())
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HAPPO 3v2 reference checkpoints")
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--output-dir", dest="experiment_dir", default=argparse.SUPPRESS,
                        help="Alias for --experiment-dir")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--checkpoint-mode", choices=["best_only", "latest_only", "all"], default="all")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Quick screening: 20 episodes and 3v2 seen config only.")
    args = parser.parse_args()

    if args.fast:
        args.episodes = 20
    exp_dir = _rel(args.experiment_dir)
    if not exp_dir.exists():
        print(f"experiment directory does not exist: {exp_dir}", file=sys.stderr)
        return 2
    out_dir = exp_dir / "checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    try:
        if args.configs == DEFAULT_CONFIGS:
            args.configs = _default_configs_for(exp_dir)
        if args.fast:
            args.configs = [cfg for cfg in args.configs if "3v2" in cfg] or args.configs[:1]
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
    _write_final_decision(exp_dir, records)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    print(f"final_decision_json: {exp_dir / 'final_decision.json'}")
    print(f"records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
