"""Run flat/entity approach-and-fire easy-geometry imitation experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_approach_fire_easy_f16_mav_surrogate.yaml"
ORACLE_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_base(path: str | Path, *, dry_run: bool) -> Path:
    base = _rel(path)
    if dry_run:
        return base
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(
            f"output directory already exists and is non-empty: {base}. "
            "Choose a new --output-dir to avoid overwriting existing results."
        )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _train_cmd(policy_arch: str, output_dir: Path, args) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/train_happo_reference.py",
        "--config",
        CONFIG,
        "--policy-arch",
        policy_arch,
        "--reward-mode",
        "happo_ref_v0",
        "--opponent-policy",
        args.opponent_policy,
        "--total-env-steps",
        str(args.total_env_steps),
        "--rollout-length",
        str(args.rollout_length),
        "--num-envs",
        "1",
        "--max-steps",
        "1000",
        "--ppo-epochs",
        "10",
        "--entropy-coef",
        "0.02",
        "--actor-lr",
        "2e-4",
        "--critic-lr",
        "5e-4",
        "--eval-during-training",
        "--eval-interval-steps",
        str(args.eval_interval_steps),
        "--train-eval-episodes",
        str(args.train_eval_episodes),
        "--eval-configs",
        CONFIG,
        "--uav-imitation-dataset",
        ORACLE_DATASET,
        "--uav-imitation-coef",
        str(args.imitation_coef),
        "--uav-imitation-until-steps",
        str(args.total_env_steps),
        "--uav-imitation-batch-size",
        "1024",
        "--enable-rich-logging",
        "--rich-log-dir",
        str(output_dir / "rich_logs"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]


def _diag_cmd(output_dir: Path, label: str, args) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_policy_launch_diagnostics.py",
        "--output-dir",
        str(output_dir),
        "--checkpoint-name",
        "best",
        "--episodes",
        str(args.diagnostic_episodes),
        "--scenario",
        "3v2",
        "--config",
        CONFIG,
        "--diagnostic-output-dir",
        str(output_dir / "launch_diagnostics_3v2"),
        "--label",
        label,
        "--device",
        args.device,
        "--opponent-policy",
        args.opponent_policy,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run approach-fire curriculum short experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/approach_fire_curriculum_50k")
    parser.add_argument("--total-env-steps", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--eval-interval-steps", type=int, default=25000)
    parser.add_argument("--train-eval-episodes", type=int, default=5)
    parser.add_argument("--diagnostic-episodes", type=int, default=5)
    parser.add_argument("--imitation-coef", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opponent-policy", default="brma_rule")
    args = parser.parse_args()

    base = _safe_base(args.output_dir, dry_run=args.dry_run)
    jobs = [
        ("flat", base / "flat_easy_imitation", "flat_easy_imitation"),
        ("entity_attention", base / "entity_easy_imitation", "entity_easy_imitation"),
    ]
    for policy_arch, output_dir, label in jobs:
        _run(_train_cmd(policy_arch, output_dir, args), dry_run=args.dry_run)
        _run(_diag_cmd(output_dir, label, args), dry_run=args.dry_run)
    _run([
        sys.executable,
        "scripts/summarize_approach_fire_curriculum.py",
        "--base-dir",
        str(base),
        "--steps",
        str(args.total_env_steps),
    ], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
