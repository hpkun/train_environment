"""Run 100k HAPPO reference training on the easy-combat 3v2 task."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml"
ORACLE_INIT = "outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt"
HAPPO_BEST_INIT = "outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/best/model.pt"
OUTPUT_DIR = "outputs/happo_easy_combat_100k"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _init_checkpoint() -> str | None:
    if _rel(ORACLE_INIT).exists():
        return ORACLE_INIT
    if _rel(HAPPO_BEST_INIT).exists():
        return HAPPO_BEST_INIT
    return None


def _train_cmd(args) -> list[str]:
    cmd = [
        "python", "-u", "scripts/train_happo_reference.py",
        "--config", CONFIG,
        "--reward-mode", "happo_ref_v0",
        "--opponent-policy", "brma_rule",
        "--total-env-steps", str(args.total_env_steps),
        "--rollout-length", "256",
        "--max-steps", "1000",
        "--ppo-epochs", "10",
        "--entropy-coef", "0.02",
        "--actor-lr", "2e-4",
        "--critic-lr", "5e-4",
        "--eval-during-training",
        "--eval-interval-steps", "25000",
        "--train-eval-episodes", "2",
        "--eval-configs", CONFIG,
        "--output-dir", args.output_dir,
        "--device", "cuda",
    ]
    init = _init_checkpoint()
    if init:
        cmd.extend(["--init-checkpoint", init])
    return cmd


def _fast_eval_cmd(args) -> list[str]:
    return [
        "python", "scripts/evaluate_happo_3v2_reference_checkpoints.py",
        "--output-dir", args.output_dir,
        "--fast",
        "--checkpoint-mode", "all",
        "--configs", CONFIG,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run easy-combat HAPPO 100k training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--total-env-steps", type=int, default=100000)
    args = parser.parse_args()

    commands = [("train", _train_cmd(args)), ("fast_eval", _fast_eval_cmd(args))]
    for label, cmd in commands:
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
