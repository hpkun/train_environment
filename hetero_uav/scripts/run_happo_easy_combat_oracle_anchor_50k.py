"""Run 50k easy-combat HAPPO with a UAV oracle imitation anchor."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml"
ORACLE_INIT = "outputs/oracle_pretrain/uav_actor_oracle_pretrained_easy_combat/model.pt"
ORACLE_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_easy_combat.npz"
OUTPUT_DIR = "outputs/happo_easy_combat_oracle_anchor_50k"


def _train_cmd(args) -> list[str]:
    return [
        sys.executable, "-u", "scripts/train_happo_reference.py",
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
        "--init-checkpoint", ORACLE_INIT,
        "--uav-imitation-dataset", ORACLE_DATASET,
        "--uav-imitation-coef", "0.1",
        "--uav-imitation-until-steps", str(args.total_env_steps),
        "--uav-imitation-batch-size", "1024",
        "--output-dir", args.output_dir,
        "--device", "cuda",
    ]


def _fast_eval_cmd(args) -> list[str]:
    return [
        sys.executable, "scripts/evaluate_happo_3v2_reference_checkpoints.py",
        "--output-dir", args.output_dir,
        "--fast",
        "--checkpoint-mode", "all",
        "--configs", CONFIG,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run easy-combat oracle-anchor HAPPO 50k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--total-env-steps", type=int, default=50000)
    args = parser.parse_args()

    commands = [("train", _train_cmd(args)), ("fast_eval", _fast_eval_cmd(args))]
    for label, cmd in commands:
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
