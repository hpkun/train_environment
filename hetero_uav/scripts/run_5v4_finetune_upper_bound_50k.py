"""Run the 5v4 fine-tune upper-bound from the best 3v2 curriculum checkpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_happo_reference.py"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_happo_3v2_reference_checkpoints.py"
DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_happo_ref_v0_f16_mav_surrogate.yaml"
)
DEFAULT_INIT = "outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt"
DEFAULT_OUTPUT = "outputs/happo_5v4_finetune_upper_bound_50k"


def _cmd_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    train_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--config",
        args.config,
        "--total-env-steps",
        str(args.total_env_steps),
        "--rollout-length",
        str(args.rollout_length),
        "--max-steps",
        str(args.max_steps),
        "--device",
        args.device,
        "--opponent-policy",
        args.opponent_policy,
        "--output-dir",
        args.output_dir,
        "--init-checkpoint",
        args.init_checkpoint,
        "--eval-during-training",
        "--eval-interval-steps",
        str(args.eval_interval_steps),
        "--train-eval-episodes",
        str(args.train_eval_episodes),
        "--eval-configs",
        args.config,
    ]
    eval_cmd = [
        sys.executable,
        "-u",
        str(EVAL_SCRIPT),
        "--output-dir",
        args.output_dir,
        "--episodes",
        str(args.eval_episodes),
        "--checkpoint-mode",
        "all",
        "--configs",
        "5v4_zero_shot",
        "--device",
        args.device,
        "--opponent-policy",
        args.opponent_policy,
    ]
    return train_cmd, eval_cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 50k 5v4 adaptation upper-bound; this is not zero-shot evaluation."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--eval-interval-steps", type=int, default=25_000)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_cmd, eval_cmd = build_commands(args)
    print("train command:")
    print(_cmd_text(train_cmd))
    print("eval command:")
    print(_cmd_text(eval_cmd))

    if args.dry_run:
        return 0

    subprocess.run(train_cmd, cwd=ROOT, check=True)
    subprocess.run(eval_cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
