"""Run or print a 4-env stability debug command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir
    cmd = [
        sys.executable,
        "scripts/train_happo_reference.py",
        "--config",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
        "--output-dir",
        output_dir,
        "--total-env-steps",
        str(args.total_env_steps),
        "--rollout-length",
        "256",
        "--num-envs",
        str(args.num_envs),
        "--max-steps",
        str(args.max_steps),
        "--device",
        "cuda",
        "--init-checkpoint",
        "outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt",
        "--uav-imitation-dataset",
        "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz",
        "--uav-imitation-coef",
        "0.03",
        "--uav-imitation-until-steps",
        "100000",
        "--heartbeat-log",
        f"{output_dir}/heartbeat.log",
        "--heartbeat-every-steps",
        "50",
        "--heartbeat-stall-timeout-sec",
        str(args.heartbeat_stall_timeout_sec),
        "--debug-rollout-heartbeat",
    ]
    if args.exit_on_heartbeat_stall:
        cmd.append("--exit-on-heartbeat-stall")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--total-env-steps", type=int, default=500000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/debug_4env_max1000_500k")
    parser.add_argument("--heartbeat-stall-timeout-sec", type=float, default=300)
    parser.add_argument("--exit-on-heartbeat-stall", action="store_true")
    parser.add_argument("--debug-rollout-heartbeat", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
