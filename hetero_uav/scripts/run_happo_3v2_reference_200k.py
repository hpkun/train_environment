"""Configured 200k HAPPO reference validation runner.

Do not run this as part of smoke tests.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        "python", "-u", "scripts/train_happo_reference.py",
        "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
        "--reward-mode", "happo_ref_v0",
        "--opponent-policy", "brma_rule",
        "--total-env-steps", "200000",
        "--rollout-length", "256",
        "--ppo-epochs", "10",
        "--entropy-coef", "0.02",
        "--actor-lr", "2e-4",
        "--critic-lr", "5e-4",
        "--eval-during-training",
        "--eval-interval-steps", "25000",
        "--train-eval-episodes", "5",
        "--output-dir", "outputs/happo_3v2_reference_200k",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
