"""Configured 1M HAPPO reference validation runner with F-16 MAV surrogate."""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def main() -> None:
    cmd = [
        "python", "-u", "scripts/train_happo_reference.py",
        "--config", CONFIG,
        "--reward-mode", "happo_ref_v0",
        "--opponent-policy", "brma_rule",
        "--total-env-steps", "1000000",
        "--rollout-length", "512",
        "--ppo-epochs", "10",
        "--entropy-coef", "0.02",
        "--actor-lr", "2e-4",
        "--critic-lr", "5e-4",
        "--eval-during-training",
        "--eval-interval-steps", "100000",
        "--train-eval-episodes", "5",
        "--eval-configs", CONFIG,
        "--output-dir", "outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
