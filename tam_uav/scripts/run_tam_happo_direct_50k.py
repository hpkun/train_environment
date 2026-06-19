"""Launch the formal 50k TAM-HAPPO diagnostic run."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([
        sys.executable, "-u", str(ROOT / "scripts" / "train_tam_happo_direct.py"),
        "--config", "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "--output-dir", "outputs/tam_happo_direct_f22_50k_smoke",
        "--total-env-steps", "50000", "--rollout-length", "256",
        "--num-envs", "1", "--max-steps", "1000", "--device", "cuda",
        "--policy-arch", "brma_recurrent_masked",
        "--opponent-policy", "tam_direct_fsm", "--reward-mode", "happo_ref_v0",
        "--eval-during-training", "--eval-at-start", "--eval-interval-steps", "25000",
        "--train-eval-episodes", "3", "--eval-configs",
        "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml",
        "--enable-rich-logging",
        "--rich-log-dir", "outputs/tam_happo_direct_f22_50k_smoke/rich_logs",
        "--save-eval-checkpoints", "--eval-checkpoint-metric", "combined",
        "--keep-eval-checkpoints", "10",
        "--heartbeat-log", "outputs/tam_happo_direct_f22_50k_smoke/heartbeat.log",
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
