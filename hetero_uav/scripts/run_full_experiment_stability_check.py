"""Run a short single-env stability check for the full experiment setup."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = "outputs/full_experiment_stability_check_50k"
    cmd = [
        sys.executable,
        "scripts/train_happo_reference.py",
        "--config",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
        "--output-dir",
        output_dir,
        "--total-env-steps",
        "50000",
        "--rollout-length",
        "256",
        "--num-envs",
        "1",
        "--device",
        "cuda",
        "--init-checkpoint",
        "outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt",
        "--uav-imitation-dataset",
        "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz",
        "--uav-imitation-coef",
        "0.03",
        "--uav-imitation-until-steps",
        "10000",
        "--heartbeat-log",
        f"{output_dir}/heartbeat.log",
        "--heartbeat-every-steps",
        "50",
        "--debug-rollout-heartbeat",
    ]
    print("[stability] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[stability] output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
