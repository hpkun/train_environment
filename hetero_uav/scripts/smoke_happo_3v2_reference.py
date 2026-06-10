"""One-command HAPPO reference v0 smoke: train + eval."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/test_happo_3v2_reference")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = args.output_dir
    _run([
        "python", "-u", "scripts/train_happo_reference.py",
        "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
        "--reward-mode", "happo_ref_v0",
        "--opponent-policy", "brma_rule",
        "--total-env-steps", "64",
        "--rollout-length", "16",
        "--max-steps", "64",
        "--ppo-epochs", "2",
        "--train-eval-episodes", "1",
        "--eval-during-training",
        "--eval-interval-steps", "32",
        "--device", args.device,
        "--output-dir", out,
    ])
    _run([
        "python", "-u", "scripts/eval_happo_reference.py",
        "--model", f"{out}/latest/model.pt",
        "--episodes", "1",
        "--device", args.device,
        "--opponent-policy", "brma_rule",
        "--summary-json", f"{out}/eval_summary.json",
        "--configs", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
    ])
    print(f"smoke_output_dir: {ROOT / out}", flush=True)


if __name__ == "__main__":
    main()
