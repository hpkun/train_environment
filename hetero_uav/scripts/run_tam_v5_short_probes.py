from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    calibration = Path(args.calibration_dir)
    candidates_path = calibration / "v5_unknown_constants_candidates.csv"
    if not candidates_path.exists():
        raise RuntimeError("no feasible candidate table; probes are forbidden")
    candidates = pd.read_csv(candidates_path)
    for _, candidate in candidates.iterrows():
        name = str(candidate["candidate"])
        config = calibration / f"tam_v5_{name}.yaml"
        for seed in (0, 1, 2):
            output = root / "outputs" / f"tam_v5_{name}_20k_probe_s{seed}"
            command = [sys.executable, "-u", str(root / "scripts/train_happo_reference.py"),
                       "--config", str(config), "--reward-mode", "tam_happo_paper_formula_v5",
                       "--output-dir", str(output), "--total-env-steps", "20480",
                       "--rollout-length", "256", "--num-envs", "1", "--max-steps", "1000",
                       "--device", args.device, "--policy-arch", "pure_happo",
                       "--opponent-policy", "tam_greedy_rule", "--actor-lr", "0.0001",
                       "--critic-lr", "0.0005", "--clip-param", "0.2", "--entropy-coef", "0.01",
                       "--value-coef", "0.5", "--max-grad-norm", "10", "--ppo-epochs", "3",
                       "--critic-epochs", "5", "--gamma", "0.99", "--gae-lambda", "0.95",
                       "--checkpoint-interval-steps", "10240", "--keep-checkpoints", "3",
                       "--enable-rich-logging", "--rich-log-dir", str(output / "rich_logs"),
                       "--seed", str(seed)]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                output.mkdir(parents=True, exist_ok=True)
                (output / "probe_manifest.json").write_text(json.dumps({
                    "candidate": name, "seed": seed, "config": str(config.resolve()),
                    "config_sha256": _sha256(config), "expected_steps": 20480,
                    "known_reward_contract": {
                        "global_reward_scale": 0.005,
                        "uav_weights": [10, 10, 15, 10, 30],
                        "target_weights": [0.35, 0.25, 0.20, 0.20],
                    },
                    "command": command,
                }, indent=2), encoding="utf-8")
                subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
