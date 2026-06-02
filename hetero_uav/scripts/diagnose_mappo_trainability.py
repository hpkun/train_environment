"""Diagnose Stage 1 MAPPO trainability: 20 iterations, CSV log, NaN check."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    output_dir = "outputs/mappo_trainability"
    log_csv = f"{output_dir}/train_log.csv"

    print(f"Running {args.iterations} iterations...")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--config", args.config,
         "--iterations", str(args.iterations),
         "--rollout-length", str(args.rollout_length),
         "--seed", str(args.seed),
         "--device", args.device,
         "--output-dir", output_dir,
         "--log-csv", log_csv,
         "--opponent-policy", args.opponent_policy,
         "--save-interval", "10",
         ],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"TRAIN FAILED: {result.stderr[-500:]}")
        return

    assert Path(log_csv).exists(), f"Missing {log_csv}"
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    r0 = rows[0]
    rl = rows[-1]

    print()
    print("=== Trainability Summary ===")
    print(f'iterations_completed: {n}')
    print(f'first_return:          {float(r0["average_team_return"]):+.2f}')
    print(f'last_return:           {float(rl["average_team_return"]):+.2f}')
    best_ret = max(float(r["average_team_return"]) for r in rows)
    print(f'best_return:           {best_ret:+.2f}')
    print(f'first_episode_length:  {float(r0["average_episode_length"]):.0f}')
    print(f'last_episode_length:   {float(rl["average_episode_length"]):.0f}')
    print(f'final_red_alive:       {float(rl["average_red_alive"]):.1f}')
    print(f'final_blue_alive:      {float(rl["average_blue_alive"]):.1f}')
    print(f'final_entropy:         {float(rl["entropy"]):.4f}')
    print(f'nan_detected:          {rl["nan_detected"]}')
    print(f"checkpoint_path:       {output_dir}/latest/model.pt")
    print(f"log_csv:               {log_csv}")

    assert int(rl["nan_detected"]) == 0, "NaN detected"
    assert n == args.iterations, f"Expected {args.iterations} iters, got {n}"
    print("trainability smoke: PASSED")


if __name__ == "__main__":
    main()
