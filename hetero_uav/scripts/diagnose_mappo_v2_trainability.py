"""Diagnose V2 MAPPO trainability: actor 96-dim, critic 480-dim."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config",
                        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    output_dir = "outputs/mappo_v2_trainability"
    log_csv = f"{output_dir}/train_log.csv"

    print(f"Running {args.iterations} iterations (V2, 96-dim actor)...")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--config", args.env_config,
         "--obs-adapter-version", "v2",
         "--iterations", str(args.iterations),
         "--rollout-length", str(args.rollout_length),
         "--seed", str(args.seed),
         "--device", args.device,
         "--output-dir", output_dir,
         "--log-csv", log_csv,
         "--opponent-policy", args.opponent_policy,
         "--save-interval", "10"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"TRAIN FAILED: {result.stderr[-500:]}")
        return

    assert Path(log_csv).exists()
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    rl = rows[-1]

    print()
    print("=== V2 Trainability Summary ===")
    print(f"iterations: {n}")
    print("obs_adapter_version: v2")
    print("actor_obs_dim: 96")
    print("critic_state_dim: 480")
    print("observation_mode: mav_shared_geo")
    print("first_return:         "
          f'{float(rows[0]["average_team_return"]):+.2f}')
    print("last_return:          "
          f'{float(rl["average_team_return"]):+.2f}')
    best_ret = max(float(r["average_team_return"]) for r in rows)
    print(f"best_return:          {best_ret:+.2f}")
    print("final_red_alive:      "
          f'{float(rl["average_red_alive"]):.1f}')
    print("final_blue_alive:     "
          f'{float(rl["average_blue_alive"]):.1f}')
    print("final_entropy:        "
          f'{float(rl["entropy"]):.4f}')
    print(f'nan_detected:         {rl["nan_detected"]}')
    print(f"checkpoint:           {output_dir}/latest/model.pt")
    print(f"log_csv:              {log_csv}")

    assert int(rl["nan_detected"]) == 0
    assert n == args.iterations
    print("V2 trainability smoke: PASSED")


if __name__ == "__main__":
    main()
