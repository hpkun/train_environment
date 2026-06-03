"""Short trainability comparison: V1 brma_sensor vs V2 mav_shared_geo."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
V1_CONFIG = "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"

RUNS = [
    ("v1", V1_CONFIG, "v1", "outputs/compare_mappo_v1_v2/v1"),
    ("v2", V2_CONFIG, "v2", "outputs/compare_mappo_v1_v2/v2"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    for version, config, adapter_ver, out_dir in RUNS:
        print(f"=== Training {version} ({adapter_ver}) ===")
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT),
             "--config", config,
             "--obs-adapter-version", adapter_ver,
             "--iterations", str(args.iterations),
             "--rollout-length", str(args.rollout_length),
             "--max-steps", str(args.max_steps),
             "--device", args.device,
             "--output-dir", out_dir,
             "--log-csv", f"{out_dir}/train_log.csv",
             "--opponent-policy", args.opponent_policy,
             "--save-interval", "10"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(ROOT), timeout=600, env=env,
        )
        print(result.stdout[-200:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"TRAIN FAILED: {result.stderr[-300:]}")
        print()

    # Read results
    print("=== V1 vs V2 Comparison ===")
    header = ["version", "observation_mode", "actor_dim", "critic_dim",
              "iterations", "total_steps", "episodes_completed",
              "first_return", "last_return", "best_return",
              "final_red_alive", "final_blue_alive",
              "final_entropy", "nan_detected", "log_csv", "checkpoint"]
    print(" | ".join(f"{h:>16}" for h in header))

    obs_modes = {"v1": "brma_sensor", "v2": "mav_shared_geo"}
    actor_dims = {"v1": 140, "v2": 96}
    critic_dims = {"v1": 700, "v2": 480}

    for version, _, _, out_dir in RUNS:
        log_csv = Path(f"{out_dir}/train_log.csv")
        if not log_csv.exists():
            print(f"  {version}: MISSING LOG")
            continue
        with open(log_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  {version}: EMPTY LOG")
            continue
        rl = rows[-1]
        r0 = rows[0]
        best = max(float(r["average_team_return"]) for r in rows)
        ep = int(float(rl["episodes_completed"]))
        vals = {
            "version": version,
            "observation_mode": obs_modes[version],
            "actor_dim": actor_dims[version],
            "critic_dim": critic_dims[version],
            "iterations": len(rows),
            "total_steps": rl["total_steps"],
            "episodes_completed": ep,
            "first_return": f'{float(r0["average_team_return"]):+.2f}',
            "last_return": f'{float(rl["average_team_return"]):+.2f}',
            "best_return": f"{best:+.2f}",
            "final_red_alive": f'{float(rl["average_red_alive"]):.1f}',
            "final_blue_alive": f'{float(rl["average_blue_alive"]):.1f}',
            "final_entropy": f'{float(rl["entropy"]):.4f}',
            "nan_detected": str(rl["nan_detected"]),
            "log_csv": str(log_csv),
            "checkpoint": f"{out_dir}/latest/model.pt",
        }
        print(" | ".join(f"{str(vals[k]):>16}" for k in header))
        if ep == 0:
            print(f"  [WARN] {version}: no episodes completed")

    print()
    print("Note: this is a short trainability smoke, not a formal experiment.")


if __name__ == "__main__":
    main()
