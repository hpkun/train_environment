"""Short V1/V2 trainability comparison. Subprocess failure = script failure."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
V1_CONFIG = "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir", default="outputs/compare_mappo_v1_v2")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    summary_json = args.summary_json or f"{out_dir}/trainability_summary.json"
    summary_csv = args.summary_csv or f"{out_dir}/trainability_summary.csv"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    runs = [
        ("v1", V1_CONFIG, "v1", f"{out_dir}/v1"),
        ("v2", V2_CONFIG, "v2", f"{out_dir}/v2"),
    ]

    summaries = []
    obs_modes = {"v1": "brma_sensor", "v2": "mav_shared_geo"}
    actor_dims = {"v1": 140, "v2": 96}
    critic_dims = {"v1": 700, "v2": 480}

    for version, config, adapter_ver, run_dir in runs:
        print(f"=== Training {version} ({adapter_ver}) ===")
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT),
             "--config", config,
             "--obs-adapter-version", adapter_ver,
             "--iterations", str(args.iterations),
             "--rollout-length", str(args.rollout_length),
             "--max-steps", str(args.max_steps),
             "--device", args.device,
             "--output-dir", run_dir,
             "--log-csv", f"{run_dir}/train_log.csv",
             "--opponent-policy", args.opponent_policy,
             "--save-interval", "10"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(ROOT), timeout=600, env=env,
        )
        print(result.stdout[-300:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"TRAIN FAILED for {version}:")
            print(f"STDERR: {result.stderr[-500:]}")
            raise RuntimeError(f"Training {version} failed with rc={result.returncode}")
        print()

        log_csv = f"{run_dir}/train_log.csv"
        if not Path(log_csv).exists():
            raise RuntimeError(f"Missing {log_csv}")
        with open(log_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"Empty {log_csv}")
        rl, r0 = rows[-1], rows[0]
        best = max(float(r["average_team_return"]) for r in rows)
        ep = int(float(rl["episodes_completed"]))
        nan = int(float(rl["nan_detected"]))

        if nan != 0:
            raise RuntimeError(f"{version} NaN detected in training log")

        summary = {
            "version": version, "observation_mode": obs_modes[version],
            "config": config, "obs_adapter_version": adapter_ver,
            "actor_dim": actor_dims[version], "critic_dim": critic_dims[version],
            "iterations": len(rows), "total_steps": rl["total_steps"],
            "episodes_completed": ep,
            "first_return": float(r0["average_team_return"]),
            "last_return": float(rl["average_team_return"]),
            "best_return": best,
            "final_red_alive": float(rl["average_red_alive"]),
            "final_blue_alive": float(rl["average_blue_alive"]),
            "final_entropy": float(rl["entropy"]),
            "nan_detected": nan,
            "log_csv": log_csv, "checkpoint": f"{run_dir}/latest/model.pt",
        }
        summaries.append(summary)
        if ep == 0:
            print(f"  [WARN] {version}: no episodes completed")

    # Write summary outputs
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_json, "w") as f:
        json.dump(summaries, f, indent=2)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)

    print("=== V1 vs V2 Summary ===")
    for s in summaries:
        print(f'{s["version"]}: episodes={s["episodes_completed"]} '
              f'best_ret={s["best_return"]:+.2f} '
              f'last_ret={s["last_return"]:+.2f} nan={s["nan_detected"]}')
    print(f"summary_json: {summary_json}")
    print(f"summary_csv:  {summary_csv}")


if __name__ == "__main__":
    main()
