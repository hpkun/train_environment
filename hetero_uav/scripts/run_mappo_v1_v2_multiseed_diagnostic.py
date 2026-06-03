"""Multi-seed V1/V2 MAPPO diagnostic. Fail-fast on NaN/dim mismatch."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"

V1_CONFIG = "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
V1_EVAL_CFGS = [
    "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml",
    "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml",
]
V2_EVAL_CFGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]

OBS_MODES = {"v1": "brma_sensor", "v2": "mav_shared_geo"}
ACTOR_DIMS = {"v1": 140, "v2": 96}
CRITIC_DIMS = {"v1": 700, "v2": 480}


def _subp(cmd, timeout=600):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT), timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED rc={r.returncode}: {r.stderr[-500:]}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir", default="outputs/mappo_v1_v2_multiseed")
    parser.add_argument("--train-summary-csv", default=None)
    parser.add_argument("--eval-summary-csv", default=None)
    parser.add_argument("--aggregate-json", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    train_csv = args.train_summary_csv or f"{out_dir}/train_summary.csv"
    eval_csv = args.eval_summary_csv or f"{out_dir}/eval_summary.csv"
    agg_json = args.aggregate_json or f"{out_dir}/aggregate_summary.json"

    train_rows = []
    eval_rows = []

    for version, train_cfg, adapter_ver, eval_cfgs in [
        ("v1", V1_CONFIG, "v1", V1_EVAL_CFGS),
        ("v2", V2_CONFIG, "v2", V2_EVAL_CFGS),
    ]:
        for seed in args.seeds:
            run_dir = f"{out_dir}/{version}/seed_{seed}"
            log_csv = f"{run_dir}/train_log.csv"

            # ---- train ----
            print(f"=== Train {version} seed={seed} ===")
            _subp([sys.executable, str(TRAIN_SCRIPT),
                   "--config", train_cfg,
                   "--obs-adapter-version", adapter_ver,
                   "--iterations", str(args.iterations),
                   "--rollout-length", str(args.rollout_length),
                   "--max-steps", str(args.max_steps),
                   "--seed", str(seed),
                   "--device", args.device,
                   "--output-dir", run_dir,
                   "--log-csv", log_csv,
                   "--opponent-policy", args.opponent_policy,
                   "--save-interval", "10"])

            with open(log_csv) as f:
                rows = list(csv.DictReader(f))
            rl, r0 = rows[-1], rows[0]
            best = max(float(r["average_team_return"]) for r in rows)
            ep = int(float(rl["episodes_completed"]))
            nan = int(float(rl["nan_detected"]))
            if nan != 0:
                raise RuntimeError(f"{version} s{seed} NaN")
            if ep == 0:
                print(f"  [WARN] {version} s{seed}: no episodes")

            train_rows.append({
                "version": version, "seed": seed,
                "observation_mode": OBS_MODES[version],
                "config": train_cfg,
                "obs_adapter_version": adapter_ver,
                "actor_dim": ACTOR_DIMS[version],
                "critic_dim": CRITIC_DIMS[version],
                "iterations": len(rows), "total_steps": rl["total_steps"],
                "episodes_completed": ep,
                "first_return": float(r0["average_team_return"]),
                "last_return": float(rl["average_team_return"]),
                "best_return": best,
                "final_red_alive": float(rl["average_red_alive"]),
                "final_blue_alive": float(rl["average_blue_alive"]),
                "final_entropy": float(rl["entropy"]),
                "nan_detected": nan,
                "model_path": f"{run_dir}/latest/model.pt",
                "log_csv": log_csv,
            })

            # ---- eval ----
            eval_summary_path = f"{run_dir}/eval_summary.json"
            model_path = f"{run_dir}/latest/model.pt"
            print(f"=== Eval {version} seed={seed} ===")
            _subp([sys.executable, str(EVAL_SCRIPT),
                   "--model", model_path,
                   "--obs-adapter-version", adapter_ver,
                   "--episodes", str(args.eval_episodes),
                   "--device", args.device,
                   "--opponent-policy", args.opponent_policy,
                   "--configs", *eval_cfgs,
                   "--summary-json", eval_summary_path])

            with open(eval_summary_path) as f:
                eval_data = json.load(f)
            for rec in eval_data:
                rec["version"] = version
                rec["seed"] = seed
                eval_rows.append(rec)
                if rec["nan_detected"] or not rec["actor_dim_ok"] or not rec["critic_dim_ok"]:
                    raise RuntimeError(
                        f"{version} s{seed} {rec['config']}: "
                        f"nan={rec['nan_detected']} "
                        f"adim={rec['actor_dim_ok']} cdim={rec['critic_dim_ok']}")
            print()

    # ---- write CSVs ----
    os.makedirs(out_dir, exist_ok=True)
    for path, rows, cols in [
        (train_csv, train_rows, train_rows[0].keys()),
        (eval_csv, eval_rows, eval_rows[0].keys()),
    ]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # ---- aggregate ----
    agg = {}
    for version in ("v1", "v2"):
        t_rows = [r for r in train_rows if r["version"] == version]
        e_rows = [r for r in eval_rows if r["version"] == version]
        bests = [r["best_return"] for r in t_rows]
        lasts = [r["last_return"] for r in t_rows]
        eps = [r["episodes_completed"] for r in t_rows]

        eval_cfg_agg = {}
        for cfg in sorted(set(r["config"] for r in e_rows)):
            cfg_rows = [r for r in e_rows if r["config"] == cfg]
            rets = [r["avg_return"] for r in cfg_rows]
            eval_cfg_agg[cfg] = {
                "avg_return_mean": float(np.mean(rets)),
                "avg_return_std": float(np.std(rets)),
                "avg_length_mean": float(np.mean([r["avg_length"] for r in cfg_rows])),
                "avg_red_alive_mean": float(np.mean([r["avg_red_alive"] for r in cfg_rows])),
                "avg_blue_alive_mean": float(np.mean([r["avg_blue_alive"] for r in cfg_rows])),
            }

        agg[version] = {
            "seeds": [r["seed"] for r in t_rows],
            "train_best_return_mean": float(np.mean(bests)),
            "train_best_return_std": float(np.std(bests)),
            "train_last_return_mean": float(np.mean(lasts)),
            "train_last_return_std": float(np.std(lasts)),
            "episodes_completed_mean": float(np.mean(eps)),
            "final_red_alive_mean": float(np.mean([r["final_red_alive"] for r in t_rows])),
            "final_blue_alive_mean": float(np.mean([r["final_blue_alive"] for r in t_rows])),
            "eval_by_config": eval_cfg_agg,
        }

    with open(agg_json, "w") as f:
        json.dump(agg, f, indent=2)

    # Print summary
    for version in ("v1", "v2"):
        a = agg[version]
        print(f"=== {version} aggregate ===")
        print(f"  train_best_ret: {a['train_best_return_mean']:+.2f} "
              f"({a['train_best_return_std']:.2f})")
        for cfg, d in a["eval_by_config"].items():
            print(f"  eval {Path(cfg).stem}: ret={d['avg_return_mean']:+.2f}"
                  f"({d['avg_return_std']:.2f})")
    print(f"aggregate_json: {agg_json}")
    print("Note: diagnostic only, not a formal experiment.")


if __name__ == "__main__":
    main()
