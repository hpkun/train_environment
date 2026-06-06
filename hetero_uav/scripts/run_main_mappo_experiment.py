"""Minimal main-experiment runner — train then eval MAPPO baseline.

Fixed protocol:
  train: hetero_mav_shared_geo_3v2.yaml
  eval:  hetero_mav_shared_geo_3v2.yaml + hetero_mav_shared_geo_5v4.yaml
  obs_adapter: v2
  reward: brma_legacy
  opponent: greedy_fsm
  algorithm: current shared-actor MAPPO baseline (unchanged)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
OBS_ADAPTER = "v2"
OPPONENT = "greedy_fsm"


def _run(cmd: list[str], label: str, timeout: int | None = None) -> None:
    print(f"[exp] {label}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"[exp] FAIL {label}", flush=True)
        print(result.stdout[-500:], flush=True)
        print(result.stderr[-500:], flush=True)
        raise SystemExit(f"{label} failed (rc={result.returncode})")
    print(f"[exp] OK {label}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal main-experiment runner (MAPPO baseline)"
    )
    parser.add_argument("--total-env-steps", type=int, default=100000)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/main_mappo_experiment")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train ----
    train_cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "train_mappo_baseline.py"),
        "--config", TRAIN_CONFIG,
        "--obs-adapter-version", OBS_ADAPTER,
        "--total-env-steps", str(args.total_env_steps),
        "--rollout-length", str(args.rollout_length),
        "--max-steps", str(args.max_steps),
        "--seed", "0",
        "--device", args.device,
        "--output-dir", str(out_dir),
        "--log-csv", str(out_dir / "train_log.csv"),
        "--opponent-policy", OPPONENT,
        "--save-interval", "10",
    ]
    _run(train_cmd, "train")

    model_pt = out_dir / "latest" / "model.pt"
    meta_json = out_dir / "latest" / "meta.json"
    if not model_pt.exists():
        raise SystemExit(f"missing {model_pt}")
    if not meta_json.exists():
        raise SystemExit(f"missing {meta_json}")

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    actor_dim = int(meta.get("actor_obs_dim", 0))
    critic_dim = int(meta.get("critic_state_dim", 0))
    if actor_dim != 96 or critic_dim != 480:
        raise SystemExit(f"dim mismatch: actor={actor_dim}, critic={critic_dim}")

    # Check train log for NaN
    train_csv = out_dir / "train_log.csv"
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("nan_detected", "1")) != 0:
                raise SystemExit("train_log.csv has nan_detected")

    # ---- 2. Eval ----
    eval_json = out_dir / "eval_summary.json"
    eval_cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model", str(model_pt),
        "--obs-adapter-version", OBS_ADAPTER,
        "--episodes", str(args.eval_episodes),
        "--device", args.device,
        "--opponent-policy", OPPONENT,
        "--configs", *EVAL_CONFIGS,
        "--summary-json", str(eval_json),
    ]
    _run(eval_cmd, "eval")

    if not eval_json.exists():
        raise SystemExit(f"missing {eval_json}")
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    # ---- 3. Summary ----
    summary_records: list[dict] = []
    for rec in eval_data:
        cfg_name = Path(rec.get("config", "")).name
        summary_records.append({
            "seed": args.seed,
            "total_env_steps": args.total_env_steps,
            "train_config": TRAIN_CONFIG,
            "eval_config": rec.get("config", ""),
            "opponent_policy": OPPONENT,
            "obs_adapter_version": OBS_ADAPTER,
            "actor_dim": actor_dim,
            "critic_dim": critic_dim,
            "avg_return": rec.get("avg_return", 0.0),
            "avg_length": rec.get("avg_length", 0.0),
            "red_win_rate": rec.get("red_win_rate", 0.0),
            "blue_win_rate": rec.get("blue_win_rate", 0.0),
            "draw_rate": rec.get("draw_rate", 0.0),
            "timeout_rate": rec.get("timeout_rate", 0.0),
            "mav_survival_rate": rec.get("mav_survival_rate", 0.0),
            "red_alive_final_mean": rec.get("red_alive_final_mean", 0.0),
            "blue_alive_final_mean": rec.get("blue_alive_final_mean", 0.0),
            "nan_detected": rec.get("nan_detected", True),
            "actor_dim_ok": rec.get("actor_dim_ok", False),
            "critic_dim_ok": rec.get("critic_dim_ok", False),
        })

    summary_json = out_dir / "main_experiment_summary.json"
    summary_csv = out_dir / "main_experiment_summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    if summary_records:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)

    # Final checks
    for rec in summary_records:
        if rec["nan_detected"]:
            raise SystemExit(f"eval NaN: {rec['eval_config']}")
        if not rec["actor_dim_ok"] or not rec["critic_dim_ok"]:
            raise SystemExit(f"dim mismatch: {rec['eval_config']}")

    print(f"[exp] output_dir: {out_dir}", flush=True)
    print(f"[exp] summary: {summary_json}", flush=True)
    print(f"[exp] passed — main experiment smoke OK", flush=True)


if __name__ == "__main__":
    main()
