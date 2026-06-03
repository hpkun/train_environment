"""Short V1/V2 trainability comparison diagnostics.

This script is intentionally fail-fast: subprocess failure or NaN in the
training log aborts the whole diagnostic instead of producing a misleading
summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
V1_CONFIG = "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"


def _tail(text: str, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    return text[-limit:]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_training(version: str, config: str, adapter_ver: str, run_dir: Path,
                  args: argparse.Namespace, env: dict[str, str]) -> None:
    log_csv = run_dir / "train_log.csv"
    cmd = [
        "python", str(TRAIN_SCRIPT),
        "--config", config,
        "--obs-adapter-version", adapter_ver,
        "--iterations", str(args.iterations),
        "--rollout-length", str(args.rollout_length),
        "--max-steps", str(args.max_steps),
        "--device", args.device,
        "--output-dir", str(run_dir),
        "--log-csv", str(log_csv),
        "--opponent-policy", args.opponent_policy,
        "--seed", str(args.seed),
        "--save-interval", "10",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=600,
        env=env,
    )
    print(_tail(result.stdout, 1200))
    if result.returncode != 0:
        print(f"TRAIN FAILED for {version} with returncode={result.returncode}")
        print("stdout tail:")
        print(_tail(result.stdout))
        print("stderr tail:")
        print(_tail(result.stderr))
        raise RuntimeError(f"Training {version} failed with rc={result.returncode}")


def _summary_from_run(version: str, config: str, adapter_ver: str,
                      run_dir: Path) -> dict:
    log_csv = run_dir / "train_log.csv"
    checkpoint = run_dir / "latest" / "model.pt"
    meta = _read_json(run_dir / "latest" / "meta.json")
    if not log_csv.exists():
        raise RuntimeError(f"Missing training log: {log_csv}")
    if not checkpoint.exists():
        raise RuntimeError(f"Missing checkpoint: {checkpoint}")

    with log_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty training log: {log_csv}")

    first = rows[0]
    last = rows[-1]
    nan_detected = int(float(last["nan_detected"]))
    if nan_detected != 0:
        raise RuntimeError(f"{version} NaN detected in training log")

    episodes_completed = int(float(last["episodes_completed"]))
    if episodes_completed == 0:
        print(f"warning: {version} has no completed episode in short diagnostic")

    return {
        "version": version,
        "observation_mode": meta.get("observation_mode", ""),
        "config": config,
        "obs_adapter_version": adapter_ver,
        "actor_dim": int(meta.get("actor_obs_dim", 0)),
        "critic_dim": int(meta.get("critic_state_dim", 0)),
        "iterations": len(rows),
        "total_steps": int(float(last["total_steps"])),
        "episodes_completed": episodes_completed,
        "first_return": float(first["average_team_return"]),
        "last_return": float(last["average_team_return"]),
        "best_return": max(float(r["average_team_return"]) for r in rows),
        "final_red_alive": float(last["average_red_alive"]),
        "final_blue_alive": float(last["average_blue_alive"]),
        "final_entropy": float(last["entropy"]),
        "nan_detected": nan_detected,
        "log_csv": str(log_csv),
        "checkpoint": str(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir", default="outputs/compare_mappo_v1_v2")
    parser.add_argument("--summary-json",
                        default="outputs/compare_mappo_v1_v2/trainability_summary.json")
    parser.add_argument("--summary-csv",
                        default="outputs/compare_mappo_v1_v2/trainability_summary.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)
    summary_csv = Path(args.summary_csv)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    runs = [
        ("v1", V1_CONFIG, "v1", out_dir / "v1"),
        ("v2", V2_CONFIG, "v2", out_dir / "v2"),
    ]

    summaries = []

    for version, config, adapter_ver, run_dir in runs:
        print(f"=== Training {version} ({adapter_ver}) ===")
        _run_training(version, config, adapter_ver, run_dir, args, env)
        print()
        summaries.append(_summary_from_run(version, config, adapter_ver, run_dir))

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)

    print("=== V1 vs V2 Summary ===")
    print("this is a diagnostic smoke, not a formal experiment.")
    for s in summaries:
        print(
            f'{s["version"]}: obs={s["observation_mode"]} '
            f'actor_dim={s["actor_dim"]} critic_dim={s["critic_dim"]} '
            f'episodes={s["episodes_completed"]} '
            f'best_ret={s["best_return"]:+.2f} '
            f'last_ret={s["last_return"]:+.2f} nan={s["nan_detected"]}'
        )
    print(f"summary_json: {summary_json}")
    print(f"summary_csv:  {summary_csv}")


if __name__ == "__main__":
    main()
