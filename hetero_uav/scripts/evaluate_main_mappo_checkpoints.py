"""Evaluate MAPPO checkpoints from a training run to find best candidates.

This script evaluates every discovered checkpoint on the fixed main experiment
configs, then ranks checkpoints with a diagnostic score. It does not modify the
environment, reward, termination, or training algorithm.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
OBS_ADAPTER = "v2"


def _run(cmd, label, timeout=1200):
    print(f"[ckpt] {label}", flush=True)
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    if result.returncode != 0:
        print(f"[ckpt] FAIL {label}: {result.stderr[-300:]}", flush=True)
    return result


def _eval_one(checkpoint_path, checkpoint_name, episode_count, device, opponent_policy):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_name)
    output_json = str(ROOT / "outputs" / "_checkpoint_eval_tmp" / f"{safe_name}.json")
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model", checkpoint_path,
        "--obs-adapter-version", OBS_ADAPTER,
        "--episodes", str(episode_count),
        "--device", device,
        "--opponent-policy", opponent_policy,
        "--configs", *EVAL_CONFIGS,
        "--summary-json", output_json,
    ]
    result = _run(cmd, f"eval {checkpoint_name} ep={episode_count}")
    if result.returncode != 0:
        return None
    try:
        return json.loads(Path(output_json).read_text(encoding="utf-8"))
    except Exception:
        return None


def _score(records):
    for r in records:
        if "3v2" in r.get("config", ""):
            return (
                r.get("red_win_rate", 0.0)
                + 0.1 * r.get("mav_survival_rate", 0.0)
                + 0.01 * r.get("avg_return", 0.0)
            )
    return 0.0


def _record_from(records, ckpt_name, ckpt_iter):
    out = []
    for r in records:
        out.append({
            "checkpoint": ckpt_name,
            "iteration": ckpt_iter,
            "eval_config": Path(r.get("config", "")).name,
            "avg_return": r.get("avg_return", 0.0),
            "avg_length": r.get("avg_length", 0.0),
            "red_win_rate": r.get("red_win_rate", 0.0),
            "blue_win_rate": r.get("blue_win_rate", 0.0),
            "draw_rate": r.get("draw_rate", 0.0),
            "timeout_rate": r.get("timeout_rate", 0.0),
            "mav_survival_rate": r.get("mav_survival_rate", 0.0),
            "red_alive_final_mean": r.get("red_alive_final_mean", 0.0),
            "blue_alive_final_mean": r.get("blue_alive_final_mean", 0.0),
            "nan_detected": r.get("nan_detected", True),
            "actor_dim_ok": r.get("actor_dim_ok", False),
            "critic_dim_ok": r.get("critic_dim_ok", False),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="outputs/main_mappo_experiment_f22_100k")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="greedy_fsm")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    ckpt_dir = exp_dir / "checkpoints"
    latest_pt = exp_dir / "latest" / "model.pt"

    if not ckpt_dir.exists() and not latest_pt.exists():
        print("[ckpt] ERROR: no checkpoints/ dir and no latest/model.pt", flush=True)
        raise SystemExit(1)

    ckpt_paths = []
    if ckpt_dir.exists():
        for pt in sorted(ckpt_dir.glob("iter_*.pt")):
            name = pt.name
            try:
                it = int(name.replace("iter_", "").replace(".pt", ""))
            except ValueError:
                it = 0
            ckpt_paths.append((name, it, str(pt)))
    if latest_pt.exists():
        ckpt_paths.append(("latest", -1, str(latest_pt)))

    if not ckpt_paths:
        print("[ckpt] ERROR: no checkpoints found", flush=True)
        raise SystemExit(1)

    ckpt_paths.sort(key=lambda x: (x[1] if x[1] >= 0 else 999999999))
    print(f"[ckpt] found {len(ckpt_paths)} checkpoints", flush=True)

    # ---- Full eval ALL checkpoints ----
    ranked = []
    all_records = []
    for name, it, path in ckpt_paths:
        records = _eval_one(path, name, args.eval_episodes, args.device, args.opponent_policy)
        if records is None:
            print(f"[ckpt] skip {name} (eval failed)", flush=True)
            continue
        s = _score(records)
        ranked.append((s, name, it, path))
        all_records.extend(_record_from(records, name, it))
        print(f"[ckpt] evaluated {name} score={s:.4f}", flush=True)

    if not ranked:
        print("[ckpt] ERROR: all checkpoint evaluations failed", flush=True)
        raise SystemExit(1)

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_k = ranked[: args.top_k]

    # ---- Analysis ----
    any_red_win = any(r["red_win_rate"] > 0.0 for r in all_records)
    any_mav_surv = any(r["mav_survival_rate"] > 0.0 for r in all_records)
    all_blue_win = all(r["blue_win_rate"] == 1.0 for r in all_records)

    summary = {
        "total_checkpoints_found": len(ckpt_paths),
        "checkpoints_evaluated": len(ranked),
        "top_k_requested": args.top_k,
        "top_k_returned": len(top_k),
        "any_red_win_gt_zero": any_red_win,
        "any_mav_survival_gt_zero": any_mav_surv,
        "all_checkpoints_blue_win_rate_1": all_blue_win,
        "no_effective_checkpoint_found": not any_red_win and all_blue_win,
        "top_checkpoints": [
            {"checkpoint": n, "iteration": it, "score": round(s, 4)}
            for s, n, it, _ in top_k
        ],
    }

    out_json = Path(args.output_json or str(exp_dir / "checkpoint_eval_summary.json"))
    out_csv = Path(args.output_csv or str(exp_dir / "checkpoint_eval_summary.csv"))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"records": all_records, "summary": summary}, f, indent=2)

    if all_records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    print(f"\noutput_json: {out_json}", flush=True)
    print(f"output_csv:  {out_csv}", flush=True)
    print(f"total checkpoints: {summary['total_checkpoints_found']}", flush=True)
    print(f"evaluated: {summary['checkpoints_evaluated']}", flush=True)
    print(f"any_red_win>0: {any_red_win}", flush=True)
    print(f"any_mav_surv>0: {any_mav_surv}", flush=True)
    print(f"all_blue_win_rate_1: {all_blue_win}", flush=True)
    print(f"no_effective_checkpoint_found: {summary['no_effective_checkpoint_found']}", flush=True)
    if summary["no_effective_checkpoint_found"]:
        print("[ckpt] WARNING: no effective checkpoint found", flush=True)


if __name__ == "__main__":
    main()
