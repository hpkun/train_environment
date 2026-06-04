"""Validate hetero environment stability with the plain MAPPO MLP baseline.

This is an environment stability diagnostic. It does not implement or evaluate
new algorithms, and it does not make formal zero-shot claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"

TRAIN_CONFIG_V2 = "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml"
EVAL_CONFIGS_V2 = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]
EXPECTED_DIMS = {
    "v1": (140, 700),
    "v2": (96, 480),
}


def _tail(text: str, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    return text[-limit:]


def _run_subprocess(cmd: list[str], label: str, timeout: int = 900) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=timeout,
        env=env,
    )
    print(_tail(result.stdout, 1200))
    if result.returncode != 0:
        print(f"{label} failed with returncode={result.returncode}")
        print("stdout tail:")
        print(_tail(result.stdout))
        print("stderr tail:")
        print(_tail(result.stderr))
        raise RuntimeError(f"{label} failed with rc={result.returncode}")


def _read_json(path: Path) -> dict | list:
    if not path.exists():
        raise RuntimeError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_train_log(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Missing train log: {path}")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty train log: {path}")
    return rows


def _finite_float(value, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{field} is not finite: {value}")
    return out


def _validate_training_outputs(seed: int, run_dir: Path,
                               obs_adapter_version: str) -> dict:
    model_path = run_dir / "latest" / "model.pt"
    meta_path = run_dir / "latest" / "meta.json"
    log_csv = run_dir / "train_log.csv"
    if not model_path.exists():
        raise RuntimeError(f"Missing model checkpoint: {model_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing model meta: {meta_path}")

    meta = _read_json(meta_path)
    expected_actor, expected_critic = EXPECTED_DIMS[obs_adapter_version]
    if meta.get("obs_adapter_version") != obs_adapter_version:
        raise RuntimeError(
            f"meta obs_adapter_version mismatch: {meta.get('obs_adapter_version')}"
        )
    if int(meta.get("actor_obs_dim", -1)) != expected_actor:
        raise RuntimeError(f"actor_obs_dim mismatch in {meta_path}")
    if int(meta.get("critic_state_dim", -1)) != expected_critic:
        raise RuntimeError(f"critic_state_dim mismatch in {meta_path}")
    if meta.get("actor_arch", "mlp") != "mlp":
        raise RuntimeError(f"unexpected actor_arch: {meta.get('actor_arch')}")

    rows = _read_train_log(log_csv)
    nan_values = [int(float(row["nan_detected"])) for row in rows]
    if any(v != 0 for v in nan_values):
        raise RuntimeError(f"train NaN detected for seed {seed}")

    first = rows[0]
    last = rows[-1]
    episodes_completed = int(float(last["episodes_completed"]))
    if episodes_completed == 0:
        print(f"warning: seed {seed} completed no episodes in short diagnostic")

    return {
        "seed": seed,
        "obs_adapter_version": obs_adapter_version,
        "observation_mode": meta.get("observation_mode", ""),
        "actor_dim": int(meta["actor_obs_dim"]),
        "critic_dim": int(meta["critic_state_dim"]),
        "iterations": len(rows),
        "total_steps": int(float(last["total_steps"])),
        "episodes_completed": episodes_completed,
        "first_return": _finite_float(first["average_team_return"], "first_return"),
        "last_return": _finite_float(last["average_team_return"], "last_return"),
        "best_return": max(
            _finite_float(row["average_team_return"], "best_return")
            for row in rows
        ),
        "final_red_alive": _finite_float(last["average_red_alive"], "final_red_alive"),
        "final_blue_alive": _finite_float(last["average_blue_alive"], "final_blue_alive"),
        "final_entropy": _finite_float(last["entropy"], "final_entropy"),
        "nan_detected": max(nan_values),
        "model_path": str(model_path),
        "log_csv": str(log_csv),
    }


def _validate_eval_outputs(seed: int, eval_summary_path: Path,
                           eval_episodes: int) -> list[dict]:
    data = _read_json(eval_summary_path)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Invalid eval summary: {eval_summary_path}")

    rows = []
    for rec in data:
        if rec.get("nan_detected"):
            raise RuntimeError(f"eval NaN detected for seed {seed}: {rec['config']}")
        if not rec.get("actor_dim_ok"):
            raise RuntimeError(f"actor_dim_ok False for seed {seed}: {rec['config']}")
        if not rec.get("critic_dim_ok"):
            raise RuntimeError(f"critic_dim_ok False for seed {seed}: {rec['config']}")
        avg_length = _finite_float(rec["avg_length"], "avg_length")
        avg_return = _finite_float(rec["avg_return"], "avg_return")
        if avg_length <= 0:
            raise RuntimeError(f"avg_length <= 0 for seed {seed}: {rec['config']}")

        rows.append({
            "seed": seed,
            "config": rec["config"],
            "episodes": int(rec.get("episodes", eval_episodes)),
            "avg_return": avg_return,
            "avg_length": avg_length,
            "avg_red_alive": _finite_float(rec["avg_red_alive"], "avg_red_alive"),
            "avg_blue_alive": _finite_float(rec["avg_blue_alive"], "avg_blue_alive"),
            "nan_detected": bool(rec["nan_detected"]),
            "actor_dim_ok": bool(rec["actor_dim_ok"]),
            "critic_dim_ok": bool(rec["critic_dim_ok"]),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_report(args: argparse.Namespace, train_rows: list[dict],
                  eval_rows: list[dict], warnings: list[str]) -> dict:
    best_returns = [float(row["best_return"]) for row in train_rows]
    eval_by_config = {}
    for config in sorted({row["config"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["config"] == config]
        returns = [float(row["avg_return"]) for row in rows]
        lengths = [float(row["avg_length"]) for row in rows]
        eval_by_config[config] = {
            "avg_return_mean": float(np.mean(returns)),
            "avg_return_std": float(np.std(returns)),
            "avg_length_mean": float(np.mean(lengths)),
        }

    return {
        "status": "passed",
        "seeds": args.seeds,
        "train_config": args.train_config,
        "eval_configs": args.eval_configs,
        "obs_adapter_version": args.obs_adapter_version,
        "actor_dim": EXPECTED_DIMS[args.obs_adapter_version][0],
        "critic_dim": EXPECTED_DIMS[args.obs_adapter_version][1],
        "total_train_runs": len(train_rows),
        "total_eval_runs": len(eval_rows),
        "any_train_nan": any(int(row["nan_detected"]) != 0 for row in train_rows),
        "any_eval_nan": any(bool(row["nan_detected"]) for row in eval_rows),
        "all_actor_dim_ok": all(bool(row["actor_dim_ok"]) for row in eval_rows),
        "all_critic_dim_ok": all(bool(row["critic_dim_ok"]) for row in eval_rows),
        "min_episodes_completed": min(
            int(row["episodes_completed"]) for row in train_rows
        ),
        "train_best_return_mean": float(np.mean(best_returns)),
        "train_best_return_std": float(np.std(best_returns)),
        "eval_return_by_config": eval_by_config,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir",
                        default="outputs/mappo_baseline_env_stability")
    parser.add_argument("--obs-adapter-version", choices=["v1", "v2"],
                        default="v2")
    parser.add_argument("--train-config", default=TRAIN_CONFIG_V2)
    parser.add_argument("--eval-configs", nargs="+", default=EVAL_CONFIGS_V2)
    args = parser.parse_args()

    if args.obs_adapter_version != "v2":
        print("warning: default stability validation target is v2; running optional check")

    output_dir = Path(args.output_dir)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    warnings: list[str] = []

    for seed in args.seeds:
        run_dir = output_dir / f"seed_{seed}"
        log_csv = run_dir / "train_log.csv"
        print(f"=== Train seed={seed} ===")
        _run_subprocess([
            "python", str(TRAIN_SCRIPT),
            "--config", args.train_config,
            "--obs-adapter-version", args.obs_adapter_version,
            "--iterations", str(args.iterations),
            "--rollout-length", str(args.rollout_length),
            "--max-steps", str(args.max_steps),
            "--seed", str(seed),
            "--device", args.device,
            "--output-dir", str(run_dir),
            "--log-csv", str(log_csv),
            "--opponent-policy", args.opponent_policy,
            "--save-interval", "10",
        ], label=f"train seed={seed}")

        train_row = _validate_training_outputs(
            seed, run_dir, args.obs_adapter_version
        )
        train_rows.append(train_row)
        if int(train_row["episodes_completed"]) == 0:
            warnings.append(f"seed {seed}: no completed episode")

        eval_summary_path = run_dir / "eval_summary.json"
        print(f"=== Eval seed={seed} ===")
        _run_subprocess([
            "python", str(EVAL_SCRIPT),
            "--model", str(run_dir / "latest" / "model.pt"),
            "--obs-adapter-version", args.obs_adapter_version,
            "--episodes", str(args.eval_episodes),
            "--device", args.device,
            "--opponent-policy", args.opponent_policy,
            "--configs", *args.eval_configs,
            "--summary-json", str(eval_summary_path),
        ], label=f"eval seed={seed}")

        eval_rows.extend(
            _validate_eval_outputs(seed, eval_summary_path, args.eval_episodes)
        )

    train_summary = output_dir / "stability_train_summary.csv"
    eval_summary = output_dir / "stability_eval_summary.csv"
    report_path = output_dir / "stability_report.json"
    _write_csv(train_summary, train_rows)
    _write_csv(eval_summary, eval_rows)

    report = _build_report(args, train_rows, eval_rows, warnings)
    if report["any_train_nan"] or report["any_eval_nan"]:
        report["status"] = "failed"
    if not report["all_actor_dim_ok"] or not report["all_critic_dim_ok"]:
        report["status"] = "failed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["status"] != "passed":
        raise RuntimeError(f"Stability validation failed: {report_path}")

    print(f"stability_train_summary: {train_summary}")
    print(f"stability_eval_summary: {eval_summary}")
    print(f"stability_report: {report_path}")
    print("MAPPO baseline environment stability validation passed")


if __name__ == "__main__":
    main()
