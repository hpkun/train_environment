"""Run the plain MAPPO MLP baseline long-run stability check.

This runner is for environment/baseline stability. It is not a formal
zero-shot experiment and does not implement a new algorithm.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"

TRAIN_CONFIG = "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml"
EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]


def _tail(text: str, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    return text[-limit:]


def _tail_file(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return "(missing)"
    text = path.read_text(encoding="utf-8", errors="replace")
    return _tail(text, limit)


def _stream_pipe(pipe, log_file) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_file.write(line)
            log_file.flush()
    finally:
        pipe.close()


def _run_streaming(cmd: list[str], label: str, stdout_path: Path,
                   stderr_path: Path, timeout: int | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label} stdout log: {stdout_path}", flush=True)
    print(f"{label} stderr log: {stderr_path}", flush=True)

    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_log, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(ROOT),
            env=env,
        )

        stderr_thread = threading.Thread(
            target=_stream_pipe, args=(process.stderr, stderr_log), daemon=True)
        stderr_thread.start()

        start = time.monotonic()
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                stdout_log.write(line)
                stdout_log.flush()
                if timeout is not None and time.monotonic() - start > timeout:
                    process.kill()
                    raise TimeoutError(f"{label} timed out after {timeout}s")
        finally:
            if process.stdout is not None:
                process.stdout.close()

        returncode = process.wait()
        stderr_thread.join(timeout=5)

    if returncode != 0:
        print(f"{label} failed with returncode={returncode}")
        print("stdout tail:")
        print(_tail_file(stdout_path))
        print("stderr tail:")
        print(_tail_file(stderr_path))
        raise RuntimeError(f"{label} failed with rc={returncode}")


def _run(cmd: list[str], label: str, timeout: int = 7200) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
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


def _read_json(path: Path):
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


def _finite(value, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{name} is not finite: {value}")
    return out


def _validate_train(seed: int, run_dir: Path, target_steps: int) -> dict:
    model_path = run_dir / "latest" / "model.pt"
    meta_path = run_dir / "latest" / "meta.json"
    log_csv = run_dir / "train_log.csv"
    if not model_path.exists():
        raise RuntimeError(f"Missing model checkpoint: {model_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing model meta: {meta_path}")

    meta = _read_json(meta_path)
    if meta.get("obs_adapter_version") != "v2":
        raise RuntimeError(f"Unexpected obs_adapter_version: {meta.get('obs_adapter_version')}")
    if int(meta.get("actor_obs_dim", -1)) != 96:
        raise RuntimeError(f"Unexpected actor_obs_dim in {meta_path}")
    if int(meta.get("critic_state_dim", -1)) != 480:
        raise RuntimeError(f"Unexpected critic_state_dim in {meta_path}")
    if meta.get("actor_arch", "mlp") != "mlp":
        raise RuntimeError(f"Unexpected actor_arch: {meta.get('actor_arch')}")

    actual_steps = int(meta.get("total_env_steps_actual", -1))
    if actual_steps < target_steps:
        raise RuntimeError(
            f"total_env_steps_actual {actual_steps} < target {target_steps}")

    rows = _read_train_log(log_csv)
    nan_values = [int(float(row["nan_detected"])) for row in rows]
    if any(v != 0 for v in nan_values):
        raise RuntimeError(f"train NaN detected for seed {seed}")

    first = rows[0]
    last = rows[-1]
    episodes_completed = int(float(last["episodes_completed"]))
    if episodes_completed == 0:
        print(f"warning: seed {seed} completed no episodes")

    return {
        "seed": seed,
        "total_env_steps_target": int(meta.get("total_env_steps_target", target_steps)),
        "total_env_steps_actual": actual_steps,
        "iterations_completed": int(meta.get("iterations_completed", len(rows))),
        "episodes_completed": episodes_completed,
        "first_return": _finite(first["average_team_return"], "first_return"),
        "last_return": _finite(last["average_team_return"], "last_return"),
        "best_return": max(_finite(row["average_team_return"], "best_return")
                           for row in rows),
        "final_red_alive": _finite(last["average_red_alive"], "final_red_alive"),
        "final_blue_alive": _finite(last["average_blue_alive"], "final_blue_alive"),
        "final_entropy": _finite(last["entropy"], "final_entropy"),
        "final_actor_loss": _finite(last["actor_loss"], "final_actor_loss"),
        "final_critic_loss": _finite(last["critic_loss"], "final_critic_loss"),
        "nan_detected": max(nan_values),
        "model_path": str(model_path),
        "log_csv": str(log_csv),
    }


def _validate_eval(seed: int, eval_summary_path: Path) -> list[dict]:
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
        avg_return = _finite(rec["avg_return"], "avg_return")
        avg_length = _finite(rec["avg_length"], "avg_length")
        if avg_length <= 0:
            raise RuntimeError(f"avg_length <= 0 for seed {seed}: {rec['config']}")
        rows.append({
            "seed": seed,
            "config": rec["config"],
            "episodes": int(rec["episodes"]),
            "avg_return": avg_return,
            "avg_length": avg_length,
            "avg_red_alive": _finite(rec["avg_red_alive"], "avg_red_alive"),
            "avg_blue_alive": _finite(rec["avg_blue_alive"], "avg_blue_alive"),
            "red_win_rate": _finite(rec.get("red_win_rate", 0.0), "red_win_rate"),
            "blue_win_rate": _finite(rec.get("blue_win_rate", 0.0), "blue_win_rate"),
            "draw_rate": _finite(rec.get("draw_rate", 0.0), "draw_rate"),
            "timeout_rate": _finite(rec.get("timeout_rate", 0.0), "timeout_rate"),
            "mav_survival_rate": _finite(
                rec.get("mav_survival_rate", 0.0), "mav_survival_rate"),
            "red_alive_final_mean": _finite(
                rec.get("red_alive_final_mean", rec["avg_red_alive"]),
                "red_alive_final_mean"),
            "blue_alive_final_mean": _finite(
                rec.get("blue_alive_final_mean", rec["avg_blue_alive"]),
                "blue_alive_final_mean"),
            "red_dead_final_mean": _finite(
                rec.get("red_dead_final_mean", 0.0), "red_dead_final_mean"),
            "blue_dead_final_mean": _finite(
                rec.get("blue_dead_final_mean", 0.0), "blue_dead_final_mean"),
            "episode_end_reason_counts": json.dumps(
                rec.get("episode_end_reason_counts", {}), sort_keys=True),
            "winner_counts": json.dumps(rec.get("winner_counts", {}), sort_keys=True),
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


def _report(args: argparse.Namespace, train_rows: list[dict],
            eval_rows: list[dict], warnings: list[str]) -> dict:
    best_returns = [float(row["best_return"]) for row in train_rows]
    last_returns = [float(row["last_return"]) for row in train_rows]
    eval_return_by_config = {}
    for cfg in sorted({row["config"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["config"] == cfg]
        returns = [float(row["avg_return"]) for row in rows]
        lengths = [float(row["avg_length"]) for row in rows]
        eval_return_by_config[cfg] = {
            "avg_return_mean": float(np.mean(returns)),
            "avg_return_std": float(np.std(returns)),
            "avg_length_mean": float(np.mean(lengths)),
            "red_win_rate_mean": float(np.mean([r["red_win_rate"] for r in rows])),
            "blue_win_rate_mean": float(np.mean([r["blue_win_rate"] for r in rows])),
            "draw_rate_mean": float(np.mean([r["draw_rate"] for r in rows])),
            "timeout_rate_mean": float(np.mean([r["timeout_rate"] for r in rows])),
            "mav_survival_rate_mean": float(
                np.mean([r["mav_survival_rate"] for r in rows])),
            "red_alive_final_mean": float(
                np.mean([r["red_alive_final_mean"] for r in rows])),
            "blue_alive_final_mean": float(
                np.mean([r["blue_alive_final_mean"] for r in rows])),
            "blue_dead_final_mean": float(
                np.mean([r["blue_dead_final_mean"] for r in rows])),
        }

    return {
        "status": "passed",
        "seeds": args.seeds,
        "total_env_steps": args.total_env_steps,
        "rollout_length": args.rollout_length,
        "train_config": args.train_config,
        "eval_configs": args.eval_configs,
        "obs_adapter_version": "v2",
        "actor_dim": 96,
        "critic_dim": 480,
        "actor_arch": "mlp",
        "total_train_runs": len(train_rows),
        "total_eval_runs": len(eval_rows),
        "any_train_nan": any(int(row["nan_detected"]) != 0 for row in train_rows),
        "any_eval_nan": any(bool(row["nan_detected"]) for row in eval_rows),
        "all_actor_dim_ok": all(bool(row["actor_dim_ok"]) for row in eval_rows),
        "all_critic_dim_ok": all(bool(row["critic_dim_ok"]) for row in eval_rows),
        "min_total_env_steps_actual": min(
            int(row["total_env_steps_actual"]) for row in train_rows),
        "min_episodes_completed": min(
            int(row["episodes_completed"]) for row in train_rows),
        "train_best_return_mean": float(np.mean(best_returns)),
        "train_best_return_std": float(np.std(best_returns)),
        "train_last_return_mean": float(np.mean(last_returns)),
        "train_last_return_std": float(np.std(last_returns)),
        "eval_return_by_config": eval_return_by_config,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--total-env-steps", type=int, default=500000)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir",
                        default="outputs/mappo_balanced_baseline_500k")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--train-config", default=TRAIN_CONFIG)
    parser.add_argument("--eval-configs", nargs="+", default=EVAL_CONFIGS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    warnings: list[str] = []

    for seed in args.seeds:
        run_dir = out_dir / f"seed_{seed}"
        log_csv = run_dir / "train_log.csv"
        train_stdout = run_dir / "train_stdout.log"
        train_stderr = run_dir / "train_stderr.log"
        print(f"=== Long-run train seed={seed} ===")
        _run_streaming([
            "python", "-u", str(TRAIN_SCRIPT),
            "--config", args.train_config,
            "--obs-adapter-version", "v2",
            "--total-env-steps", str(args.total_env_steps),
            "--rollout-length", str(args.rollout_length),
            "--max-steps", str(args.max_steps),
            "--seed", str(seed),
            "--device", args.device,
            "--output-dir", str(run_dir),
            "--log-csv", str(log_csv),
            "--opponent-policy", args.opponent_policy,
            "--save-interval", str(args.save_interval),
        ], label=f"train seed={seed}", stdout_path=train_stdout,
            stderr_path=train_stderr)

        train_row = _validate_train(seed, run_dir, args.total_env_steps)
        train_row["train_stdout_log"] = str(train_stdout)
        train_row["train_stderr_log"] = str(train_stderr)
        train_rows.append(train_row)
        if int(train_row["episodes_completed"]) == 0:
            warnings.append(f"seed {seed}: no completed episode")

        eval_summary = run_dir / "eval_summary.json"
        eval_stdout = run_dir / "eval_stdout.log"
        eval_stderr = run_dir / "eval_stderr.log"
        print(f"=== Long-run eval seed={seed} ===")
        _run_streaming([
            "python", "-u", str(EVAL_SCRIPT),
            "--model", str(run_dir / "latest" / "model.pt"),
            "--obs-adapter-version", "v2",
            "--episodes", str(args.eval_episodes),
            "--device", args.device,
            "--opponent-policy", args.opponent_policy,
            "--configs", *args.eval_configs,
            "--summary-json", str(eval_summary),
        ], label=f"eval seed={seed}", stdout_path=eval_stdout,
            stderr_path=eval_stderr)
        seed_eval_rows = _validate_eval(seed, eval_summary)
        for row in seed_eval_rows:
            row["eval_stdout_log"] = str(eval_stdout)
            row["eval_stderr_log"] = str(eval_stderr)
        eval_rows.extend(seed_eval_rows)

    train_csv = out_dir / "longrun_train_summary.csv"
    eval_csv = out_dir / "longrun_eval_summary.csv"
    report_json = out_dir / "longrun_report.json"
    _write_csv(train_csv, train_rows)
    _write_csv(eval_csv, eval_rows)
    report = _report(args, train_rows, eval_rows, warnings)
    if (report["any_train_nan"] or report["any_eval_nan"]
            or not report["all_actor_dim_ok"] or not report["all_critic_dim_ok"]
            or report["min_total_env_steps_actual"] < args.total_env_steps):
        report["status"] = "failed"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"Long-run check failed: {report_json}")

    print(f"longrun_train_summary: {train_csv}")
    print(f"longrun_eval_summary: {eval_csv}")
    print(f"longrun_report: {report_json}")
    print("MAPPO balanced baseline long-run completed")
    print("This is a long-run baseline stability check, not a formal zero-shot claim.")


if __name__ == "__main__":
    main()
