"""Minimal main-experiment runner — train then eval MAPPO baseline.

Fixed protocol:
  train: hetero_mav_shared_geo_3v2.yaml
  eval:  hetero_mav_shared_geo_3v2.yaml + hetero_mav_shared_geo_5v4.yaml
  obs_adapter: v2
  reward: brma_legacy
  algorithm: current shared-actor MAPPO baseline (unchanged)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentSpec:
    train_config: str = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
    eval_configs: list[str] = field(default_factory=lambda: [
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    ])
    obs_adapter_version: str = "v2"
    opponent_policy: str = "greedy_fsm"
    total_env_steps: int = 100000
    rollout_length: int = 128
    max_steps: int = 1000
    eval_episodes: int = 20
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "outputs/main_mappo_experiment"
    actor_arch: str = "mlp"
    enable_eval_during_training: bool = False
    eval_interval_steps: int = 50000
    train_eval_episodes: int = 5


# -- internal helpers --------------------------------------------------------


def _stream_pipe(pipe, log_path: Path, echo: bool, tail: deque[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as log_file:
        for line in iter(pipe.readline, ""):
            log_file.write(line)
            log_file.flush()
            tail.append(line.rstrip())
            if echo:
                print(line, end="", flush=True)
    pipe.close()


def _run_streaming(cmd: list[str], label: str, stdout_path: Path,
                   stderr_path: Path, timeout: int | None = None) -> None:
    print(f"[exp] {label}: {' '.join(cmd)}", flush=True)
    print(f"[exp] {label} stdout log: {stdout_path}", flush=True)
    print(f"[exp] {label} stderr log: {stderr_path}", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    stdout_tail: deque[str] = deque(maxlen=40)
    stderr_tail: deque[str] = deque(maxlen=40)
    process = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=_stream_pipe, args=(process.stdout, stdout_path, True, stdout_tail), daemon=True)
    stderr_thread = threading.Thread(
        target=_stream_pipe, args=(process.stderr, stderr_path, False, stderr_tail), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        raise SystemExit(f"{label} timed out")
    stdout_thread.join()
    stderr_thread.join()
    if returncode != 0:
        print(f"[exp] FAIL {label}", flush=True)
        print("[exp] stdout tail:", flush=True)
        print("\n".join(stdout_tail), flush=True)
        print("[exp] stderr tail:", flush=True)
        print("\n".join(stderr_tail), flush=True)
        raise SystemExit(f"{label} failed (rc={returncode})")
    print(f"[exp] OK {label}", flush=True)


# -- public API --------------------------------------------------------------


def run_experiment(spec: ExperimentSpec) -> None:
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[exp] protocol", flush=True)
    print(f"[exp] train_config={spec.train_config}", flush=True)
    print(f"[exp] eval_configs={spec.eval_configs}", flush=True)
    print(f"[exp] obs_adapter_version={spec.obs_adapter_version}", flush=True)
    print(f"[exp] opponent_policy={spec.opponent_policy}", flush=True)
    print(f"[exp] actor_arch={spec.actor_arch}", flush=True)
    print(f"[exp] total_env_steps={spec.total_env_steps}", flush=True)
    print(f"[exp] rollout_length={spec.rollout_length}", flush=True)
    print(f"[exp] max_steps={spec.max_steps}", flush=True)
    print(f"[exp] eval_episodes={spec.eval_episodes}", flush=True)
    print(f"[exp] eval_during_training={spec.enable_eval_during_training}", flush=True)
    print(f"[exp] output_dir={out_dir}", flush=True)

    # ---- 1. Train ----
    train_cmd = [
        "python", "-u",
        str(ROOT / "scripts" / "train_mappo_baseline.py"),
        "--config", spec.train_config,
        "--obs-adapter-version", spec.obs_adapter_version,
        "--total-env-steps", str(spec.total_env_steps),
        "--rollout-length", str(spec.rollout_length),
        "--max-steps", str(spec.max_steps),
        "--seed", str(spec.seed),
        "--device", spec.device,
        "--output-dir", str(out_dir),
        "--log-csv", str(out_dir / "train_log.csv"),
        "--opponent-policy", spec.opponent_policy,
        "--actor-arch", spec.actor_arch,
        "--save-interval", "10",
    ]
    if spec.enable_eval_during_training:
        train_cmd.extend([
            "--eval-during-training",
            "--eval-interval-steps", str(spec.eval_interval_steps),
            "--train-eval-episodes", str(spec.train_eval_episodes),
            "--eval-configs", *spec.eval_configs,
        ])
    _run_streaming(train_cmd, "train", out_dir / "train_stdout.log", out_dir / "train_stderr.log")

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

    train_csv = out_dir / "train_log.csv"
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("nan_detected", "1")) != 0:
                raise SystemExit("train_log.csv has nan_detected")

    # ---- 2. Eval ----
    eval_json = out_dir / "eval_summary.json"
    eval_cmd = [
        "python", "-u",
        str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model", str(model_pt),
        "--obs-adapter-version", spec.obs_adapter_version,
        "--episodes", str(spec.eval_episodes),
        "--device", spec.device,
        "--opponent-policy", spec.opponent_policy,
        "--configs", *spec.eval_configs,
        "--summary-json", str(eval_json),
    ]
    _run_streaming(eval_cmd, "eval", out_dir / "eval_stdout.log", out_dir / "eval_stderr.log")

    if not eval_json.exists():
        raise SystemExit(f"missing {eval_json}")
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

    # ---- 3. Summary ----
    summary_records: list[dict] = []
    for rec in eval_data:
        summary_records.append({
            "seed": spec.seed,
            "total_env_steps": spec.total_env_steps,
            "train_config": spec.train_config,
            "eval_config": rec.get("config", ""),
            "opponent_policy": spec.opponent_policy,
            "actor_arch": spec.actor_arch,
            "obs_adapter_version": spec.obs_adapter_version,
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

    for rec in summary_records:
        if rec["nan_detected"]:
            raise SystemExit(f"eval NaN: {rec['eval_config']}")
        if not rec["actor_dim_ok"] or not rec["critic_dim_ok"]:
            raise SystemExit(f"dim mismatch: {rec['eval_config']}")

    best_pt = out_dir / "best" / "model.pt"
    print(f"[exp] output_dir: {out_dir}", flush=True)
    print(f"[exp] summary: {summary_json}", flush=True)
    print(f"[exp] best_checkpoint_exists: {best_pt.exists()}", flush=True)
    print(f"[exp] passed — main experiment smoke OK", flush=True)


# -- CLI entry (brma_legacy mainline) ----------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal main-experiment runner (MAPPO baseline)")
    parser.add_argument("--total-env-steps", type=int, default=100000)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/main_mappo_experiment")
    parser.add_argument("--opponent-policy", choices=["rule_nearest", "greedy_fsm"], default="greedy_fsm")
    parser.add_argument('--eval-during-training', action='store_true')
    parser.add_argument('--eval-interval-steps', type=int, default=50000)
    parser.add_argument('--train-eval-episodes', type=int, default=5)
    args = parser.parse_args()

    spec = ExperimentSpec(
        total_env_steps=args.total_env_steps,
        rollout_length=args.rollout_length,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        opponent_policy=args.opponent_policy,
        enable_eval_during_training=args.eval_during_training,
        eval_interval_steps=args.eval_interval_steps,
        train_eval_episodes=args.train_eval_episodes,
    )
    run_experiment(spec)


if __name__ == "__main__":
    main()
