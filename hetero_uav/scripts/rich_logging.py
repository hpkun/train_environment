"""Small helpers for optional rich experiment logging."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from scripts.experiment_logging_schema import (
    FILE_SCHEMAS,
    TRAIN_METRICS_COLUMNS,
    ensure_schema_files,
)


class RichExperimentLogger:
    def __init__(
        self,
        directory: Path,
        run_id: str,
        method_name: str,
        scenario_name: str,
        device: str,
        num_envs: int,
        rollout_length_per_env: int,
        transitions_per_rollout: int,
    ) -> None:
        self.directory = directory
        self.run_id = run_id
        self.method_name = method_name
        self.scenario_name = scenario_name
        self.device = device
        self.num_envs = int(num_envs)
        self.rollout_length_per_env = int(rollout_length_per_env)
        self.transitions_per_rollout = int(transitions_per_rollout)
        self.start_time = time.time()
        ensure_schema_files(directory)
        self._train_file = (directory / "train_metrics.csv").open("w", newline="", encoding="utf-8")
        self._train_writer = csv.DictWriter(self._train_file, fieldnames=TRAIN_METRICS_COLUMNS)
        self._train_writer.writeheader()

    def close(self) -> None:
        self._train_file.close()

    def write_train_metrics(self, row: dict[str, Any]) -> None:
        elapsed = max(time.time() - self.start_time, 1e-9)
        total_steps = float(row.get("total_env_steps_actual", 0.0) or 0.0)
        defaults = {
            "run_id": self.run_id,
            "method_name": self.method_name,
            "scenario_name": self.scenario_name,
            "wall_time_sec": elapsed,
            "steps_per_second": total_steps / elapsed,
        }
        payload = {col: "" for col in TRAIN_METRICS_COLUMNS}
        payload.update(defaults)
        payload.update(row)
        self._train_writer.writerow(payload)
        self._train_file.flush()

    def write_training_efficiency(self, total_steps: int, nan_detected: bool = False) -> None:
        elapsed = max(time.time() - self.start_time, 1e-9)
        data = {
            "run_id": self.run_id,
            "method_name": self.method_name,
            "device": self.device,
            "num_envs": self.num_envs,
            "rollout_length_per_env": self.rollout_length_per_env,
            "transitions_per_rollout": self.transitions_per_rollout,
            "total_train_steps": int(total_steps),
            "total_wall_time_sec": elapsed,
            "steps_per_second_mean": float(total_steps / elapsed),
            "single_step_inference_time_ms": None,
            "ppo_update_time_ms": None,
            "peak_gpu_memory_gb": None,
            "peak_cpu_memory_gb": None,
            "train_start_time": self.start_time,
            "train_end_time": time.time(),
            "nan_detected": bool(nan_detected),
        }
        (self.directory / "training_efficiency.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")


def write_not_available_attention(directory: Path, method_name: str, scenario: str) -> None:
    ensure_schema_files(directory)
    path = directory / "attention_metrics.csv"
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILE_SCHEMAS["attention_metrics.csv"])
        writer.writerow({
            "method_name": method_name,
            "scenario": scenario,
            "episode_id": "",
            "agent_id": "",
            "availability": "not_available",
        })
