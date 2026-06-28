"""Small helpers for optional rich experiment logging."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from scripts.experiment_logging_schema import (
    FILE_SCHEMAS,
    MISSILE_EVENTS_COLUMNS,
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
        self._missile_file = (directory / "missile_events.csv").open("a", newline="", encoding="utf-8")
        self._missile_writer = csv.DictWriter(self._missile_file, fieldnames=MISSILE_EVENTS_COLUMNS)

    def close(self) -> None:
        self._train_file.close()
        self._missile_file.close()

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

    def write_missile_events(
        self,
        info: dict[str, Any],
        *,
        scenario: str,
        episode_id: int | str,
        step: int | str,
        sim_time: float | str = "",
    ) -> None:
        """Write launch and termination records exposed by env info.

        The environment already decides launch and hit/miss outcomes. This
        method only persists those diagnostics to the rich logging schema.
        """

        rows = []
        for record in info.get("__launch_quality_step__", []) or []:
            rows.append(self._missile_row(
                record,
                scenario=scenario,
                episode_id=episode_id,
                step=step,
                sim_time=sim_time,
                event_type="launch",
            ))
        for record in info.get("__launch_quality_done__", []) or []:
            reason = str(record.get("termination_reason") or "termination")
            rows.append(self._missile_row(
                record,
                scenario=scenario,
                episode_id=episode_id,
                step=step,
                sim_time=sim_time,
                event_type=reason,
            ))
        if not rows:
            return
        for row in rows:
            self._missile_writer.writerow(row)
        self._missile_file.flush()

    def _missile_row(
        self,
        record: dict[str, Any],
        *,
        scenario: str,
        episode_id: int | str,
        step: int | str,
        sim_time: float | str,
        event_type: str,
    ) -> dict[str, Any]:
        is_launch = event_type == "launch"
        hit = bool(record.get("is_success")) or event_type == "hit"
        death_caused = int(hit) if not is_launch else ""
        # For kill_cooldown_blocked / multi_kill_blocked, is_success is False
        # because the status was overridden.  The target was NOT killed.
        raw_reason = record.get("raw_termination_reason") or record.get("termination_reason", "")
        if raw_reason in ("kill_cooldown_blocked", "multi_kill_blocked"):
            death_caused = 0
        return {
            "run_id": self.run_id,
            "scenario": scenario,
            "episode_id": episode_id,
            "step": step,
            "sim_time": sim_time,
            "event_type": event_type,
            "missile_id": record.get("missile_id", ""),
            "owner_id": record.get("shooter_id", ""),
            "owner_team": record.get("shooter_team") or record.get("team", ""),
            "target_id": record.get("target_id", ""),
            "target_team": record.get("target_team", ""),
            "team": record.get("shooter_team") or record.get("team", ""),
            "shooter_id": record.get("shooter_id", ""),
            "shooter_role": record.get("shooter_role", ""),
            "target_role": record.get("target_role", ""),
            "hit": int(hit) if not is_launch else "",
            "lon": "",
            "lat": "",
            "altitude": record.get("shooter_alt_m", ""),
            "distance_to_target": record.get("range_m", ""),
            "hit_success": int(hit) if not is_launch else "",
            "death_caused": death_caused,
            "raw_termination_reason": raw_reason if not is_launch else "",
            "AO_rad": record.get("AO_rad", ""),
            "AO_deg": record.get("AO_deg", ""),
            "TA_rad": record.get("TA_rad", ""),
            "TA_deg": record.get("TA_deg", ""),
            "flight_time_sec": record.get("flight_time_sec", ""),
            "launch_step": record.get("launch_step", record.get("current_step", "")),
            "termination_step": record.get("termination_step", ""),
            "step_delta": record.get("step_delta", ""),
            "target_alive_at_launch": record.get("target_alive_at_launch", ""),
            "target_alive_at_termination": record.get("target_alive_at_termination", ""),
            "shooter_speed_mps": record.get("shooter_speed_mps", ""),
            "target_speed_mps": record.get("target_speed_mps", ""),
            "closing_speed_mps": record.get("closing_speed_mps", ""),
            "shooter_alt_m": record.get("shooter_alt_m", ""),
            "target_alt_m": record.get("target_alt_m", ""),
            "target_selection_mode": record.get("target_selection_mode", ""),
            "selected_target_score": record.get("selected_target_score", ""),
            "selected_target_threat_score": record.get("selected_target_threat_score", ""),
            "selected_target_mav_support_score": record.get("selected_target_mav_support_score", ""),
            "selected_target_shot_quality_score": record.get("selected_target_shot_quality_score", ""),
            "selected_target_range_m": record.get("selected_target_range_m", ""),
            "selected_target_AO_rad": record.get("selected_target_AO_rad", ""),
            "selected_target_TA_rad": record.get("selected_target_TA_rad", ""),
            "selected_target_is_mav_observed": record.get("selected_target_is_mav_observed", ""),
            "candidate_count": record.get("candidate_count", ""),
            "launch_track_source": record.get("launch_track_source", ""),
            "launch_track_ok": record.get("launch_track_ok", ""),
            "launch_track_block_reason": record.get("launch_track_block_reason", ""),
            "range_3d_m": record.get("range_3d_m", ""),
            "range_2d_m": record.get("range_2d_m", ""),
            "ATA_3d_rad": record.get("ATA_3d_rad", ""),
            "TA_3d_rad": record.get("TA_3d_rad", ""),
            "boresight_3d_rad": record.get("boresight_3d_rad", ""),
            "los_elevation_body_rad": record.get("los_elevation_body_rad", ""),
            "los_azimuth_body_rad": record.get("los_azimuth_body_rad", ""),
            "target_relative_altitude_m": record.get("target_relative_altitude_m", ""),
            "launch_geometry_ok_3d": record.get("launch_geometry_ok_3d", ""),
            "range_ok_3d": record.get("range_ok_3d", ""),
            "ata_ok_3d": record.get("ata_ok_3d", ""),
            "ta_ok_3d": record.get("ta_ok_3d", ""),
            "boresight_ok_3d": record.get("boresight_ok_3d", ""),
            "legacy_AO_2d_rad": record.get("legacy_AO_2d_rad", ""),
            "legacy_TA_2d_rad": record.get("legacy_TA_2d_rad", ""),
            "legacy_range_2d_m": record.get("legacy_range_2d_m", ""),
        }

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
