"""Unified rich logging schema for full experiment runs.

The schema is intentionally plain Python so training/eval scripts can import it
without additional dependencies. Missing metrics should be written as empty
strings or NaN, but columns should remain stable.
"""
from __future__ import annotations

from pathlib import Path
import csv


TRAIN_METRICS_COLUMNS = [
    "run_id", "method_name", "scenario_name", "train_steps",
    "total_env_steps_actual", "wall_time_sec", "steps_per_second",
    "avg_episode_return", "avg_team_reward", "avg_mav_reward", "avg_uav_reward",
    "red_win_rate", "blue_win_rate", "draw_rate", "timeout_rate",
    "red_elimination_win_rate", "red_timeout_alive_advantage_rate",
    "mav_survival_rate", "red_alive_final_mean", "blue_alive_final_mean",
    "red_missiles_fired_mean", "blue_missiles_fired_mean",
    "red_missile_hits_mean", "blue_missile_hits_mean",
    "red_dead_mean", "blue_dead_mean", "kill_death_ratio",
    "relative_win_ratio", "actor_loss", "critic_loss", "entropy",
    "policy_gradient_norm", "value_gradient_norm", "action_saturation_rate",
    "mav_action_saturation_rate", "uav_action_saturation_rate", "nan_detected",
]

EVAL_EPISODE_COLUMNS = [
    "run_id", "checkpoint_name", "eval_scenario", "episode_id", "seed",
    "outcome", "episode_return", "team_reward", "mav_reward",
    "uav_reward_mean", "episode_length", "red_win", "blue_win", "draw",
    "timeout", "red_elimination_win", "red_timeout_alive_advantage",
    "mav_alive", "red_alive_final", "blue_alive_final", "red_missiles_fired",
    "blue_missiles_fired", "red_missile_hits", "blue_missile_hits",
    "red_dead", "blue_dead", "kill_death_ratio", "relative_win_ratio",
    "first_red_fire_time", "first_blue_fire_time", "first_hit_time",
    "first_death_time",
]

EVAL_SUMMARY_COLUMNS = [
    "checkpoint_name", "eval_scenario", "episodes", "avg_episode_return_mean",
    "avg_episode_return_std", "red_win_rate", "blue_win_rate", "draw_rate",
    "timeout_rate", "red_elimination_win_rate",
    "red_timeout_alive_advantage_rate", "mav_survival_rate",
    "red_alive_final_mean", "blue_alive_final_mean",
    "red_missiles_fired_mean", "blue_missiles_fired_mean",
    "red_missile_hits_mean", "blue_missile_hits_mean",
    "red_dead_mean", "blue_dead_mean", "kill_death_ratio",
    "relative_win_ratio", "red_win_rate_ci95",
]

AIRCRAFT_TIMESERIES_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "agent_id",
    "role", "team", "alive", "lon", "lat", "altitude", "roll", "pitch",
    "yaw", "heading", "velocity", "mach", "speed", "alpha", "beta",
    "action_pitch", "action_heading", "action_speed", "action_raw_0",
    "action_raw_1", "action_raw_2", "nearest_enemy_id",
    "nearest_enemy_distance", "target_id", "missile_warning", "is_mav",
    "is_uav",
]

MISSILE_EVENTS_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "event_type",
    "missile_id", "owner_id", "owner_team", "target_id", "target_team",
    "lon", "lat", "altitude", "distance_to_target", "hit_success",
    "death_caused",
]

MISSILE_TIMESERIES_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "missile_id",
    "owner_id", "target_id", "alive", "lon", "lat", "altitude", "speed",
]

REWARD_COMPONENT_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "agent_id",
    "role", "total_reward", "mav_survival_reward", "mav_support_reward",
    "uav_attack_reward", "uav_fire_reward", "uav_hit_reward", "event_reward",
]

PERTURBATION_EVAL_COLUMNS = [
    "perturbation_level", "altitude_delta", "lon_delta", "lat_delta",
    "heading_delta", "velocity_delta", "episodes", "win_rate",
    "avg_cumulative_team_reward", "std_cumulative_team_reward",
    "mav_survival_rate", "red_missile_hits_mean", "blue_dead_mean",
    "availability",
]

ATTENTION_METRICS_COLUMNS = [
    "method_name", "scenario", "episode_id", "agent_id", "attention_entropy",
    "attention_top1_entity", "attention_top1_weight", "masked_enemy_count",
    "masked_ally_count", "availability",
]

FILE_SCHEMAS = {
    "train_metrics.csv": TRAIN_METRICS_COLUMNS,
    "eval_episode_metrics.csv": EVAL_EPISODE_COLUMNS,
    "eval_summary_metrics.csv": EVAL_SUMMARY_COLUMNS,
    "aircraft_timeseries.csv": AIRCRAFT_TIMESERIES_COLUMNS,
    "missile_events.csv": MISSILE_EVENTS_COLUMNS,
    "missile_timeseries.csv": MISSILE_TIMESERIES_COLUMNS,
    "reward_components.csv": REWARD_COMPONENT_COLUMNS,
    "perturbation_eval_summary.csv": PERTURBATION_EVAL_COLUMNS,
    "attention_metrics.csv": ATTENTION_METRICS_COLUMNS,
}

FIELD_DESCRIPTIONS = {
    "relative_win_ratio": "red_win_rate / max(blue_win_rate, epsilon)",
    "kill_death_ratio": "blue_dead_mean / max(red_dead_mean, epsilon)",
    "attention_metrics.csv": "not_available rows are valid when no attention module is implemented",
}


def ensure_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)


def ensure_schema_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, columns in FILE_SCHEMAS.items():
        ensure_csv(directory / filename, columns)
