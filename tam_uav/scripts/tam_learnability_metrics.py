"""Small logging helpers for TAM paper Vanilla-HAPPO learnability runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.vanilla_happo_runtime import new_side_tracker, update_side_timing


class RecordWriter:
    """Append stable-schema records to CSV and JSONL immediately."""

    def __init__(self, csv_path: Path, jsonl_path: Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.fieldnames = None

    def append(self, record: dict) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(record)
        extras = set(record) - set(self.fieldnames)
        if extras:
            raise ValueError(f"record schema gained unexpected fields: {sorted(extras)}")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
            handle.flush()
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def start_episode(env, episode_index: int, episode_seed: int,
                  environment_step_start: int, policy_version: int) -> dict:
    agents = list(env.task.agents)
    controlled = list(env.agent_ids)
    by_side = {side: [agent for agent in agents if agent.side == side]
               for side in ("red", "blue")}
    combat = {side: [agent for agent in by_side[side]
                     if agent.aircraft_type.role == "attack_uav"]
              for side in ("red", "blue")}
    return {
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "environment_step_start": int(environment_step_start),
        "policy_version": int(policy_version),
        "controlled_ids": controlled,
        "roles": dict(env.agent_roles),
        "returns": {aid: 0.0 for aid in controlled},
        "reward_components": {aid: {} for aid in controlled},
        "side_tracker": new_side_tracker(),
        "initial_side_count": {side: len(by_side[side]) for side in by_side},
        "initial_combat_count": {side: len(combat[side]) for side in combat},
        "target_consistency_violation": 0,
        "finite": True,
    }


def update_episode(accumulator: dict, env, rewards: dict, info: dict) -> None:
    for aid in accumulator["controlled_ids"]:
        accumulator["returns"][aid] += float(rewards[aid])
        for key, value in info["reward_components"].get(aid, {}).items():
            target = accumulator["reward_components"][aid]
            target[key] = target.get(key, 0.0) + float(value)
    update_side_timing(accumulator["side_tracker"], info, env.task.agents)
    accumulator["target_consistency_violation"] += len(
        info.get("target_consistency_violation", []))
    numeric_components = [
        value for aid in accumulator["controlled_ids"]
        for value in info["reward_components"].get(aid, {}).values()]
    accumulator["finite"] &= bool(
        np.isfinite(list(rewards.values())).all()
        and np.isfinite(numeric_components).all()
        and np.isfinite(env.get_state()).all())


def _safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def finish_episode(accumulator: dict, env, info: dict,
                   environment_step_end: int, algorithm_label: str,
                   environment_revision: str, experiment_protocol: str) -> dict:
    agents = list(env.task.agents)
    by_side = {side: [agent for agent in agents if agent.side == side]
               for side in ("red", "blue")}
    combat = {side: [agent for agent in by_side[side]
                     if agent.aircraft_type.role == "attack_uav"]
              for side in ("red", "blue")}
    record = {
        "episode_index": accumulator["episode_index"],
        "episode_seed": accumulator["episode_seed"],
        "environment_step_start": accumulator["environment_step_start"],
        "environment_step_end": int(environment_step_end),
        "episode_length": int(info["episode_step"]),
        "winner": info["winner"],
        "termination_reason": info["termination_reason"],
        "red_team_episode_return": float(sum(accumulator["returns"].values())),
    }
    for aid in accumulator["controlled_ids"]:
        record[f"agent_return/{aid}"] = float(accumulator["returns"][aid])
    for role in ("mav", "attack_uav"):
        values = [accumulator["returns"][aid]
                  for aid in accumulator["controlled_ids"]
                  if accumulator["roles"].get(aid) == role]
        record[f"role_mean_return/{role}"] = (
            float(np.mean(values)) if values else None)
        record[f"role_total_return/{role}"] = (
            float(sum(values)) if values else None)
    for side in ("red", "blue"):
        survivors = sum(agent.alive for agent in by_side[side])
        combat_survivors = sum(agent.alive for agent in combat[side])
        tracker = accumulator["side_tracker"][side]
        fired, hits = tracker["missiles_fired"], tracker["hits"]
        metrics = [info["aircraft_metrics"][agent.agent_id]
                   for agent in by_side[side]]
        record.update({
            f"{side}_initial_count": int(
                accumulator["initial_side_count"][side]),
            f"{side}_initial_combat_count": int(
                accumulator["initial_combat_count"][side]),
            f"{side}_survival_rate": _safe_rate(
                survivors, accumulator["initial_side_count"][side]),
            f"{side}_combat_survival_rate": _safe_rate(
                combat_survivors, accumulator["initial_combat_count"][side]),
            f"{side}_survivor_count": int(survivors),
            f"{side}_combat_survivor_count": int(combat_survivors),
            f"{side}_missiles_fired": int(fired),
            f"{side}_hits": int(hits),
            f"{side}_hit_rate": _safe_rate(hits, fired),
            f"{side}_kills": int(info["kills"][side]),
            f"{side}_boundary_deaths": int(sum(
                item["death_reason"] == "boundary" for item in metrics)),
            f"{side}_crashes": int(sum(
                item["death_reason"] in {"crash", "nonfinite"} for item in metrics)),
            f"{side}_structural_failures": 0,
            f"{side}_maximum_speed_mps": float(max(
                (item["max_speed_mps"] for item in metrics), default=0.0)),
            f"{side}_maximum_load_g": float(max(
                (item["max_abs_load_factor_g"] for item in metrics), default=0.0)),
            f"first_detection_time_s/{side}": tracker["first_detection_time_s"],
            f"first_attack_range_entry_s/{side}": tracker[
                "first_attack_range_entry_s"],
            f"first_launch_time_s/{side}": tracker["first_launch_time_s"],
            f"first_hit_time_s/{side}": tracker["first_hit_time_s"],
            f"target_switches/{side}": int(tracker["target_switches"]),
        })
    controlled_totals = {}
    for aid in accumulator["controlled_ids"]:
        for key, value in accumulator["reward_components"][aid].items():
            record[f"reward_component/agent/{aid}/{key}"] = float(value)
            controlled_totals[key] = controlled_totals.get(key, 0.0) + float(value)
    for key, value in controlled_totals.items():
        record[f"reward_component/controlled_total/{key}"] = value
    record.update({
        "target_consistency_violation": int(
            accumulator["target_consistency_violation"]),
        "structural_failures": int(
            record["red_structural_failures"] + record["blue_structural_failures"]),
        "finite": bool(accumulator["finite"]),
        "environment_fidelity_revision": environment_revision,
        "experiment_protocol": experiment_protocol,
        "algorithm_label": algorithm_label,
        "policy_version": accumulator["policy_version"],
    })
    return record


def flatten_evaluation(result: dict, *, environment_steps: int,
                       trainer_update_count: int, policy_version: int,
                       actor_sharing: str, algorithm_label: str,
                       evaluation_stage: str, checkpoint=None) -> dict:
    episode = result["episodes_detail"][0]
    row = {
        "environment_steps": int(environment_steps),
        "trainer_update_count": int(trainer_update_count),
        "policy_version": int(policy_version),
        "episode_seed": int(episode["episode_seed"]),
        "winner": episode["winner"],
        "termination_reason": episode["termination_reason"],
        "red_team_episode_return": float(episode["red_episode_return"]),
        "episode_length": int(episode["episode_steps"]),
        "red_survival_rate": float(episode["red_survival_rate"]),
        "blue_survival_rate": float(episode["blue_survival_rate"]),
        "red_combat_survival_rate": float(episode["red_combat_survival_rate"]),
        "blue_combat_survival_rate": float(episode["blue_combat_survival_rate"]),
        "red_missiles_fired": int(episode["red_missiles_fired"]),
        "blue_missiles_fired": int(episode["blue_missiles_fired"]),
        "red_hits": int(episode["red_hits"]),
        "blue_hits": int(episode["blue_hits"]),
        "red_hit_rate": _safe_rate(episode["red_hits"], episode["red_missiles_fired"]),
        "blue_hit_rate": _safe_rate(episode["blue_hits"], episode["blue_missiles_fired"]),
        "red_kills": int(episode["red_kills"]),
        "blue_kills": int(episode["blue_kills"]),
        "red_boundary_deaths": int(episode["red_boundary"]),
        "blue_boundary_deaths": int(episode["blue_boundary"]),
        "red_crashes": int(episode["red_crashes"]),
        "blue_crashes": int(episode["blue_crashes"]),
        "red_maximum_speed_mps": float(episode["red_maximum_speed_mps"]),
        "blue_maximum_speed_mps": float(episode["blue_maximum_speed_mps"]),
        "red_maximum_load_g": float(episode["red_maximum_load_g"]),
        "blue_maximum_load_g": float(episode["blue_maximum_load_g"]),
        "target_consistency_violation": int(episode["target_consistency_violations"]),
        "structural_failures": int(
            episode["red_structural_failures"] + episode["blue_structural_failures"]),
        "finite": bool(episode["finite"]),
        "actor_sharing": actor_sharing,
        "algorithm_label": algorithm_label,
        "environment_fidelity_revision": result["environment_fidelity_revision"],
        "experiment_protocol": result["experiment_protocol"],
        "evaluation_stage": evaluation_stage,
        "checkpoint": checkpoint,
    }
    for side in ("red", "blue"):
        for name in ("initial_count", "initial_combat_count", "survivor_count",
                     "combat_survivor_count"):
            row[f"{side}_{name}"] = int(episode[f"{side}_{name}"])
    for aid, value in episode["red_agent_returns"].items():
        row[f"agent_return/{aid}"] = float(value)
    for side in ("red", "blue"):
        for name in ("first_detection_time_s", "first_attack_range_entry_s",
                     "first_launch_time_s", "first_hit_time_s", "target_switches"):
            row[f"{name}/{side}"] = episode[f"{side}_{name}"]
    return row


def summarize_baseline(result: dict) -> dict:
    episodes = result["episodes_detail"]
    summary = {
        "baseline": result["baseline"],
        "episode_seeds": list(result["episode_seeds"]),
        "episodes": len(episodes),
        "win_rate": float(np.mean([item["winner"] == "red" for item in episodes])),
        "red_team_return": float(np.mean([item["red_episode_return"] for item in episodes])),
        "episode_length": float(np.mean([item["episode_steps"] for item in episodes])),
        "red_survival_rate": float(np.mean([item["red_survival_rate"] for item in episodes])),
        "blue_survival_rate": float(np.mean([item["blue_survival_rate"] for item in episodes])),
        "red_missiles_fired": float(np.mean([item["red_missiles_fired"] for item in episodes])),
        "blue_missiles_fired": float(np.mean([item["blue_missiles_fired"] for item in episodes])),
        "red_hits": float(np.mean([item["red_hits"] for item in episodes])),
        "blue_hits": float(np.mean([item["blue_hits"] for item in episodes])),
        "red_hit_rate": _safe_rate(
            sum(item["red_hits"] for item in episodes),
            sum(item["red_missiles_fired"] for item in episodes)),
        "blue_hit_rate": _safe_rate(
            sum(item["blue_hits"] for item in episodes),
            sum(item["blue_missiles_fired"] for item in episodes)),
        "red_boundary_deaths": int(sum(item["red_boundary"] for item in episodes)),
        "blue_boundary_deaths": int(sum(item["blue_boundary"] for item in episodes)),
        "red_crashes": int(sum(item["red_crashes"] for item in episodes)),
        "blue_crashes": int(sum(item["blue_crashes"] for item in episodes)),
        "finite": bool(all(item["finite"] for item in episodes)),
        "target_consistency_violation": int(sum(
            item["target_consistency_violations"] for item in episodes)),
    }
    timing_names = (
        "first_detection_time_s", "first_attack_range_entry_s",
        "first_launch_time_s", "first_hit_time_s", "target_switches")
    for side in ("red", "blue"):
        for name in timing_names:
            values = [item[f"{side}_{name}"] for item in episodes
                      if item.get(f"{side}_{name}") is not None]
            numeric = np.asarray(values, dtype=float)
            summary[f"{name}/{side}"] = {
                "mean": float(numeric.mean()) if len(numeric) else None,
                "std": float(numeric.std()) if len(numeric) else None,
                "sample_count": int(len(numeric)),
                "missing_count": int(len(episodes) - len(numeric)),
            }
    return summary


def strictly_better_evaluation(best_return, candidate_return) -> bool:
    return best_return is None or float(candidate_return) > float(best_return)
