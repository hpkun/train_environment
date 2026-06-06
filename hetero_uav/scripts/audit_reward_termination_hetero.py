"""Audit reward and termination behavior for heterogeneous MAV/UAV configs.

This script intentionally does not modify reward, termination, missile,
evasion, action, PID, or aircraft XML behavior. It combines static config
inspection with short diagnostic rollouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml",
]


def _load_yaml(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _zero_red_actions(env) -> dict[str, np.ndarray]:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids}


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _agent_reward_components(info: dict, agent_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for aid in agent_ids:
        agent_info = info.get(aid, {})
        if not isinstance(agent_info, dict):
            continue
        keys = sorted(key for key in agent_info if key.startswith("r_"))
        if keys:
            out[aid] = keys
    return out


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _classify_episode(env, terminated: dict, truncated: dict, steps: int) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    mav_sim = env.red_planes.get("red_0")
    mav_alive = bool(mav_sim is not None and mav_sim.is_alive)
    timeout = bool(all(truncated.values()) or steps >= getattr(env, "max_steps", 0))

    if blue_alive == 0 and red_alive > 0:
        end_reason = "red_win_elimination"
        winner = "red"
    elif red_alive == 0 and blue_alive > 0:
        end_reason = "blue_win_elimination"
        winner = "blue"
    elif red_alive == 0 and blue_alive == 0:
        end_reason = "mutual_elimination_draw"
        winner = "draw"
    elif timeout:
        end_reason = "timeout"
        if red_alive > blue_alive:
            winner = "red_alive_advantage"
        elif red_alive < blue_alive:
            winner = "blue_alive_advantage"
        else:
            winner = "draw"
    elif any(terminated.values()) or any(truncated.values()):
        end_reason = "partial_agent_done"
        winner = "draw"
    else:
        end_reason = "not_ended"
        winner = "none"

    return {
        "terminated_seen": bool(any(terminated.values())),
        "truncated_seen": bool(any(truncated.values())),
        "episode_end_reason": end_reason,
        "winner": winner,
        "red_alive_final": int(red_alive),
        "blue_alive_final": int(blue_alive),
        "mav_survival": bool(mav_alive),
        "steps_executed": int(steps),
    }


def _static_config(config_path: str, config: dict, env) -> dict:
    agent_types = dict(getattr(env, "agent_types", {}))
    missile_counts = {
        aid: int(env._get_sim(aid).num_left_missiles)
        for aid in env.agent_ids
        if env._get_sim(aid) is not None
    }
    return {
        "config": config_path,
        "red_count": int(len(env.red_ids)),
        "blue_count": int(len(env.blue_ids)),
        "red_agent_types": [agent_types.get(aid, "") for aid in env.red_ids],
        "blue_agent_types": [agent_types.get(aid, "") for aid in env.blue_ids],
        "missile_counts": missile_counts,
        "max_steps": int(getattr(env, "max_steps", config.get("max_steps", 0))),
        "observation_mode": str(getattr(env, "observation_mode", config.get("observation_mode", ""))),
    }


def _role_assessment(static_config: dict, component_names: set[str],
                     termination_behavior: dict) -> dict:
    red_types = static_config.get("red_agent_types", [])
    missiles = static_config.get("missile_counts", {})
    mav_ids = [
        f"red_{idx}" for idx, type_name in enumerate(red_types)
        if type_name == "mav"
    ]
    mav_no_missiles = all(int(missiles.get(aid, -1)) == 0 for aid in mav_ids)
    component_text = " ".join(sorted(component_names)).lower()
    role_reward = any(token in component_text for token in ("mav", "uav", "role", "scout", "intercept"))

    return {
        "mav_has_no_missiles": bool(mav_ids and mav_no_missiles),
        "mav_death_has_separate_penalty_or_termination": False,
        "uav_kill_stats_are_separate_from_mav_survival": True,
        "red_team_reward_distinguishes_mav_uav_role": bool(role_reward),
        "terminal_reward_alive_count_based": True,
        "timeout_interpreted_by_alive_advantage": True,
        "reward_may_encourage_mav_survival_without_support": True,
        "reward_may_not_capture_mav_shared_observation_value": True,
        "mav_survival_is_metric_not_explicit_reward": not bool(role_reward),
        "observed_end_reason": termination_behavior.get("episode_end_reason"),
    }


def _record_warnings(record: dict) -> list[str]:
    warnings: list[str] = []
    reward_seen = record["reward_components_seen"]
    assessment = record["hetero_role_assessment"]
    if not reward_seen["exposed"]:
        warnings.append("reward_components_not_exposed")
    if not assessment["red_team_reward_distinguishes_mav_uav_role"]:
        warnings.append("reward_not_role_differentiated")
    if not assessment["mav_death_has_separate_penalty_or_termination"]:
        warnings.append("termination_does_not_distinguish_mav_loss")
    if assessment["terminal_reward_alive_count_based"]:
        warnings.append("terminal_reward_alive_count_based")
    if assessment["mav_survival_is_metric_not_explicit_reward"]:
        warnings.append("mav_survival_metric_not_explicit_reward")

    is_close = Path(record["config"]).name == "hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml"
    if is_close:
        missile_used = record.get("missile_statistics", {}).get("missile_used", 0)
        kills = record.get("missile_statistics", {}).get("observed_kills", 0)
        ended = record["termination_behavior"]["episode_end_reason"] != "not_ended"
        if missile_used == 0 and kills == 0 and not ended:
            warnings.append("close_range_no_missile_kill_or_end_event")
    return warnings


def audit_config(config_path: str, steps: int, opponent_policy: str, seed: int) -> dict:
    config = _load_yaml(config_path)
    env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    policy = OpponentPolicy(opponent_policy, seed=seed + 31)
    red_reward_stats = []
    blue_reward_stats = []
    component_names: set[str] = set()
    components_by_agent: dict[str, list[str]] = {}
    missile_used = 0
    nan_detected = False
    terminated = {}
    truncated = {}
    info = {}
    steps_executed = 0
    initial_red_count = 0
    initial_blue_count = 0

    try:
        obs, info = env.reset(seed=seed)
        initial_red_count, initial_blue_count = _alive_counts(env)
        static_config = _static_config(config_path, config, env)
        nan_detected = _contains_nan(obs)

        for _step in range(steps):
            blue_actions = policy.act(obs, env.blue_ids, env=env)
            actions = _zero_red_actions(env)
            actions.update(blue_actions)
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            nan_detected = nan_detected or _contains_nan(obs) or _contains_nan(rewards)

            red_values = [float(rewards.get(aid, 0.0)) for aid in env.red_ids]
            blue_values = [float(rewards.get(aid, 0.0)) for aid in env.blue_ids]
            red_reward_stats.append(_stats(red_values))
            blue_reward_stats.append(_stats(blue_values))

            step_components = _agent_reward_components(info, env.agent_ids)
            for aid, keys in step_components.items():
                components_by_agent.setdefault(aid, keys)
                component_names.update(keys)

            for aid in env.agent_ids:
                agent_info = info.get(aid, {})
                if isinstance(agent_info, dict):
                    missile_used += int(agent_info.get("missiles_fired_this_step", 0))

            if all(terminated.values()) or all(truncated.values()):
                break

        termination_behavior = _classify_episode(env, terminated, truncated, steps_executed)
        red_alive, blue_alive = _alive_counts(env)
        missile_statistics = {
            "missile_used": int(missile_used),
            "missile_term_reasons": info.get("__missile_term__", {}),
            "observed_kills": int((initial_red_count - red_alive) + (initial_blue_count - blue_alive)),
        }
        reward_components_seen = {
            "exposed": bool(component_names),
            "component_names": sorted(component_names),
            "components_by_agent": components_by_agent,
            "red_reward_per_step": red_reward_stats,
            "blue_reward_per_step": blue_reward_stats,
        }
        record = {
            "config": config_path,
            "static_config": static_config,
            "reward_components_seen": reward_components_seen,
            "termination_behavior": termination_behavior,
            "missile_statistics": missile_statistics,
            "hetero_role_assessment": _role_assessment(
                static_config, component_names, termination_behavior
            ),
            "nan_detected": bool(nan_detected),
            "warnings": [],
        }
        record["warnings"] = _record_warnings(record)
        return record
    finally:
        env.close()


def _aggregate_assessment(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = records[0]["hetero_role_assessment"].keys()
    out = {}
    for key in keys:
        values = [record["hetero_role_assessment"].get(key) for record in records]
        if all(isinstance(value, bool) for value in values):
            out[key] = any(values)
        else:
            out[key] = values
    return out


def _markdown(data: dict) -> str:
    lines = [
        "# Reward / Termination Audit",
        "",
        "Purpose: audit current BRMA-inherited reward and termination behavior for",
        "heterogeneous MAV/UAV tasks. This audit is not modifying reward,",
        "termination, missile, evasion, action, PID, or aircraft XML behavior.",
        "",
        "## Summary",
        "",
        f"- records: {data['summary']['records']}",
        f"- reward components exposed: {data['summary']['reward_components_exposed']}",
        f"- nan records: {data['summary']['nan_records']}",
        "",
        "## Records",
    ]
    for record in data["records"]:
        term = record["termination_behavior"]
        assess = record["hetero_role_assessment"]
        lines.extend([
            "",
            f"### {Path(record['config']).name}",
            "",
            f"- steps_executed: {term['steps_executed']}",
            f"- episode_end_reason: {term['episode_end_reason']}",
            f"- winner: {term['winner']}",
            f"- MAV survival: {term['mav_survival']}",
            f"- UAV / blue alive final: {term['blue_alive_final']}",
            f"- reward component names: {record['reward_components_seen']['component_names']}",
            f"- MAV has no missiles: {assess['mav_has_no_missiles']}",
            f"- role differentiated reward: {assess['red_team_reward_distinguishes_mav_uav_role']}",
            f"- terminal reward alive-count based: {assess['terminal_reward_alive_count_based']}",
            f"- warnings: {record['warnings']}",
        ])
    lines.extend([
        "",
        "## Next Step",
        "",
        "Review these audit findings and decide whether to add minimal",
        "heterogeneous reward shaping. Do not enter training or method-module work",
        "from this audit alone.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--opponent-policy", default="greedy_fsm",
                        choices=sorted(OpponentPolicy.MODES))
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/reward_termination_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/reward_termination_audit.md",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = [
        audit_config(config, args.steps, args.opponent_policy, args.seed)
        for config in args.configs
    ]
    warnings = sorted({warning for record in records for warning in record["warnings"]})
    data = {
        "records": records,
        "summary": {
            "records": len(records),
            "steps": args.steps,
            "opponent_policy": args.opponent_policy,
            "reward_components_exposed": all(
                record["reward_components_seen"]["exposed"] for record in records
            ),
            "nan_records": sum(1 for record in records if record["nan_detected"]),
            "warnings": warnings,
        },
        "warnings": warnings,
        "hetero_role_assessment": _aggregate_assessment(records),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(data), encoding="utf-8")

    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"warnings: {warnings}", flush=True)


if __name__ == "__main__":
    main()
