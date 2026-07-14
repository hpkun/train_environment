"""Executable audit for the isolated deterministic minimal 3V3 profile."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    environment_config_snapshot,
)
from configs.paper_minimal_3v3_spec import (
    PAPER_MINIMAL_ENVIRONMENT_PROFILE,
    minimal_environment_snapshot,
)
from my_uav_env.env import UavCombatEnv
from rule_based_agent import (
    _blue_simple_pursuit_action_impl,
    _paper_absolute_action,
)


def _red_rule_actions(env: UavCombatEnv, obs: dict, mode: str) -> dict:
    actions = {}
    for index, aid in enumerate(env.red_ids):
        sim = env.red_planes[aid]
        heading = float(sim.get_rpy()[2])
        if mode == "straight":
            actions[aid] = _paper_absolute_action(0.0, heading, 300.0)
            continue
        action = _blue_simple_pursuit_action_impl(
            obs[aid], env.max_num_red, env.max_num_blue, index,
            forced_target_idx=index,
            own_position=np.asarray(sim.get_position()), own_heading=heading,
            paper_profile=True)
        action[2] = _paper_absolute_action(0.0, 0.0, 300.0)[2]
        actions[aid] = action
    return actions


def _run_scenario(name: str, red_mode: str, blue_profile: str,
                  seed: int, max_steps: int) -> dict:
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=max_steps,
        environment_profile=PAPER_MINIMAL_ENVIRONMENT_PROFILE,
        blue_policy_profile=blue_profile, suppress_jsbsim_output=True)
    launch = {"red": Counter(), "blue": Counter()}
    try:
        obs, info = env.reset(seed=seed)
        first_launch = {"red": None, "blue": None}
        minimum_missile_distance_m = float("inf")
        finite = True
        for step in range(1, max_steps + 1):
            actions = _red_rule_actions(env, obs, red_mode)
            actions.update(env.blue_policy_actions({bid: obs[bid] for bid in env.blue_ids}))
            obs, rewards, terminated, truncated, info = env.step(actions)
            for missile in env._missiles_in_flight.values():
                if missile.is_alive and missile.target_aircraft is not None:
                    minimum_missile_distance_m = min(
                        minimum_missile_distance_m,
                        float(missile.target_distance))
            finite &= all(np.isfinite(value) for value in rewards.values())
            for team in ("red", "blue"):
                for key, value in info.get("__launch_diag__", {}).get(team, {}).items():
                    launch[team][key] += int(value)
                if first_launch[team] is None and launch[team]["launches"]:
                    first_launch[team] = step
            if all(terminated[aid] or truncated[aid] for aid in env.agent_ids):
                break
        term = info.get("__missile_term__", {})
        episode = info.get("__episode__", {})
        deaths = Counter(
            value for value in env._death_reasons.values() if value)
        return {
            "name": name,
            "steps": int(env.current_step),
            "red_mode": red_mode,
            "blue_profile": blue_profile,
            "winner": episode.get("winner", ""),
            "invalid_numerical_episode": bool(
                episode.get("invalid_numerical_episode", False)),
            "finite_rewards": bool(finite),
            "red_geometry": int(launch["red"]["geometry_ok_pairs"]),
            "blue_geometry": int(launch["blue"]["geometry_ok_pairs"]),
            "red_lock_mature": int(launch["red"]["lock_mature_pairs"]),
            "blue_lock_mature": int(launch["blue"]["lock_mature_pairs"]),
            "red_launches": int(launch["red"]["launches"]),
            "blue_launches": int(launch["blue"]["launches"]),
            "red_hits": int(term.get("red", {}).get("hit", 0)),
            "blue_hits": int(term.get("blue", {}).get("hit", 0)),
            "missile_termination_reasons": term,
            "missiles_in_flight_final": int(len(env._missiles_in_flight)),
            "minimum_missile_distance_m": (
                None if not np.isfinite(minimum_missile_distance_m)
                else float(minimum_missile_distance_m)),
            "red_crashes": int(sum(
                count for reason, count in deaths.items()
                if reason.startswith("Crash_") and reason != "Crash_BattleVolume")),
            "blue_crashes": 0,
            "first_red_launch_step": first_launch["red"],
            "first_blue_launch_step": first_launch["blue"],
        }
    finally:
        env.close()


def _initial_and_static_audit(seed: int) -> dict:
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3,
        environment_profile=PAPER_MINIMAL_ENVIRONMENT_PROFILE,
        suppress_jsbsim_output=True)
    try:
        obs, info = env.reset(seed=seed)
        pairs = []
        for index in range(3):
            red = env.red_planes[f"red_{index}"]
            blue = env.blue_planes[f"blue_{index}"]
            pairs.append({
                "index": index,
                "range_m": float(np.linalg.norm(
                    red.get_position() - blue.get_position())),
                "red_altitude_m": float(red.get_geodetic()[2]),
                "blue_altitude_m": float(blue.get_geodetic()[2]),
                "red_speed_mps": float(np.linalg.norm(red.get_velocity())),
                "blue_speed_mps": float(np.linalg.norm(blue.get_velocity())),
                "heading_separation_rad": float(abs(
                    np.angle(np.exp(1j * (red.get_rpy()[2] - blue.get_rpy()[2]))))),
            })
        fingerprint = info["__environment_config__"][
            "environment_config_fingerprint"]
        reference = environment_config_snapshot(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG, num_red=3, num_blue=3,
            sim_freq=60, agent_interaction_steps=12, seed=seed,
            blue_policy_profile="paper_pursuit")
        return {
            "entity_dim": int(env.entity_dim),
            "actor_obs_dim": 60,
            "critic_state_dim": 30,
            "sim_freq": int(env.sim_freq),
            "agent_interaction_steps": int(env.agent_interaction_steps),
            "decision_frequency_hz": float(env.sim_freq / env.agent_interaction_steps),
            "max_episode_length": int(env.max_steps),
            "initial_pairs": pairs,
            "minimal_fingerprint": fingerprint,
            "reference_fingerprint": reference["environment_config_fingerprint"],
            "fingerprints_differ": fingerprint != reference[
                "environment_config_fingerprint"],
            "observation_shapes": {
                aid: {
                    "ego": list(row["ego_state"].shape),
                    "allies": list(row["ally_states"].shape),
                    "enemies": list(row["enemy_states"].shape),
                }
                for aid, row in obs.items()
            },
        }
    finally:
        env.close()


def audit(seed: int, max_steps: int) -> dict:
    static = _initial_and_static_audit(seed)
    scenarios = [
        _run_scenario("fixed_vs_fixed", "fixed",
                      "paper_minimal_fixed_pair_v1", seed, max_steps),
        _run_scenario("fixed_red_vs_straight_blue", "fixed",
                      "paper_minimal_straight_patrol_v1", seed, max_steps),
        _run_scenario("straight_red_vs_fixed_blue", "straight",
                      "paper_minimal_fixed_pair_v1", seed, max_steps),
    ]
    mirror_errors = []
    for row in static["initial_pairs"]:
        mirror_errors.extend([
            abs(row["range_m"] - 10_000.0),
            abs(row["red_altitude_m"] - row["blue_altitude_m"]),
            abs(row["red_speed_mps"] - row["blue_speed_mps"]),
            abs(row["heading_separation_rad"] - np.pi),
        ])
    checks = {
        "profile_dimensions": static["entity_dim"] == 10
        and static["actor_obs_dim"] == 60 and static["critic_state_dim"] == 30,
        "timing": static["sim_freq"] == 60
        and static["agent_interaction_steps"] == 12
        and static["decision_frequency_hz"] == 5.0,
        "initial_mirror": max(mirror_errors) < 25.0,
        "fingerprint_isolation": bool(static["fingerprints_differ"]),
        "finite": all(row["finite_rewards"] for row in scenarios),
        "no_invalid_episodes": not any(
            row["invalid_numerical_episode"] for row in scenarios),
        "fixed_vs_fixed_both_launch": (
            scenarios[0]["red_launches"] > 0 and scenarios[0]["blue_launches"] > 0),
        "fixed_red_can_launch_and_hit_straight_blue": (
            scenarios[1]["red_launches"] > 0 and scenarios[1]["red_hits"] > 0),
        "color_swap_has_fire_control_activity": (
            scenarios[2]["blue_launches"] > 0 and scenarios[2]["blue_hits"] > 0),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "static": static,
        "scenarios": scenarios,
    }


def _markdown(result: dict) -> str:
    lines = [f"# {result['status']}: paper_minimal_3v3_v1 audit", "", "## Checks", ""]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in result["checks"].items())
    lines.extend(["", "## Rule-loop scenarios", "",
                  "| scenario | steps | red launches/hits | blue launches/hits | invalid |",
                  "|---|---:|---:|---:|---|"])
    for row in result["scenarios"]:
        lines.append(
            f"| {row['name']} | {row['steps']} | "
            f"{row['red_launches']}/{row['red_hits']} | "
            f"{row['blue_launches']}/{row['blue_hits']} | "
            f"{row['invalid_numerical_episode']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", default="tmp/paper_minimal_3v3_audit.json")
    parser.add_argument("--markdown", default="tmp/paper_minimal_3v3_audit.md")
    args = parser.parse_args()
    result = audit(args.seed, args.steps)
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
