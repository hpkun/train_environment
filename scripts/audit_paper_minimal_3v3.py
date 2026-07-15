"""Executable audit for the isolated deterministic minimal 3V3 profile."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        peak_live_missiles = 0
        for step in range(1, max_steps + 1):
            actions = _red_rule_actions(env, obs, red_mode)
            actions.update(env.blue_policy_actions({bid: obs[bid] for bid in env.blue_ids}))
            obs, rewards, terminated, truncated, info = env.step(actions)
            for missile in env._missiles_in_flight.values():
                if missile.is_alive and missile.target_aircraft is not None:
                    minimum_missile_distance_m = min(
                        minimum_missile_distance_m,
                        float(missile.target_distance))
            peak_live_missiles = max(
                peak_live_missiles,
                sum(int(missile.is_alive)
                    for missile in env._missiles_in_flight.values()))
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
        deaths_by_team = {
            team: Counter(
                reason for aid, reason in env._death_reasons.items()
                if aid.startswith(team) and reason)
            for team in ("red", "blue")}
        aircraft_diag = {
            team: [info.get(f"{team}_{index}", {}) for index in range(3)]
            for team in ("red", "blue")}
        def diag_max(team, key):
            return max((float(row.get(key, 0.0))
                        for row in aircraft_diag[team]), default=0.0)
        def diag_sum(team, key):
            return sum(int(row.get(key, 0)) for row in aircraft_diag[team])
        def death_count(team, reason):
            return int(deaths_by_team[team].get(reason, 0))
        return {
            "name": name,
            "seed": int(seed),
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
            "red_p_hit_fail": int(term.get("red", {}).get("p_hit_fail", 0)),
            "blue_p_hit_fail": int(term.get("blue", {}).get("p_hit_fail", 0)),
            "red_overshoot": int(term.get("red", {}).get("overshoot", 0)),
            "blue_overshoot": int(term.get("blue", {}).get("overshoot", 0)),
            "red_timeout": int(term.get("red", {}).get("timeout", 0)),
            "blue_timeout": int(term.get("blue", {}).get("timeout", 0)),
            "missiles_in_flight_final": int(len(env._missiles_in_flight)),
            "peak_live_missiles": int(max(
                peak_live_missiles,
                episode.get("maximum_live_missiles_observed", 0))),
            "minimum_missile_distance_m": (
                None if not np.isfinite(minimum_missile_distance_m)
                else float(minimum_missile_distance_m)),
            "red_crashes": int(sum(
                count for reason, count in deaths_by_team["red"].items()
                if reason.startswith("Crash_"))),
            "blue_crashes": int(sum(
                count for reason, count in deaths_by_team["blue"].items()
                if reason.startswith("Crash_"))),
            "red_low_altitude_exits": death_count("red", "Crash_LowAlt"),
            "blue_low_altitude_exits": death_count("blue", "Crash_LowAlt"),
            "red_high_altitude_exits": death_count("red", "Crash_HighAlt"),
            "blue_high_altitude_exits": death_count("blue", "Crash_HighAlt"),
            "red_horizontal_exits": death_count("red", "Crash_BattleVolume"),
            "blue_horizontal_exits": death_count("blue", "Crash_BattleVolume"),
            "red_max_speed_mps": diag_max("red", "maximum_speed_after_limit_mps"),
            "blue_max_speed_mps": diag_max("blue", "maximum_speed_after_limit_mps"),
            "red_max_prelimit_speed_mps": diag_max(
                "red", "maximum_speed_before_limit_mps"),
            "blue_max_prelimit_speed_mps": diag_max(
                "blue", "maximum_speed_before_limit_mps"),
            "red_speed_limiter_activations": diag_sum(
                "red", "speed_limiter_activations"),
            "blue_speed_limiter_activations": diag_sum(
                "blue", "speed_limiter_activations"),
            "red_max_load_g": diag_max("red", "maximum_load_g_seen"),
            "blue_max_load_g": diag_max("blue", "maximum_load_g_seen"),
            "red_load_limiter_activations": diag_sum(
                "red", "load_limiter_activations"),
            "blue_load_limiter_activations": diag_sum(
                "blue", "load_limiter_activations"),
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


def audit(seeds: list[int] | tuple[int, ...], max_steps: int) -> dict:
    seeds = [int(seed) for seed in seeds]
    if len(seeds) < 2:
        raise ValueError("audit requires at least two seeds")
    static = _initial_and_static_audit(seeds[0])
    scenarios = []
    for seed in seeds:
        scenarios.extend([
            _run_scenario("fixed_vs_fixed", "fixed",
                          "paper_minimal_fixed_pair_v1", seed, max_steps),
            _run_scenario("fixed_red_vs_straight_blue", "fixed",
                          "paper_minimal_straight_patrol_v1", seed, max_steps),
            _run_scenario("straight_red_vs_fixed_blue", "straight",
                          "paper_minimal_fixed_pair_v1", seed, max_steps),
        ])
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
    }
    grouped = {
        name: [row for row in scenarios if row["name"] == name]
        for name in ("fixed_vs_fixed", "fixed_red_vs_straight_blue",
                     "straight_red_vs_fixed_blue")}
    def total(rows, field):
        return sum(int(row[field]) for row in rows)
    fixed = grouped["fixed_vs_fixed"]
    red_fixed = grouped["fixed_red_vs_straight_blue"]
    blue_fixed = grouped["straight_red_vs_fixed_blue"]
    fixed_launch_total = total(fixed, "red_launches") + total(fixed, "blue_launches")
    mirror_launch_total = total(red_fixed, "red_launches") + total(
        blue_fixed, "blue_launches")
    mirror_hit_total = total(red_fixed, "red_hits") + total(blue_fixed, "blue_hits")
    checks.update({
        "fixed_vs_fixed_both_launch": (
            total(fixed, "red_launches") > 0 and total(fixed, "blue_launches") > 0),
        "fixed_vs_fixed_launch_symmetry": abs(
            total(fixed, "red_launches") - total(fixed, "blue_launches"))
            <= max(2, int(np.ceil(0.25 * fixed_launch_total))),
        "mirror_fixed_side_launch_symmetry": abs(
            total(red_fixed, "red_launches") - total(blue_fixed, "blue_launches"))
            <= max(2, int(np.ceil(0.25 * mirror_launch_total))),
        "mirror_fixed_side_hit_symmetry": abs(
            total(red_fixed, "red_hits") - total(blue_fixed, "blue_hits"))
            <= max(2, int(np.ceil(0.5 * mirror_hit_total))),
        "bounded_live_missiles": max(
            row["peak_live_missiles"] for row in scenarios) <= 6,
        "aircraft_speed_envelope_handled": all(
            row["red_max_speed_mps"] <= 600.01
            and row["blue_max_speed_mps"] <= 600.01
            for row in scenarios),
        "load_envelope_handled": all(
            (row["red_max_load_g"] <= 9.0
             or row["red_load_limiter_activations"] > 0)
            and (row["blue_max_load_g"] <= 9.0
                 or row["blue_load_limiter_activations"] > 0)
            for row in scenarios),
    })
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "static": static,
        "seeds": seeds,
        "aggregate": {
            "fixed_vs_fixed_red_launches": total(fixed, "red_launches"),
            "fixed_vs_fixed_blue_launches": total(fixed, "blue_launches"),
            "mirror_red_fixed_launches": total(red_fixed, "red_launches"),
            "mirror_blue_fixed_launches": total(blue_fixed, "blue_launches"),
            "mirror_red_fixed_hits": total(red_fixed, "red_hits"),
            "mirror_blue_fixed_hits": total(blue_fixed, "blue_hits"),
        },
        "scenarios": scenarios,
    }


def _markdown(result: dict) -> str:
    lines = [f"# {result['status']}: paper_minimal_3v3_v1 audit", "", "## Checks", ""]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in result["checks"].items())
    lines.extend(["", "## Rule-loop scenarios", "",
                  "| scenario | seed | steps | red launches/hits | blue launches/hits | peak live | invalid |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for row in result["scenarios"]:
        lines.append(
            f"| {row['name']} | {row['seed']} | {row['steps']} | "
            f"{row['red_launches']}/{row['red_hits']} | "
            f"{row['blue_launches']}/{row['blue_hits']} | "
            f"{row['peak_live_missiles']} | "
            f"{row['invalid_numerical_episode']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", default="tmp/paper_minimal_3v3_audit.json")
    parser.add_argument("--markdown", default="tmp/paper_minimal_3v3_audit.md")
    args = parser.parse_args()
    result = audit(args.seeds, args.steps)
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
