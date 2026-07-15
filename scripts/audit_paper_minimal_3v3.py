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


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return True


def _run_scenario(name: str, red_mode: str, blue_profile: str,
                  seed: int, max_steps: int) -> dict:
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=max_steps,
        environment_profile=PAPER_MINIMAL_ENVIRONMENT_PROFILE,
        blue_policy_profile=blue_profile, suppress_jsbsim_output=True)
    launch = {"red": Counter(), "blue": Counter()}
    try:
        obs, info = env.reset(seed=seed)
        env.set_team_mws_enabled("red", red_mode != "straight")
        env.set_team_mws_enabled(
            "blue", blue_profile != "paper_minimal_straight_patrol_v1")
        first_launch = {"red": None, "blue": None}
        minimum_missile_distance_m = float("inf")
        finite = True
        peak_live_missiles = 0
        completed_missiles: list[dict] = []
        for step in range(1, max_steps + 1):
            actions = _red_rule_actions(env, obs, red_mode)
            actions.update(env.blue_policy_actions({bid: obs[bid] for bid in env.blue_ids}))
            finite &= _all_finite(actions)
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
            dynamic_info = {
                key: value for key, value in info.items()
                if key != "__environment_config__"
            }
            finite &= (
                _all_finite(obs)
                and _all_finite(rewards)
                and _all_finite(dynamic_info)
            )
            completed_missiles.extend(
                row for row in info.get("__launch_quality_done__", [])
                if isinstance(row, dict))
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
        def missile_lifetimes(team):
            values = [float(row.get("flight_time_sec"))
                      for row in completed_missiles
                      if row.get("team") == team]
            return [value for value in values if np.isfinite(value)]
        def rate(num, den):
            return float(num) / float(den) if den else 0.0
        red_lifetimes = missile_lifetimes("red")
        blue_lifetimes = missile_lifetimes("blue")
        physics_frames = max(int(env.current_step) * env.agent_interaction_steps, 1)
        red_launches = int(launch["red"]["launches"])
        blue_launches = int(launch["blue"]["launches"])
        red_hits = int(term.get("red", {}).get("hit", 0))
        blue_hits = int(term.get("blue", {}).get("hit", 0))
        red_overshoot = int(term.get("red", {}).get("overshoot", 0))
        blue_overshoot = int(term.get("blue", {}).get("overshoot", 0))
        return {
            "name": name,
            "seed": int(seed),
            "steps": int(env.current_step),
            "red_mode": red_mode,
            "blue_profile": blue_profile,
            "winner": episode.get("winner", ""),
            "invalid_numerical_episode": bool(
                episode.get("invalid_numerical_episode", False)),
            "finite_state_action_reward_log": bool(finite),
            "red_geometry": int(launch["red"]["geometry_ok_pairs"]),
            "blue_geometry": int(launch["blue"]["geometry_ok_pairs"]),
            "red_lock_mature": int(launch["red"]["lock_mature_pairs"]),
            "blue_lock_mature": int(launch["blue"]["lock_mature_pairs"]),
            "red_launches": red_launches,
            "blue_launches": blue_launches,
            "red_hits": red_hits,
            "blue_hits": blue_hits,
            "missile_termination_reasons": term,
            "red_p_hit_fail": int(term.get("red", {}).get("p_hit_fail", 0)),
            "blue_p_hit_fail": int(term.get("blue", {}).get("p_hit_fail", 0)),
            "red_overshoot": red_overshoot,
            "blue_overshoot": blue_overshoot,
            "red_timeout": int(term.get("red", {}).get("timeout", 0)),
            "blue_timeout": int(term.get("blue", {}).get("timeout", 0)),
            "red_target_dead": int(term.get("red", {}).get("target_dead", 0)),
            "blue_target_dead": int(term.get("blue", {}).get("target_dead", 0)),
            "red_missile_deaths": int(sum(term.get("red", {}).values())),
            "blue_missile_deaths": int(sum(term.get("blue", {}).values())),
            "red_mean_missile_lifetime_s": (
                float(np.mean(red_lifetimes)) if red_lifetimes else None),
            "blue_mean_missile_lifetime_s": (
                float(np.mean(blue_lifetimes)) if blue_lifetimes else None),
            "red_median_missile_lifetime_s": (
                float(np.median(red_lifetimes)) if red_lifetimes else None),
            "blue_median_missile_lifetime_s": (
                float(np.median(blue_lifetimes)) if blue_lifetimes else None),
            "red_launch_to_hit_rate": rate(red_hits, red_launches),
            "blue_launch_to_hit_rate": rate(blue_hits, blue_launches),
            "red_overshoot_to_launch_rate": rate(red_overshoot, red_launches),
            "blue_overshoot_to_launch_rate": rate(blue_overshoot, blue_launches),
            "red_launches_per_episode": float(red_launches),
            "blue_launches_per_episode": float(blue_launches),
            "red_launches_per_1000_environment_steps": (
                1000.0 * red_launches / max(env.current_step, 1)),
            "blue_launches_per_1000_environment_steps": (
                1000.0 * blue_launches / max(env.current_step, 1)),
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
            "red_speed_limiter_rate_per_1000_physics_steps": (
                1000.0 * diag_sum("red", "speed_limiter_activations")
                / physics_frames),
            "blue_speed_limiter_rate_per_1000_physics_steps": (
                1000.0 * diag_sum("blue", "speed_limiter_activations")
                / physics_frames),
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


def _relative_difference(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        (abs(float(a)) + abs(float(b))) / 2.0, 1.0)


def evaluate_rule_audit(static: dict, scenarios: list[dict]) -> dict:
    mirror_errors = []
    for row in static["initial_pairs"]:
        mirror_errors.extend([
            abs(row["range_m"] - 10_000.0),
            abs(row["red_altitude_m"] - row["blue_altitude_m"]),
            abs(row["red_speed_mps"] - row["blue_speed_mps"]),
            abs(row["heading_separation_rad"] - np.pi),
        ])
    grouped = {
        name: [row for row in scenarios if row["name"] == name]
        for name in ("fixed_vs_fixed", "fixed_red_vs_straight_blue",
                     "straight_red_vs_fixed_blue")}
    def total(rows, field):
        return sum(float(row.get(field, 0)) for row in rows)
    def mean_present(rows, field):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return float(np.mean(values)) if values else 0.0
    fixed = grouped["fixed_vs_fixed"]
    red_fixed = grouped["fixed_red_vs_straight_blue"]
    blue_fixed = grouped["straight_red_vs_fixed_blue"]
    fixed_red_hits = total(fixed, "red_hits")
    fixed_blue_hits = total(fixed, "blue_hits")
    mirror_metrics = {
        "geometry_ok": (
            total(red_fixed, "red_geometry"), total(blue_fixed, "blue_geometry"), 0.20),
        "lock_mature": (
            total(red_fixed, "red_lock_mature"), total(blue_fixed, "blue_lock_mature"), 0.20),
        "launches": (
            total(red_fixed, "red_launches"), total(blue_fixed, "blue_launches"), 0.20),
        "hits": (
            total(red_fixed, "red_hits"), total(blue_fixed, "blue_hits"), 0.30),
        "first_launch_step": (
            mean_present(red_fixed, "first_red_launch_step"),
            mean_present(blue_fixed, "first_blue_launch_step"), 0.15),
    }
    mirror_checks = {
        f"mirror_{name}": (
            a > 0 and b > 0 and _relative_difference(a, b) <= tolerance)
        for name, (a, b, tolerance) in mirror_metrics.items()
    }
    mirror_checks["mirror_crash_count"] = abs(
        total(red_fixed, "red_crashes") - total(blue_fixed, "blue_crashes")) <= 2
    numerical_checks = {
        "finite_state_action_reward_log": all(
            row["finite_state_action_reward_log"] for row in scenarios),
        "no_invalid_episodes": not any(
            row["invalid_numerical_episode"] for row in scenarios),
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
    }
    rule_checks = {
        "profile_dimensions": static["entity_dim"] == 10
        and static["actor_obs_dim"] == 60 and static["critic_state_dim"] == 30,
        "timing": static["sim_freq"] == 60
        and static["agent_interaction_steps"] == 12
        and static["decision_frequency_hz"] == 5.0,
        "initial_mirror": max(mirror_errors) < 25.0,
        "fingerprint_isolation": bool(static["fingerprints_differ"]),
        "fixed_vs_fixed_red_fire_control": (
            total(fixed, "red_geometry") > 0
            and total(fixed, "red_lock_mature") > 0
            and total(fixed, "red_launches") > 0 and fixed_red_hits > 0),
        "fixed_vs_fixed_blue_fire_control": (
            total(fixed, "blue_geometry") > 0
            and total(fixed, "blue_lock_mature") > 0
            and total(fixed, "blue_launches") > 0 and fixed_blue_hits > 0),
        "fixed_red_vs_straight_blue_effective": (
            total(red_fixed, "red_launches") > 0
            and total(red_fixed, "red_hits") > 0),
        "straight_red_vs_fixed_blue_effective": (
            total(blue_fixed, "blue_launches") > 0
            and total(blue_fixed, "blue_hits") > 0),
        "bounded_live_missiles": max(
            row["peak_live_missiles"] for row in scenarios) <= 6,
    }
    missile_checks = {
        "nonzero_fixed_hits": (
            fixed_red_hits > 0 and fixed_blue_hits > 0
            and total(red_fixed, "red_hits") > 0
            and total(blue_fixed, "blue_hits") > 0),
        "missile_deaths_accounted": all(
            row["red_missile_deaths"] <= row["red_launches"]
            and row["blue_missile_deaths"] <= row["blue_launches"]
            for row in scenarios),
    }
    categories = {
        "rule_environment_health": rule_checks,
        "mirror_symmetry": mirror_checks,
        "missile_effectiveness": missile_checks,
        "numerical_stability": numerical_checks,
    }
    checks = {
        f"{category}.{name}": passed
        for category, values in categories.items()
        for name, passed in values.items()}
    aggregate = {
        "fixed_vs_fixed_red_launches": total(fixed, "red_launches"),
        "fixed_vs_fixed_blue_launches": total(fixed, "blue_launches"),
        "fixed_vs_fixed_red_hits": fixed_red_hits,
        "fixed_vs_fixed_blue_hits": fixed_blue_hits,
        "mirror_red_fixed_geometry": total(red_fixed, "red_geometry"),
        "mirror_blue_fixed_geometry": total(blue_fixed, "blue_geometry"),
        "mirror_red_fixed_lock_mature": total(red_fixed, "red_lock_mature"),
        "mirror_blue_fixed_lock_mature": total(blue_fixed, "blue_lock_mature"),
        "mirror_red_fixed_launches": total(red_fixed, "red_launches"),
        "mirror_blue_fixed_launches": total(blue_fixed, "blue_launches"),
        "mirror_red_fixed_hits": total(red_fixed, "red_hits"),
        "mirror_blue_fixed_hits": total(blue_fixed, "blue_hits"),
        "total_red_launches": total(scenarios, "red_launches"),
        "total_blue_launches": total(scenarios, "blue_launches"),
        "total_red_hits": total(scenarios, "red_hits"),
        "total_blue_hits": total(scenarios, "blue_hits"),
        "total_red_overshoots": total(scenarios, "red_overshoot"),
        "total_blue_overshoots": total(scenarios, "blue_overshoot"),
        "mean_red_missile_lifetime_s": mean_present(
            scenarios, "red_mean_missile_lifetime_s"),
        "mean_blue_missile_lifetime_s": mean_present(
            scenarios, "blue_mean_missile_lifetime_s"),
    }
    aggregate["total_red_launch_to_hit_rate"] = (
        aggregate["total_red_hits"] / aggregate["total_red_launches"]
        if aggregate["total_red_launches"] else 0.0)
    aggregate["total_blue_launch_to_hit_rate"] = (
        aggregate["total_blue_hits"] / aggregate["total_blue_launches"]
        if aggregate["total_blue_launches"] else 0.0)
    aggregate["total_red_overshoot_to_launch_rate"] = (
        aggregate["total_red_overshoots"] / aggregate["total_red_launches"]
        if aggregate["total_red_launches"] else 0.0)
    aggregate["total_blue_overshoot_to_launch_rate"] = (
        aggregate["total_blue_overshoots"] / aggregate["total_blue_launches"]
        if aggregate["total_blue_launches"] else 0.0)
    for name, (a, b, _tolerance) in mirror_metrics.items():
        aggregate[f"mirror_{name}_relative_difference"] = _relative_difference(a, b)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "categories": categories,
        "static": static,
        "aggregate": aggregate,
        "scenarios": scenarios,
    }


def audit(seeds: list[int] | tuple[int, ...], max_steps: int) -> dict:
    seeds = [int(seed) for seed in seeds]
    if len(seeds) < 5:
        raise ValueError("audit requires at least five seeds")
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
    result = evaluate_rule_audit(static, scenarios)
    result["seeds"] = seeds
    return result


def _markdown(result: dict) -> str:
    lines = [f"# {result['status']}: paper_minimal_3v3_v1 audit"]
    for category, checks in result["categories"].items():
        status = "PASS" if all(checks.values()) else "FAIL"
        lines.extend(["", f"## {category}: {status}", ""])
        lines.extend(
            f"- {'PASS' if passed else 'FAIL'}: `{name}`"
            for name, passed in checks.items())
    aggregate = result["aggregate"]
    lines.extend([
        "", "## Missile effectiveness", "",
        "| team | launches | hits | hit rate | overshoots | overshoot rate | mean lifetime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| red | {aggregate['total_red_launches']:.0f} | "
        f"{aggregate['total_red_hits']:.0f} | "
        f"{aggregate['total_red_launch_to_hit_rate']:.6f} | "
        f"{aggregate['total_red_overshoots']:.0f} | "
        f"{aggregate['total_red_overshoot_to_launch_rate']:.6f} | "
        f"{aggregate['mean_red_missile_lifetime_s']:.3f} |",
        f"| blue | {aggregate['total_blue_launches']:.0f} | "
        f"{aggregate['total_blue_hits']:.0f} | "
        f"{aggregate['total_blue_launch_to_hit_rate']:.6f} | "
        f"{aggregate['total_blue_overshoots']:.0f} | "
        f"{aggregate['total_blue_overshoot_to_launch_rate']:.6f} | "
        f"{aggregate['mean_blue_missile_lifetime_s']:.3f} |",
    ])
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
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
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
