"""Full five-seed rule-combat closure audit for paper_learnable_3v3_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_learnable_3v3_spec import (
    LEARNABLE_MISSILE_HIT_RADIUS_M,
    PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
)
from my_uav_env import UavCombatEnv
from my_uav_env.blue_policy_profiles import BluePolicyController
from scripts.audit_paper_learnable_missile_trajectory import run_audit as run_trajectory_audit


SEEDS = (3, 7, 11, 17, 23)
SCENARIOS = (
    "red_fixed_vs_blue_straight",
    "blue_fixed_vs_red_straight",
    "fixed_vs_fixed_symmetric",
)


def _straight_action(env: UavCombatEnv, heading: float) -> np.ndarray:
    speed = 2.0 * (300.0 - env.VELOCITY_MIN) / (
        env.VELOCITY_MAX - env.VELOCITY_MIN) - 1.0
    return np.asarray([0.0, heading / np.pi, speed], dtype=np.float32)


def _red_fixed_actions(
    env: UavCombatEnv, obs: dict, controller: BluePolicyController,
) -> dict[str, np.ndarray]:
    env._refresh_learnable_assignments(env.red_ids, count_wait=False)
    virtual_obs = {f"blue_{i}": obs[f"red_{i}"] for i in range(3)}
    virtual_engaged = {
        f"red_{int(target.split('_', 1)[1])}"
        for target in env.refresh_engaged_targets()
        if target.startswith("blue_")
    }
    assigned = {}
    for index in range(3):
        actual = env._fire_control_assignments.get(f"red_{index}")
        assigned[f"blue_{index}"] = (
            None if actual is None
            else f"red_{int(actual.split('_', 1)[1])}")
    virtual_actions = controller.act(
        virtual_obs, 3, 3, virtual_engaged,
        {f"blue_{i}": env.red_planes[f"red_{i}"].get_position()
         for i in range(3)},
        {f"blue_{i}": float(env.red_planes[f"red_{i}"].get_rpy()[2])
         for i in range(3)},
        env.current_step,
        own_alive={f"blue_{i}": bool(env.red_planes[f"red_{i}"].is_alive)
                   for i in range(3)},
        assigned_targets=assigned,
    )
    return {f"red_{i}": virtual_actions[f"blue_{i}"] for i in range(3)}


def _initial_mirror_error(env: UavCombatEnv) -> float:
    errors = []
    for index in range(3):
        blue = np.asarray(env.blue_planes[f"blue_{index}"].get_position())
        red = np.asarray(env.red_planes[f"red_{index}"].get_position())
        errors.extend([abs(blue[0] - red[0]), abs(blue[1] + red[1]),
                       abs(blue[2] - red[2])])
    return float(max(errors, default=0.0))


def _finite_obs(obs: dict) -> bool:
    return all(
        np.all(np.isfinite(np.asarray(value)))
        for agent_obs in obs.values()
        for value in agent_obs.values()
        if isinstance(value, (list, tuple, np.ndarray))
    )


def _circular_error(a: float, b: float) -> float:
    return abs(float((a - b + np.pi) % (2.0 * np.pi) - np.pi))


def _mirror_dynamics_errors(env: UavCombatEnv, actions: dict, info: dict) -> dict:
    result = {
        "position_m": 0.0, "velocity_mps": 0.0,
        "attitude_rad": 0.0, "speed_mps": 0.0,
        "action": 0.0, "setpoint": 0.0, "g_load": 0.0,
        "reward_component": 0.0, "load_scale": 0.0,
    }
    for index in range(3):
        blue_id, red_id = f"blue_{index}", f"red_{index}"
        blue, red = env.blue_planes[blue_id], env.red_planes[red_id]
        bp, rp = np.asarray(blue.get_position()), np.asarray(red.get_position())
        bv, rv = np.asarray(blue.get_velocity()), np.asarray(red.get_velocity())
        br, rr = np.asarray(blue.get_rpy()), np.asarray(red.get_rpy())
        result["position_m"] = max(result["position_m"], float(np.max(
            np.abs(bp - np.array([rp[0], -rp[1], rp[2]])))))
        result["velocity_mps"] = max(result["velocity_mps"], float(np.max(
            np.abs(bv - np.array([rv[0], -rv[1], rv[2]])))))
        result["speed_mps"] = max(result["speed_mps"], abs(
            float(np.linalg.norm(bv) - np.linalg.norm(rv))))
        result["attitude_rad"] = max(
            result["attitude_rad"], _circular_error(br[0], -rr[0]),
            _circular_error(br[1], rr[1]), _circular_error(br[2], -rr[2]))
        ba, ra = np.asarray(actions[blue_id]), np.asarray(actions[red_id])
        result["action"] = max(result["action"], float(np.max(np.abs(
            ba - np.array([ra[0], -ra[1], ra[2]])))))
        bs = env._learnable_setpoint_state.get(blue_id)
        rs = env._learnable_setpoint_state.get(red_id)
        if bs is not None and rs is not None:
            result["setpoint"] = max(
                result["setpoint"], abs(bs[0] - rs[0]),
                _circular_error(bs[1], -rs[1]), abs(bs[2] - rs[2]))
        bd, rd = info.get(blue_id, {}), info.get(red_id, {})
        result["g_load"] = max(result["g_load"], abs(float(
            bd.get("maximum_load_g_seen", 0.0)) - float(
            rd.get("maximum_load_g_seen", 0.0))))
        result["load_scale"] = max(result["load_scale"], abs(float(
            bd.get("load_protection_minimum_scale", 1.0)) - float(
            rd.get("load_protection_minimum_scale", 1.0))))
        for key in ("r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv"):
            result["reward_component"] = max(
                result["reward_component"], abs(float(bd.get(key, 0.0))
                                                  - float(rd.get(key, 0.0))))
    return result


def run_case(seed: int, scenario: str, max_steps: int = 1400) -> dict:
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
        suppress_jsbsim_output=True)
    try:
        obs, info = env.reset(seed=seed)
        pure_dynamics = scenario == "fixed_vs_fixed_symmetric"
        if pure_dynamics:
            env.set_team_mws_enabled("red", False)
            env.set_team_mws_enabled("blue", False)
            env._check_missile_launch = lambda: None
            max_steps = min(max_steps, 300)
        initial_headings = {
            aid: float(env._get_sim(aid).get_rpy()[2]) for aid in env.agent_ids}
        red_controller = BluePolicyController("paper_learnable_fixed_pair_v1")
        red_controller.reset(
            [f"blue_{i}" for i in range(3)], [f"red_{i}" for i in range(3)],
            {f"blue_{i}": initial_headings[f"red_{i}"] for i in range(3)},
            {f"blue_{i}": float(env.red_planes[f"red_{i}"].get_geodetic()[2])
             for i in range(3)})

        first_geometry = {"red": None, "blue": None}
        first_lock = {"red": None, "blue": None}
        first_launch = {"red": None, "blue": None}
        first_launch_record = {"red": None, "blue": None}
        launches = {"red": 0, "blue": 0}
        all_launch_records: list[dict] = []
        all_done_records: list[dict] = []
        finite = _finite_obs(obs)
        movement_matches = True
        initial_mirror_error = _initial_mirror_error(env)
        maximum_mirror_errors = {
            key: 0.0 for key in (
                "position_m", "velocity_mps", "attitude_rad", "speed_mps",
                "action", "setpoint", "g_load", "reward_component", "load_scale")}
        final_mirror_errors = dict(maximum_mirror_errors)

        for _ in range(max_steps):
            actions: dict[str, np.ndarray] = {}
            if scenario in (
                    "blue_fixed_vs_red_straight",
                    "fixed_vs_fixed_symmetric"):
                actions.update(env.blue_policy_actions(
                    {aid: obs[aid] for aid in env.blue_ids}))
            else:
                actions.update({
                    aid: _straight_action(env, initial_headings[aid])
                    for aid in env.blue_ids})

            if pure_dynamics:
                for index in range(3):
                    blue_action = actions[f"blue_{index}"]
                    actions[f"red_{index}"] = np.asarray([
                        blue_action[0], -blue_action[1], blue_action[2]],
                        dtype=np.float32)
            elif scenario == "red_fixed_vs_blue_straight":
                actions.update(_red_fixed_actions(env, obs, red_controller))
            else:
                actions.update({
                    aid: _straight_action(env, initial_headings[aid])
                    for aid in env.red_ids})

            finite = finite and all(
                np.all(np.isfinite(action)) for action in actions.values())
            obs, rewards, terminated, truncated, info = env.step(actions)
            finite = finite and _finite_obs(obs) and all(
                np.isfinite(float(value)) for value in rewards.values())
            if pure_dynamics:
                errors = _mirror_dynamics_errors(env, actions, info)
                final_mirror_errors = dict(errors)
                maximum_mirror_errors = {
                    key: max(maximum_mirror_errors[key], value)
                    for key, value in errors.items()}
            for team in ("red", "blue"):
                diag = info.get("__launch_diag__", {}).get(team, {})
                if first_geometry[team] is None and diag.get("geometry_ok_pairs", 0):
                    first_geometry[team] = int(env.current_step)
                if first_lock[team] is None and diag.get("lock_mature_pairs", 0):
                    first_lock[team] = int(env.current_step)
                launches[team] += int(diag.get("launches", 0))
            for record in info.get("__launch_quality_step__", []):
                all_launch_records.append(dict(record))
                team = str(record.get("team"))
                if team in first_launch and first_launch[team] is None:
                    first_launch[team] = int(env.current_step)
                    first_launch_record[team] = dict(record)
            all_done_records.extend(
                dict(record) for record in info.get("__launch_quality_done__", []))
            match = info.get("__target_assignment_diag__", {}).get(
                "movement_matches_fire_control", {})
            if scenario in (
                    "blue_fixed_vs_red_straight", "fixed_vs_fixed_symmetric"):
                movement_matches = movement_matches and all(match.values())
            if all(terminated[aid] or truncated[aid] for aid in env.agent_ids):
                break

        hits = {
            team: sum(record.get("team") == team
                      and record.get("raw_termination_reason") == "hit"
                      for record in all_done_records)
            for team in ("red", "blue")
        }
        inside_radius = sum(
            float(record.get("range_m", np.inf)) < LEARNABLE_MISSILE_HIT_RADIUS_M
            for record in all_launch_records)
        one_frame_hits = sum(
            record.get("raw_termination_reason") == "hit"
            and float(record.get("flight_time_sec", np.inf)) <= 1.0 / 60.0 + 1e-9
            for record in all_done_records)
        survived_decision = sum(
            float(record.get("flight_time_sec", 0.0)) >= 12.0 / 60.0
            for record in all_done_records)
        episode = info.get("__episode__", {})
        for team in ("red", "blue"):
            launch = first_launch_record[team]
            if launch is None:
                continue
            terminal = next((
                record for record in all_done_records
                if record.get("missile_id") == launch.get("missile_id")), None)
            if terminal is not None:
                launch["flight_time_sec"] = terminal.get("flight_time_sec")
                launch["pn_guidance_frames"] = terminal.get("pn_guidance_frames")
                launch["raw_termination_reason"] = terminal.get(
                    "raw_termination_reason")
        row = {
            "seed": seed,
            "scenario": scenario,
            "steps": int(env.current_step),
            "finite": bool(finite),
            "invalid_episode": int(episode.get("invalid_numerical_episode", False)),
            "initial_mirror_error_m": initial_mirror_error,
            "movement_matches_fire_control": bool(movement_matches),
            "red_first_geometry_step": first_geometry["red"],
            "blue_first_geometry_step": first_geometry["blue"],
            "red_first_lock_step": first_lock["red"],
            "blue_first_lock_step": first_lock["blue"],
            "red_first_launch_step": first_launch["red"],
            "blue_first_launch_step": first_launch["blue"],
            "red_launches": launches["red"],
            "blue_launches": launches["blue"],
            "red_hits": hits["red"],
            "blue_hits": hits["blue"],
            "launch_inside_hit_radius_count": int(inside_radius),
            "one_physics_frame_hit_count": int(one_frame_hits),
            "missiles_survived_one_decision_count": int(survived_decision),
            "first_launch_records": first_launch_record,
            "mirror_errors": maximum_mirror_errors,
            "final_mirror_errors": final_mirror_errors,
        }
        return row
    finally:
        env.close()


def _closure(rows: list[dict], team: str) -> str:
    return "PASS" if (
        all(row[f"{team}_first_geometry_step"] is not None for row in rows)
        and all(row[f"{team}_first_lock_step"] is not None for row in rows)
        and all(row[f"{team}_launches"] >= 1 for row in rows)
        and sum(row[f"{team}_hits"] for row in rows) >= 1
        and all(row["invalid_episode"] == 0 and row["finite"] for row in rows)
        and all(row["launch_inside_hit_radius_count"] == 0 for row in rows)
        and all(row["one_physics_frame_hit_count"] == 0 for row in rows)
        and sum(row["missiles_survived_one_decision_count"] for row in rows) > 0
    ) else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1400:
        raise ValueError("--steps must be in [1, 1400]")
    rows = [run_case(seed, scenario, args.steps)
            for scenario in SCENARIOS for seed in SEEDS]
    by_scenario = {
        scenario: [row for row in rows if row["scenario"] == scenario]
        for scenario in SCENARIOS}
    mirror_rows = by_scenario["fixed_vs_fixed_symmetric"]
    mirror_health = (
        all(row["initial_mirror_error_m"] <= 1e-6 for row in mirror_rows)
        and all(row["finite"] and row["invalid_episode"] == 0 for row in mirror_rows)
        and all(row["movement_matches_fire_control"] for row in mirror_rows)
        and all(row["red_launches"] == 0 and row["blue_launches"] == 0
                for row in mirror_rows)
        and all(row["final_mirror_errors"]["position_m"] <= 1.0
                for row in mirror_rows)
        and all(row["final_mirror_errors"]["speed_mps"] <= 0.1
                for row in mirror_rows)
        and all(np.rad2deg(row["final_mirror_errors"]["attitude_rad"]) <= 0.1
                for row in mirror_rows))
    load_health = all(
        row["mirror_errors"]["load_scale"] <= 1e-6
        for row in mirror_rows)
    missile_trajectory_health = run_trajectory_audit()["MissileTrajectoryHealth"]
    report = {
        "profile": PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
        "seeds": list(SEEDS),
        "maximum_decision_steps": args.steps,
        "rows": rows,
        "FireControlClosureRed": _closure(
            by_scenario["red_fixed_vs_blue_straight"], "red"),
        "FireControlClosureBlue": _closure(
            by_scenario["blue_fixed_vs_red_straight"], "blue"),
        "MirrorDynamicsHealth": "PASS" if mirror_health else "FAIL",
        "LoadControlHealth": "PASS" if load_health else "FAIL",
        "MissileTrajectoryHealth": missile_trajectory_health,
    }
    report["OverallEnvironmentAudit"] = "PASS" if all(
        report[key] == "PASS" for key in (
            "FireControlClosureRed", "FireControlClosureBlue",
            "MirrorDynamicsHealth", "LoadControlHealth",
            "MissileTrajectoryHealth")) else "FAIL"
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
