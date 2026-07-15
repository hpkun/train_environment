"""Five-seed short behavior audit for paper_learnable_3v3_v1."""
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
    PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
)
from my_uav_env import UavCombatEnv
from my_uav_env.blue_policy_profiles import BluePolicyController


def _fixed_actions(env: UavCombatEnv, obs: dict,
                   red_controller: BluePolicyController | None = None) -> dict:
    actions = env.blue_policy_actions({
        aid: obs[aid] for aid in env.blue_ids})
    if red_controller is not None:
        virtual_obs = {
            f"blue_{index}": obs[f"red_{index}"] for index in range(3)}
        virtual_engaged = {
            f"red_{int(target.split('_', 1)[1])}"
            for target in env.refresh_engaged_targets()
            if target.startswith("blue_")}
        virtual_actions = red_controller.act(
            virtual_obs, 3, 3, virtual_engaged,
            {f"blue_{index}": env.red_planes[f"red_{index}"].get_position()
             for index in range(3)},
            {f"blue_{index}": float(
                env.red_planes[f"red_{index}"].get_rpy()[2])
             for index in range(3)},
            env.current_step,
            own_alive={f"blue_{index}": bool(
                env.red_planes[f"red_{index}"].is_alive)
                for index in range(3)})
        actions.update({
            f"red_{index}": virtual_actions[f"blue_{index}"]
            for index in range(3)})
        return actions
    speed_action = 2.0 * (300.0 - env.VELOCITY_MIN) / (
        env.VELOCITY_MAX - env.VELOCITY_MIN) - 1.0
    for aid in env.red_ids:
        sim = env.red_planes[aid]
        actions[aid] = np.array([
            0.0, float(sim.get_rpy()[2]) / np.pi, speed_action],
            dtype=np.float32)
    return actions


def run_case(seed: int, scenario: str, steps: int) -> dict:
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
        suppress_jsbsim_output=False)
    try:
        obs, info = env.reset(seed=seed)
        red_controller = None
        if scenario == "deterministic_symmetric":
            env.set_team_mws_enabled("red", False)
            env.set_team_mws_enabled("blue", False)
            red_controller = BluePolicyController(
                "paper_learnable_fixed_pair_v1")
            red_controller.reset(
                [f"blue_{i}" for i in range(3)],
                [f"red_{i}" for i in range(3)],
                {f"blue_{i}": float(env.red_planes[f"red_{i}"].get_rpy()[2])
                 for i in range(3)},
                {f"blue_{i}": float(env.red_planes[f"red_{i}"].get_geodetic()[2])
                 for i in range(3)})
        if scenario == "pre_killed_reallocation":
            env.red_planes["red_0"].shotdown()
        rewards_seen = []
        actions_finite = True
        launches = {"red": 0, "blue": 0}
        hits = {"red": 0, "blue": 0}
        geometry = {"red": 0, "blue": 0}
        locks = {"red": 0, "blue": 0}
        for _ in range(steps):
            if scenario == "training_profile":
                actions = env.blue_policy_actions({aid: obs[aid] for aid in env.blue_ids})
                rng = np.random.default_rng(seed + env.current_step)
                actions.update({
                    aid: rng.uniform(-0.25, 0.25, 3).astype(np.float32)
                    for aid in env.red_ids})
            else:
                actions = _fixed_actions(env, obs, red_controller)
            actions_finite &= all(np.all(np.isfinite(value)) for value in actions.values())
            obs, rewards, terminated, truncated, info = env.step(actions)
            rewards_seen.extend(float(value) for value in rewards.values())
            for team in ("red", "blue"):
                launch_diag = info.get("__launch_diag__", {}).get(team, {})
                launches[team] += int(launch_diag.get("launches", 0))
                geometry[team] += int(launch_diag.get("geometry_ok_pairs", 0))
                locks[team] += int(launch_diag.get("lock_mature_pairs", 0))
                hits[team] = int(info.get("__missile_term__", {}).get(
                    team, {}).get("hit", hits[team]))
            if all(terminated[aid] or truncated[aid] for aid in env.agent_ids):
                break
        target_diag = info.get("__target_assignment_diag__", {})
        blue_diag = info.get("__blue_policy_diag__", {})
        mws_diag = info.get("__mws_diag__", {})
        return {
            "seed": seed,
            "scenario": scenario,
            "steps": int(env.current_step),
            "actions_finite": bool(actions_finite),
            "rewards_finite": bool(np.all(np.isfinite(rewards_seen))),
            "target_switches_while_alive": int(
                target_diag.get("target_switches_while_alive", 0)),
            "target_reallocations_after_death": int(
                target_diag.get("target_reallocations_after_death", 0)),
            "blue_target_switches_while_alive": int(
                blue_diag.get("blue_target_switches_while_alive", 0)),
            "blue_mws_override_agent_decisions": int(
                blue_diag.get("blue_mws_override_agent_decisions", 0)),
            "red_mws_detected_agent_decisions": int(
                mws_diag.get("red_mws_detected_agent_decisions", 0)),
            "red_mws_override_agent_decisions": int(
                mws_diag.get("red_mws_override_agent_decisions", 0)),
            "red_geometry": geometry["red"],
            "blue_geometry": geometry["blue"],
            "red_locks": locks["red"],
            "blue_locks": locks["blue"],
            "red_launches": launches["red"],
            "blue_launches": launches["blue"],
            "red_hits": hits["red"],
            "blue_hits": hits["blue"],
            "invalid_episode": int(info.get("__episode__", {}).get(
                "invalid_numerical_episode", False)),
            "actor_dim": 60,
            "critic_dim": 30,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = [
        run_case(seed, scenario, args.steps)
        for scenario in (
            "deterministic_symmetric", "training_profile",
            "pre_killed_reallocation")
        for seed in (3, 7, 11, 17, 23)
    ]
    report = {
        "profile": PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
        "rows": rows,
        "all_finite": all(
            row["actions_finite"] and row["rewards_finite"] for row in rows),
        "switches_while_alive_zero": all(
            row["target_switches_while_alive"] == 0
            and row["blue_target_switches_while_alive"] == 0 for row in rows),
        "blue_mws_overrides_zero": all(
            row["blue_mws_override_agent_decisions"] == 0 for row in rows),
    }
    both_sides_effective = all(
        any(row[f"{team}_launches"] > 0 for row in rows)
        and any(row[f"{team}_hits"] > 0 for row in rows)
        for team in ("red", "blue"))
    report["MissileEffectiveness"] = (
        "PASS" if both_sides_effective
        and report["all_finite"]
        and report["switches_while_alive_zero"]
        else "FAIL" if not both_sides_effective else "WARNING")
    report["red_mws_active_in_training_profile"] = any(
        row["red_mws_detected_agent_decisions"] > 0
        and row["red_mws_override_agent_decisions"] > 0
        for row in rows if row["scenario"] == "training_profile")
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
