from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


def _actions(env, policy: str, rng: np.random.Generator) -> dict:
    if policy == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    if policy == "random":
        return {
            aid: env.action_space.spaces[aid].sample().astype(np.float32)
            for aid in env.agent_ids
        }
    raise ValueError(f"unknown policy: {policy}")


def _scan_sims(env) -> dict:
    sims = list(env.blue_planes.values()) + list(env.red_planes.values())
    altitudes = {}
    speeds = {}
    crashed = []
    shotdown = []
    alive = []
    nan_detected = False
    for sim in sims:
        altitudes[sim.uid] = float(sim.get_geodetic()[2])
        speeds[sim.uid] = float(np.linalg.norm(sim.get_velocity()))
        values = np.concatenate([
            sim.get_geodetic().astype(np.float64),
            sim.get_position().astype(np.float64),
            sim.get_velocity().astype(np.float64),
            np.asarray(sim.get_rpy(), dtype=np.float64),
        ])
        nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
        if sim.is_crash:
            crashed.append(sim.uid)
        if sim.is_shotdown:
            shotdown.append(sim.uid)
        if sim.is_alive:
            alive.append(sim.uid)
    return {
        "altitudes": altitudes,
        "speeds": speeds,
        "crashed_agents": crashed,
        "shotdown_agents": shotdown,
        "alive_agents": alive,
        "nan_detected": nan_detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", choices=["jsbsim_brma", "jsbsim_hetero"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = None
    try:
        env = make_env(args.config, env_type=args.env_type, max_steps=args.steps)
        obs, info = env.reset(seed=args.seed)
        steps_executed = 0
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        rewards = {aid: 0.0 for aid in env.agent_ids}
        missile_launch_counts = {aid: 0 for aid in env.agent_ids}
        min_altitude = float("inf")
        max_altitude = float("-inf")
        nan_detected = False

        scan = _scan_sims(env)
        for altitude in scan["altitudes"].values():
            min_altitude = min(min_altitude, altitude)
            max_altitude = max(max_altitude, altitude)
        nan_detected = nan_detected or scan["nan_detected"]

        for _ in range(args.steps):
            actions = _actions(env, args.policy, rng)
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            scan = _scan_sims(env)
            for aid, altitude in scan["altitudes"].items():
                min_altitude = min(min_altitude, altitude)
                max_altitude = max(max_altitude, altitude)
            nan_detected = nan_detected or scan["nan_detected"]
            for aid in env.agent_ids:
                aid_info = info.get(aid, {})
                if isinstance(aid_info, dict):
                    missile_launch_counts[aid] = missile_launch_counts.get(aid, 0) + int(
                        aid_info.get("missiles_fired_this_step", 0)
                    )
            if all(terminated.values()) or all(truncated.values()):
                break

        scan = _scan_sims(env)
        print(f"env_type: {args.env_type}")
        print(f"steps_executed: {steps_executed}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"obs_keys: {list(obs.keys())}")
        print(f"action_keys: {list(env.action_space.spaces.keys())}")
        print(f"reward_keys: {list(rewards.keys())}")
        print(f"info_keys: {list(info.keys())}")
        if hasattr(env, "agent_models"):
            print(f"agent_models: {env.agent_models}")
        if hasattr(env, "agent_types"):
            print(f"agent_types: {env.agent_types}")
        print(f"min_altitude: {min_altitude:.3f}")
        print(f"max_altitude: {max_altitude:.3f}")
        print(f"final_altitudes: {scan['altitudes']}")
        print(f"final_speeds: {scan['speeds']}")
        print(f"crashed_agents: {scan['crashed_agents']}")
        print(f"shotdown_agents: {scan['shotdown_agents']}")
        print(f"alive_agents: {scan['alive_agents']}")
        print(f"missile_count_in_flight: {len(env._missiles_in_flight)}")
        print(f"missile_launch_counts: {missile_launch_counts}")
        print(f"missile_term_reasons: {getattr(env, '_missile_term_reasons', {})}")
        print(f"nan_detected: {nan_detected}")
    except Exception:
        print("exception traceback:")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
