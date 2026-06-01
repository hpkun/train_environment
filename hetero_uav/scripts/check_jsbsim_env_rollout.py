from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.core.opponent_policy import RuleNearestOpponentPolicy


def _actions(env, policy_name: str, rng: np.random.Generator):
    if policy_name == "random":
        return {
            aid: rng.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
            for aid in env.agent_ids
        }
    if policy_name == "zero":
        return {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    if policy_name == "rule_nearest":
        policy = RuleNearestOpponentPolicy()
        by_id = {agent.agent_id: agent for agent in env.task.agents}
        return {
            aid: policy.act(by_id[aid], env.task.agents)
            for aid in env.agent_ids
        }
    raise ValueError(f"unknown policy {policy_name!r}")


def _scan_agents(env):
    altitudes = []
    nan_detected = False
    crashed_agents = []
    alive_agents = []
    for agent in env.task.agents:
        values = np.concatenate([
            agent.position.astype(np.float64),
            agent.velocity.astype(np.float64),
            np.array([agent.pitch, agent.roll, agent.heading, agent.speed], dtype=np.float64),
        ])
        nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
        altitudes.append(float(agent.position[2]))
        if agent.crashed:
            crashed_agents.append(agent.agent_id)
        if agent.alive:
            alive_agents.append(agent.agent_id)
    return altitudes, nan_detected, crashed_agents, alive_agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/configs/hetero_2v2_jsbsim_debug.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--policy", choices=["random", "zero", "rule_nearest"], default="random")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    reset_success = False
    step_success = False
    steps_executed = 0
    env = make_env(args.config, episode_limit=args.steps)
    try:
        _obs, info = env.reset(seed=0)
        reset_success = True
        min_altitude = float("inf")
        max_altitude = float("-inf")
        nan_detected = False
        crashed_agents = []
        alive_agents = []
        for _ in range(args.steps):
            actions = _actions(env, args.policy, rng)
            _obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            altitudes, has_nan, crashed_agents, alive_agents = _scan_agents(env)
            nan_detected = nan_detected or has_nan
            min_altitude = min(min_altitude, min(altitudes))
            max_altitude = max(max_altitude, max(altitudes))
            if all(terminated.get(aid, False) or truncated.get(aid, False) for aid in env.agent_ids):
                break
        step_success = True
        print(f"config: {args.config}")
        print(f"policy: {args.policy}")
        print(f"steps_executed: {steps_executed}")
        print(f"reset_success: {reset_success}")
        print(f"step_success: {step_success}")
        print(f"crashed_agents: {crashed_agents}")
        print(f"alive_agents: {alive_agents}")
        print(f"min_altitude: {min_altitude:.3f}")
        print(f"max_altitude: {max_altitude:.3f}")
        print(f"nan_detected: {nan_detected}")
        print(f"obs_shape: {env.obs_shape}")
        print(f"state_shape: {env.state_shape}")
        print(f"action_shape: {env.action_shape}")
        print(f"final_info_keys: {sorted(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
