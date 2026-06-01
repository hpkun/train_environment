from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.uav_combat_env import UavCombatEnv


def _keys(value):
    return list(value.keys()) if isinstance(value, dict) else type(value).__name__


def main() -> None:
    env = UavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        num_missiles_per_plane=2,
        sim_freq=60,
        agent_interaction_steps=12,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        print("reset_success: true")
        print(f"obs_keys: {_keys(obs)}")
        print(f"info_keys: {_keys(info)}")
        print(f"observation_space: {env.observation_space}")
        for i in range(5):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(
                f"step={i + 1} reward_keys={_keys(rewards)} "
                f"terminated_keys={_keys(terminated)} truncated_keys={_keys(truncated)} "
                f"info_keys={_keys(info)}"
            )
        print("step_success: true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
