from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from uav_env.brma_env.env import UavCombatEnv


def _keys(value):
    if isinstance(value, dict):
        return list(value.keys())
    return type(value).__name__


def main() -> None:
    env = UavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        num_missiles_per_plane=1,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        print("reset_success: true")
        print(f"obs_keys: {_keys(obs)}")
        print(f"info_keys: {_keys(info)}")
        print(f"action_space_keys: {_keys(env.action_space.spaces)}")

        terminated = {}
        truncated = {}
        rewards = {}
        for step in range(5):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(
                f"step={step + 1} reward_keys={_keys(rewards)} "
                f"terminated_keys={_keys(terminated)} truncated_keys={_keys(truncated)} "
                f"info_keys={_keys(info)}"
            )

        parent_imported = any(
            name == "my_uav_env" or name.startswith("my_uav_env.")
            for name in sys.modules
        )
        print(f"parent_my_uav_env_imported: {parent_imported}")
        print("step_success: true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
