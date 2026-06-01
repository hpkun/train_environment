from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


def _keys(value):
    return list(value.keys()) if isinstance(value, dict) else type(value).__name__


def _has_forbidden_runtime_import() -> bool:
    return any(
        name == "my_uav_env"
        or name.startswith("my_uav_env.")
        or name == "uav_env.brma_env"
        or name.startswith("uav_env.brma_env.")
        for name in sys.modules
    )


def main() -> None:
    env = HeteroUavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
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
        print(f"agent_types: {env.agent_types}")
        print(f"agent_roles: {env.agent_roles}")
        print(f"agent_models: {env.agent_models}")
        for aid in ["red_0", "red_1", "blue_0", "blue_1"]:
            print(f"{aid} model: {env.agent_models[aid]}")
        for i in range(5):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(
                f"step={i + 1} reward_keys={_keys(rewards)} "
                f"terminated_keys={_keys(terminated)} truncated_keys={_keys(truncated)}"
            )
        print(f"forbidden_runtime_import: {_has_forbidden_runtime_import()}")
        print("step_success: true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
