"""Smoke test for entity observation tensor construction."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from my_uav_env.alignment.entity_obs import build_entity_observation, infer_entity_layout
from my_uav_env import UavCombatEnv


def _print_entity_info(agent_id: str, obs_np: dict):
    entities, entity_mask = build_entity_observation(obs_np)
    layout = infer_entity_layout(obs_np)
    print(f"agent: {agent_id}")
    print(f"  entities.shape: {entities.shape}")
    print(f"  entity_mask: {entity_mask.tolist()}")
    print(f"  layout: {layout}")


def main():
    env = UavCombatEnv(
        max_num_blue=1,
        max_num_red=1,
        max_steps=2,
        suppress_jsbsim_output=True,
        enable_gcas_for_blue=False,
    )
    try:
        obs, _ = env.reset()
        _print_entity_info("red_0", obs["red_0"])
        _print_entity_info("blue_0", obs["blue_0"])

        zero = np.zeros(3, dtype=np.float32)
        actions = {"red_0": zero, "blue_0": zero}
        obs, _rewards, _terminated, _truncated, _info = env.step(actions)
        _print_entity_info("red_0", obs["red_0"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
