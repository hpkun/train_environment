"""Gymnasium API and short random-action smoke check."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.env import AirCombat1v1Env


def main():
    check_env(AirCombat1v1Env(max_steps=20), skip_render_check=True)
    env = AirCombat1v1Env()
    observation, info = env.reset(seed=0)
    terminated = truncated = False
    reward = 0.0
    for _ in range(200):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        if (not env.observation_space.contains(observation)
                or not np.isfinite(reward)
                or not all(np.isfinite(value) for value in info.values()
                           if isinstance(value, (int, float)))):
            raise RuntimeError("non-finite environment output")
        if terminated or truncated:
            break
    print({"check_env": "passed", "steps": info["step"],
           "event": info["event"], "reward": reward,
           "distance_m": info["distance_m"]})
    env.close()


if __name__ == "__main__":
    main()
