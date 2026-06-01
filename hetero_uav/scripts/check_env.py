from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    env = make_env(args.config)
    obs, info = env.reset(seed=0)
    print(f"config: {Path(args.config)}")
    print(f"num_agents: {env.num_agents}")
    print(f"obs_shape: {env.obs_shape}")
    print(f"state_shape: {env.state_shape}")
    print(f"action_shape: {env.action_shape}")
    print(f"agent_ids: {env.agent_ids}")
    print(f"initial info keys: {sorted(info.keys())}")

    for step in range(10):
        actions = {aid: np.random.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
                   for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        dones = {aid: terminated.get(aid, False) or truncated.get(aid, False)
                 for aid in env.agent_ids}
        print(
            f"step={step + 1} reward={rewards} done={all(dones.values())} "
            f"info_keys={sorted(k for k in info.keys() if not str(k).startswith('red_') and not str(k).startswith('blue_'))}"
        )
        if all(dones.values()):
            break

    first_obs = obs[env.agent_ids[0]]
    print(f"first_agent_flat_shape: {first_obs['flat'].shape}")
    print("check_env: OK")
    env.close()


if __name__ == "__main__":
    main()
