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
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    env = make_env(args.config)
    obs, info = env.reset(seed=0)
    print(f"controlled_side={env.controlled_side} controlled_agents={env.num_agents}")
    done = False
    step = 0
    total_reward = {aid: 0.0 for aid in env.agent_ids}
    while not done:
        actions = {aid: np.random.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
                   for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid, reward in rewards.items():
            total_reward[aid] += float(reward)
        step += 1
        done = all(terminated.get(aid, False) or truncated.get(aid, False)
                   for aid in env.agent_ids)
        if step % args.log_interval == 0 or done:
            print(
                f"step={step} red_alive={info['red_alive']} blue_alive={info['blue_alive']} "
                f"mav_alive={info['mav_alive']} reward={rewards}"
            )

    print("episode summary:")
    print(f"  steps: {step}")
    print(f"  win_flag: {info['win_flag']}")
    print(f"  total_reward: {total_reward}")
    print(f"  missile_left: {info['missile_left']}")
    env.close()


if __name__ == "__main__":
    main()
