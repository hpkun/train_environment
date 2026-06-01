from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo import MAPPOTrainer
from uav_env import make_env
from uav_env.wrappers import MAPPOEnvWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    trainer, _payload = MAPPOTrainer.load(args.model, device=device)
    trainer.model.eval()
    env = MAPPOEnvWrapper(make_env(args.env_config))
    if env.obs_shape != trainer.model.obs_dim or env.state_shape != trainer.model.state_dim:
        raise ValueError(
            f"model dims obs/state={trainer.model.obs_dim}/{trainer.model.state_dim} "
            f"do not match env dims {env.obs_shape}/{env.state_shape}; use compatible max agent padding"
        )

    wins, mav_survival, red_alive, blue_alive, returns, lengths = [], [], [], [], [], []
    for _ in range(args.episodes):
        obs, state, info = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                dist = trainer.model.distribution(obs_t)
                actions = torch.tanh(dist.mean).cpu().numpy()
            obs, state, rewards, dones, info = env.step(actions)
            ep_return += float(np.mean(rewards))
            ep_len += 1
            done = bool(np.all(dones))
        wins.append(1.0 if info.get("winner") == "red" else 0.0)
        mav_survival.append(float(info.get("mav_survival", 0.0)))
        red_alive.append(float(info.get("red_alive", 0.0)))
        blue_alive.append(float(info.get("blue_alive", 0.0)))
        returns.append(ep_return)
        lengths.append(float(ep_len))

    print(f"win_rate: {np.mean(wins):.3f}")
    print(f"mav_survival_rate: {np.mean(mav_survival):.3f}")
    print(f"average_red_alive: {np.mean(red_alive):.3f}")
    print(f"average_blue_alive: {np.mean(blue_alive):.3f}")
    print(f"average_episode_return: {np.mean(returns):.3f}")
    print(f"average_episode_length: {np.mean(lengths):.3f}")
    env.close()


if __name__ == "__main__":
    main()
