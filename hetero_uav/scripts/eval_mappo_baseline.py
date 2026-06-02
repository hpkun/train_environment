"""Smoke-evaluate MAPPO baseline. Stage 1 only, no win-rate experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
from algorithms.mappo.policy import MAPPOActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    model = MAPPOActorCritic().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()
    adapter = HeteroObsAdapter()

    returns = []
    lengths = []
    crashes = 0
    nan_count = 0

    env = make_env(args.config, env_type='jsbsim_hetero', max_steps=500)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_len = 0
        while True:
            result = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs_list = []
            for rid in env.red_ids:
                actor_obs_list.append(result['actor_obs'].get(
                    rid, np.zeros(140, dtype=np.float32)))
            actor_obs_t = torch.as_tensor(np.stack(actor_obs_list), device=device)

            with torch.no_grad():
                _, _, action, _, _ = model(
                    actor_obs_t, torch.zeros(1, 700, device=device),
                    deterministic=True)

            actions_dict = {}
            for i, rid in enumerate(env.red_ids):
                actions_dict[rid] = action[i].cpu().numpy().astype(np.float32)
            for bid in env.blue_ids:
                actions_dict[bid] = np.zeros(3, dtype=np.float32)

            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            ep_ret += sum(float(rewards_dict.get(rid, 0.0))
                          for rid in env.red_ids)
            ep_len += 1

            if np.isnan(ep_ret):
                nan_count += 1
                break

            if all(terminated.values()) or all(truncated.values()):
                # Count crashes
                for rid in env.red_ids:
                    dr = info.get(rid, {}).get('death_reason', '')
                    if 'crash' in str(dr).lower():
                        crashes += 1
                break

        returns.append(ep_ret)
        lengths.append(ep_len)

    print(f'episodes: {len(returns)}')
    print(f'avg_return: {np.mean(returns):.2f}')
    print(f'avg_length: {np.mean(lengths):.1f}')
    print(f'crashes: {crashes}')
    print(f'nan_detected: {nan_count > 0}')

    env.close()


if __name__ == '__main__':
    main()
