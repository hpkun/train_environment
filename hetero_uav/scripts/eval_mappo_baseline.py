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
from algorithms.mappo.opponent_policy import OpponentPolicy


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--opponent-policy',
                        choices=['zero', 'random', 'rule_nearest'],
                        default='rule_nearest')
    args = parser.parse_args()

    device = torch.device(args.device)

    model = MAPPOActorCritic().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()
    adapter = HeteroObsAdapter()
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)

    returns = []
    lengths = []
    red_alive_counts = []
    blue_alive_counts = []
    crashes = 0
    nan_detected = False

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
            actions_dict.update(opponent.act(obs, env.blue_ids))

            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            ep_ret += sum(float(rewards_dict.get(rid, 0.0))
                          for rid in env.red_ids)
            ep_len += 1

            if np.isnan(ep_ret) or _obs_has_nan(obs):
                nan_detected = True
                break

            if all(terminated.values()) or all(truncated.values()):
                # Count crashes
                for rid in env.red_ids:
                    dr = info.get(rid, {}).get('death_reason', '')
                    if 'crash' in str(dr).lower():
                        crashes += 1
                break

        red_alive, blue_alive = _alive_counts(env)
        red_alive_counts.append(red_alive)
        blue_alive_counts.append(blue_alive)
        returns.append(ep_ret)
        lengths.append(ep_len)

    print(f'config: {args.config}')
    print(f'opponent_policy: {args.opponent_policy}')
    print(f'episodes: {len(returns)}')
    print(f'avg_return: {np.mean(returns):.2f}')
    print(f'avg_length: {np.mean(lengths):.1f}')
    print(f'avg_red_alive: {np.mean(red_alive_counts):.2f}')
    print(f'avg_blue_alive: {np.mean(blue_alive_counts):.2f}')
    print(f'crashes: {crashes}')
    print(f'nan_detected: {nan_detected}')

    env.close()


if __name__ == '__main__':
    main()
