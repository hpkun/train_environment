"""Smoke train plain shared-policy MAPPO baseline. Stage 1 only."""
from __future__ import annotations

import argparse
import json
import os
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
from algorithms.mappo.storage import RolloutBuffer
from algorithms.mappo.trainer import PPOTrainer


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rollout-length', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs/mappo_baseline')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--opponent-policy',
                        choices=['zero', 'random', 'rule_nearest'],
                        default='rule_nearest')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(f'{args.output_dir}/latest', exist_ok=True)

    env = make_env(args.config, env_type='jsbsim_hetero', max_steps=500)
    adapter = HeteroObsAdapter()
    model = MAPPOActorCritic().to(device)
    trainer = PPOTrainer(model)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)

    num_red = env.max_num_red
    obs, info = env.reset(seed=args.seed)

    if args.debug:
        print(f'num_red={num_red} actor_obs_dim={model.actor_obs_dim} '
              f'critic_state_dim={model.critic_state_dim}')
        print(f'opponent_policy={args.opponent_policy}')

    total_steps = 0
    episodes_completed = 0
    episode_returns = []
    episode_lengths = []
    episode_red_alive = []
    episode_blue_alive = []

    current_ep_returns = np.zeros(num_red, dtype=np.float32)
    current_ep_length = 0

    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer(
            max_len=args.rollout_length, num_red=num_red,
            actor_dim=140, critic_dim=700, action_dim=3)

        for step in range(args.rollout_length):
            result = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

            # Collect actor obs for red agents
            actor_obs_list = []
            for rid in env.red_ids:
                actor_obs_list.append(result['actor_obs'].get(
                    rid, np.zeros(140, dtype=np.float32)))
            actor_obs_np = np.stack(actor_obs_list)  # (num_red, 140)

            critic_state_np = result['critic_state']
            red_valid_np = result['red_valid_mask']

            actor_obs_t = torch.as_tensor(actor_obs_np, device=device)
            critic_t = torch.as_tensor(critic_state_np, device=device).unsqueeze(0)

            with torch.no_grad():
                dist, value, action, log_prob, entropy = model(
                    actor_obs_t, critic_t, deterministic=False)

            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.item()

            # Build actions dict: red from model, blue zero (placeholder)
            actions_dict = {}
            for i, rid in enumerate(env.red_ids):
                actions_dict[rid] = action_np[i].astype(np.float32)
            actions_dict.update(opponent.act(obs, env.blue_ids))

            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)

            # Per-agent reward
            rewards_np = np.array(
                [float(rewards_dict.get(rid, 0.0)) for rid in env.red_ids],
                dtype=np.float32)
            dones_np = np.array(
                [float(terminated.get(rid, False) or truncated.get(rid, False))
                 for rid in env.red_ids], dtype=np.float32)

            current_ep_returns += rewards_np
            current_ep_length += 1

            buffer.store(actor_obs_np, critic_state_np, action_np, log_prob_np,
                         rewards_np, dones_np, value_np, red_valid_np)
            total_steps += 1

            if all(terminated.values()) or all(truncated.values()):
                episodes_completed += 1
                episode_returns.append(float(current_ep_returns.mean()))
                episode_lengths.append(current_ep_length)
                red_alive, blue_alive = _alive_counts(env)
                episode_red_alive.append(float(red_alive))
                episode_blue_alive.append(float(blue_alive))
                current_ep_returns[:] = 0.0
                current_ep_length = 0
                obs, info = env.reset(seed=args.seed + total_steps)

        # PPO update
        stats = trainer.update(buffer)

        avg_ret = np.mean(episode_returns[-10:]) if episode_returns else 0.0
        avg_len = np.mean(episode_lengths[-10:]) if episode_lengths else 0
        current_red_alive, current_blue_alive = _alive_counts(env)
        avg_red_alive = (np.mean(episode_red_alive[-10:])
                         if episode_red_alive else float(current_red_alive))
        avg_blue_alive = (np.mean(episode_blue_alive[-10:])
                          if episode_blue_alive else float(current_blue_alive))
        print(f'Iter {iteration:3d} | steps={total_steps:5d} | '
              f'ret={avg_ret:+8.2f} | len={avg_len:.0f} | '
              f'average_red_return={avg_ret:+.2f} | '
              f'average_red_alive={avg_red_alive:.2f} | '
              f'average_blue_alive={avg_blue_alive:.2f} | '
              f'episode_count={episodes_completed} | '
              f'opponent_policy={args.opponent_policy} | '
              f'actor={stats["actor_loss"]:+.4f} '
              f'critic={stats["critic_loss"]:+.4f} '
              f'ent={stats["entropy"]:.4f} | ep={episodes_completed}')

    # Save
    model_path = f'{args.output_dir}/latest/model.pt'
    torch.save(model.state_dict(), model_path)
    meta = {'iterations': args.iterations, 'episodes': episodes_completed,
            'returns': episode_returns, 'lengths': episode_lengths,
            'red_alive': episode_red_alive,
            'blue_alive': episode_blue_alive,
            'opponent_policy': args.opponent_policy}
    with open(f'{args.output_dir}/latest/meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'Saved {model_path}')

    env.close()


if __name__ == '__main__':
    main()
