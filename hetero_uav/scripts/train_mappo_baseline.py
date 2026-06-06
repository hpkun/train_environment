"""Plain shared-policy MAPPO baseline. Stage 1 trainability diagnostics.

Supports CSV logging, checkpoint saving, action stats, NaN detection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
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
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--total-env-steps', type=int, default=None)
    parser.add_argument('--rollout-length', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs/mappo_baseline')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--opponent-policy',
                        choices=['zero', 'random', 'rule_nearest', 'greedy_fsm'],
                        default='rule_nearest')
    parser.add_argument('--log-csv', default=None)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--obs-adapter-version', choices=['v1', 'v2'],
                        default='v1')
    args = parser.parse_args()
    if args.total_env_steps is not None and args.total_env_steps <= 0:
        raise SystemExit('--total-env-steps must be positive')

    if args.log_csv is None:
        args.log_csv = f'{args.output_dir}/train_log.csv'

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(f'{args.output_dir}/latest', exist_ok=True)
    os.makedirs(f'{args.output_dir}/checkpoints', exist_ok=True)
    log_dir = os.path.dirname(args.log_csv)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    csv_file = open(args.log_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ['iteration', 'total_steps', 'average_team_return',
                  'average_episode_length', 'average_red_alive',
                  'average_blue_alive', 'actor_loss', 'critic_loss',
                  'entropy', 'action_mean_abs', 'action_std',
                  'action_min', 'action_max', 'value_mean', 'value_std',
                  'nan_detected', 'opponent_policy', 'episodes_completed']
    csv_writer.writerow(csv_header)
    csv_file.flush()

    env = make_env(args.config, env_type='jsbsim_hetero',
                   max_steps=args.max_steps)
    obs_mode = getattr(env, 'observation_mode', 'brma_sensor')

    if args.obs_adapter_version == 'v2':
        if obs_mode != 'mav_shared_geo':
            raise SystemExit(
                '--obs-adapter-version v2 requires observation_mode=mav_shared_geo')
        adapter = HeteroObsAdapterV2()
    else:
        adapter = HeteroObsAdapter()

    actor_obs_dim = adapter.flat_actor_obs_dim
    critic_state_dim = adapter.critic_state_dim
    model = MAPPOActorCritic(actor_obs_dim=actor_obs_dim,
                             critic_state_dim=critic_state_dim).to(device)
    computed_iterations = args.iterations
    if args.total_env_steps is not None:
        computed_iterations = int(math.ceil(args.total_env_steps / args.rollout_length))
        print(f'total_env_steps_target={args.total_env_steps} '
              f'rollout_length={args.rollout_length} '
              f'computed_iterations={computed_iterations}', flush=True)

    if args.debug:
        print(f'obs_adapter_version={args.obs_adapter_version} '
              f'observation_mode={obs_mode} '
              f'actor_obs_dim={actor_obs_dim} '
              f'critic_state_dim={critic_state_dim}', flush=True)
    trainer = PPOTrainer(model)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)

    num_red = env.max_num_red
    obs, info = env.reset(seed=args.seed)

    if args.debug:
        print(f'num_red={num_red} actor_obs_dim={model.actor_obs_dim} '
              f'critic_state_dim={model.critic_state_dim}', flush=True)
        print(f'opponent_policy={args.opponent_policy}', flush=True)

    total_steps = 0
    episodes_completed = 0
    episode_returns = []    # mean over red agents
    episode_lengths = []
    episode_red_alive = []
    episode_blue_alive = []

    current_ep_returns = np.zeros(num_red, dtype=np.float32)
    current_ep_length = 0
    nan_detected = False
    best_model_path = f'{args.output_dir}/latest/model.pt'
    avg_ret = 0.0
    avg_len = 0.0
    avg_red_alive = float(_alive_counts(env)[0])
    avg_blue_alive = float(_alive_counts(env)[1])
    iterations_completed = 0

    for iteration in range(1, computed_iterations + 1):
        if args.total_env_steps is not None and total_steps >= args.total_env_steps:
            break
        current_rollout_len = args.rollout_length
        if args.total_env_steps is not None:
            current_rollout_len = min(
                args.rollout_length, args.total_env_steps - total_steps)
        if current_rollout_len <= 0:
            break

        buffer = RolloutBuffer(
            max_len=current_rollout_len, num_red=num_red,
            actor_dim=actor_obs_dim, critic_dim=critic_state_dim, action_dim=3)

        iter_actions = []  # for action stats

        for step in range(current_rollout_len):
            result = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

            actor_obs_list = []
            for rid in env.red_ids:
                actor_obs_list.append(result['actor_obs'].get(
                    rid, np.zeros(actor_obs_dim, dtype=np.float32)))
            actor_obs_np = np.stack(actor_obs_list)

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
            iter_actions.append(action_np)

            if np.isnan(action_np).any() or np.isnan(value_np):
                nan_detected = True
                print(f'[WARN] NaN detected at iter {iteration} step {step}',
                      flush=True)
                break

            actions_dict = {}
            for i, rid in enumerate(env.red_ids):
                actions_dict[rid] = action_np[i].astype(np.float32)
            actions_dict.update(opponent.act(obs, env.blue_ids, env=env))

            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)

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

            if nan_detected:
                break

        if nan_detected:
            print('[WARN] Training stopped due to NaN', flush=True)
            break

        # PPO update
        stats = trainer.update(buffer)
        iterations_completed = iteration

        # Action statistics
        all_acts = np.concatenate(iter_actions, axis=0)
        act_mean_abs = float(np.mean(np.abs(all_acts)))
        act_std = float(np.mean(np.std(all_acts, axis=0)))
        act_min = float(all_acts.min())
        act_max = float(all_acts.max())

        # Value statistics from buffer
        buf_vals = buffer.values[:buffer.pos]
        val_mean = float(np.mean(buf_vals))
        val_std = float(np.std(buf_vals))

        avg_ret = (np.mean(episode_returns[-10:])
                   if episode_returns else 0.0)
        avg_len = (np.mean(episode_lengths[-10:])
                   if episode_lengths else 0)
        avg_red_alive = (np.mean(episode_red_alive[-10:])
                         if episode_red_alive else float(_alive_counts(env)[0]))
        avg_blue_alive = (np.mean(episode_blue_alive[-10:])
                          if episode_blue_alive else float(_alive_counts(env)[1]))

        csv_writer.writerow([
            iteration, total_steps, f'{avg_ret:.4f}',
            f'{avg_len:.1f}', f'{avg_red_alive:.2f}',
            f'{avg_blue_alive:.2f}',
            f'{stats["actor_loss"]:.6f}', f'{stats["critic_loss"]:.6f}',
            f'{stats["entropy"]:.6f}',
            f'{act_mean_abs:.6f}', f'{act_std:.6f}',
            f'{act_min:.6f}', f'{act_max:.6f}',
            f'{val_mean:.6f}', f'{val_std:.6f}',
            int(nan_detected), args.opponent_policy,
            episodes_completed,
        ])
        csv_file.flush()

        print(f'Iter {iteration:3d} | steps={total_steps:5d} | '
              f'ret={avg_ret:+7.2f} | len={avg_len:4.0f} | '
              f'r_alive={avg_red_alive:.1f} b_alive={avg_blue_alive:.1f} | '
              f'act_loss={stats["actor_loss"]:+.4f} '
              f'crit_loss={stats["critic_loss"]:+.4f} '
              f'ent={stats["entropy"]:.4f} | '
              f'act_mu={act_mean_abs:.3f} act_std={act_std:.3f} | '
              f'val={val_mean:+.3f} | ep={episodes_completed}', flush=True)

        # Checkpoint saving
        if not args.no_save and iteration % args.save_interval == 0:
            ckpt_path = (f'{args.output_dir}/checkpoints/'
                         f'iter_{iteration:04d}.pt')
            torch.save(model.state_dict(), ckpt_path)

    # Final save
    if not args.no_save:
        torch.save(model.state_dict(), best_model_path)
        meta = {
            'config': args.config, 'seed': args.seed,
            'opponent_policy': args.opponent_policy,
            'iterations': args.iterations,
            'computed_iterations': computed_iterations,
            'iterations_completed': iterations_completed,
            'rollout_length': args.rollout_length,
            'total_env_steps_target': args.total_env_steps,
            'total_env_steps_actual': total_steps,
            'episodes': episodes_completed,
            'final_return': avg_ret,
            'final_red_alive': avg_red_alive,
            'final_blue_alive': avg_blue_alive,
            'nan_detected': nan_detected,
            'obs_adapter_version': args.obs_adapter_version,
            'actor_obs_dim': actor_obs_dim,
            'critic_state_dim': critic_state_dim,
            'actor_arch': 'mlp',
            'observation_mode': obs_mode,
        }
        with open(f'{args.output_dir}/latest/meta.json', 'w') as f:
            json.dump(meta, f)
        print(f'Saved {best_model_path}', flush=True)

    csv_file.close()
    env.close()


if __name__ == '__main__':
    main()
