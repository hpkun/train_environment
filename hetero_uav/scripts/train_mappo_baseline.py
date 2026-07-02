"""Plain shared-policy MAPPO baseline. Stage 1 trainability diagnostics.

Supports CSV logging, checkpoint saving, action stats, NaN detection,
periodic eval during training, best-checkpoint tracking, and extended
episode-outcome / missile statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from collections import deque

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

DEFAULT_EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _build_red_alive_mask(info: dict, env, red_ids: list[str]) -> np.ndarray:
    """Build PPO active-agent mask for controlled red agents.

    ``red_valid_mask`` from the observation adapter means a padded slot exists.
    For PPO loss masking we need an alive mask so dead agents stop contributing
    to actor loss and team reward aggregation.
    """
    mask = np.zeros(len(red_ids), dtype=np.float32)
    for i, rid in enumerate(red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = getattr(env, "red_planes", {}).get(rid) if env is not None else None
            alive = bool(sim is not None and getattr(sim, "is_alive", False))
        mask[i] = 1.0 if alive else 0.0
    return mask


def _team_episode_done(terminated: dict, truncated: dict) -> bool:
    """Return episode-level done for centralized critic GAE.

    Individual aircraft death should be represented through the active-agent
    mask. It should not truncate centralized team value unless the whole episode
    has terminated or truncated.
    """
    return bool(all(terminated.values()) or all(truncated.values()))


def _mav_alive(env) -> bool:
    sim = env.red_planes.get("red_0")
    return sim is not None and sim.is_alive


def _count_missile_hits_from_info(mt: dict) -> int:
    """Return total missile hit count from info['__missile_term__'].

    Structure: {'red': {'hit': N, ...}, 'blue': {'hit': M, ...}}
    """
    total = 0
    if isinstance(mt, dict):
        for team, reasons in mt.items():
            if isinstance(reasons, dict):
                total += int(reasons.get("hit", 0))
    return total


def _episode_outcome(env, any_truncated: bool, ep_length: int) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    mav_survived = _mav_alive(env)
    max_steps = getattr(env, "max_steps", 0)
    timeout = bool(any_truncated or ep_length >= max_steps)
    if blue_alive == 0 and red_alive > 0:
        return {"winner": "red", "end_reason": "blue_eliminated"}
    if red_alive == 0 and blue_alive > 0:
        return {"winner": "blue", "end_reason": "red_eliminated"}
    if red_alive == 0 and blue_alive == 0:
        return {"winner": "draw", "end_reason": "mutual_elimination"}
    if timeout:
        if red_alive > blue_alive:
            return {"winner": "red", "end_reason": "timeout"}
        elif blue_alive > red_alive:
            return {"winner": "blue", "end_reason": "timeout"}
        else:
            return {"winner": "draw", "end_reason": "timeout"}
    return {"winner": "none", "end_reason": "ongoing"}


def _compute_best_score(records: list[dict]) -> float:
    for r in records:
        if "3v2" in r.get("config", ""):
            return (r.get("red_win_rate", 0.0)
                    + 0.1 * r.get("mav_survival_rate", 0.0)
                    + 0.01 * r.get("avg_return", 0.0))
    return 0.0


def _run_eval(model_path: str, opponent_policy: str, obs_adapter: str,
              episodes: int, device: str, configs: list[str],
              summary_json: str) -> list[dict] | None:
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model", model_path,
        "--obs-adapter-version", obs_adapter,
        "--episodes", str(episodes),
        "--device", device,
        "--opponent-policy", opponent_policy,
        "--configs", *configs,
        "--summary-json", summary_json,
    ]
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=1200,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(Path(summary_json).read_text(encoding="utf-8"))
    except Exception:
        return None


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
                        choices=['zero', 'random', 'rule_nearest', 'greedy_fsm', 'brma_rule', 'tam_greedy_easy', 'brma_rule_safe_pursuit_easy'],
                        default='rule_nearest')
    parser.add_argument('--log-csv', default=None)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--obs-adapter-version', choices=['v1', 'v2'],
                        default='v1')
    parser.add_argument('--console-log-interval', type=int, default=1,
                        help='Print one training progress line every N iterations.')
    parser.add_argument('--actor-arch', choices=['mlp', 'role_conditioned'],
                        default='mlp')
    # -- PPO hyperparameters --
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--actor-lr', type=float, default=5e-4)
    parser.add_argument('--critic-lr', type=float, default=5e-4)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    # -- periodic eval during training --
    parser.add_argument('--eval-during-training', action='store_true')
    parser.add_argument('--eval-interval-steps', type=int, default=50000)
    parser.add_argument('--train-eval-episodes', type=int, default=5)
    parser.add_argument('--eval-configs', nargs='*', default=None)
    parser.add_argument('--best-checkpoint-metric',
                        choices=['3v2_red_win_mav_survival_return', '3v2_return'],
                        default='3v2_red_win_mav_survival_return')
    args = parser.parse_args()

    if args.total_env_steps is not None and args.total_env_steps <= 0:
        raise SystemExit('--total-env-steps must be positive')
    if args.console_log_interval <= 0:
        raise SystemExit('--console-log-interval must be positive')

    if args.log_csv is None:
        args.log_csv = f'{args.output_dir}/train_log.csv'

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(f'{args.output_dir}/latest', exist_ok=True)
    os.makedirs(f'{args.output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{args.output_dir}/best', exist_ok=True)
    log_dir = os.path.dirname(args.log_csv)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    eval_configs = args.eval_configs if args.eval_configs else DEFAULT_EVAL_CONFIGS

    # ---- train CSV ----
    csv_file = open(args.log_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ['iteration', 'total_steps', 'average_team_return',
                  'average_episode_length', 'average_red_alive',
                  'average_blue_alive', 'actor_loss', 'critic_loss',
                  'entropy', 'action_mean_abs', 'action_std',
                  'action_min', 'action_max', 'value_mean', 'value_std',
                  'action_saturation_rate',
                  'mav_action_mean_abs_recent', 'uav_action_mean_abs_recent',
                  'mav_action_saturation_rate_recent', 'uav_action_saturation_rate_recent',
                  'train_red_win_rate_recent', 'train_blue_win_rate_recent',
                  'train_draw_rate_recent', 'train_timeout_rate_recent',
                  'train_mav_survival_rate_recent',
                  'train_red_alive_final_recent', 'train_blue_alive_final_recent',
                  'train_red_missiles_fired_recent', 'train_blue_missiles_fired_recent',
                  'train_missile_hit_count_recent', 'train_missile_hit_rate_recent',
                  'nan_detected', 'opponent_policy', 'episodes_completed']
    csv_writer.writerow(csv_header)
    csv_file.flush()

    # ---- eval CSV ----
    eval_log_path = Path(args.output_dir) / 'eval_log.csv'
    if args.eval_during_training:
        eval_csv = open(eval_log_path, 'w', newline='')
        eval_csv_writer = csv.writer(eval_csv)
        eval_csv_header = ['total_steps', 'iteration', 'eval_config',
                           'avg_return', 'avg_length', 'red_win_rate',
                           'blue_win_rate', 'draw_rate', 'timeout_rate',
                           'mav_survival_rate', 'red_alive_final_mean',
                           'blue_alive_final_mean', 'nan_detected',
                           'actor_dim_ok', 'critic_dim_ok']
        eval_csv_writer.writerow(eval_csv_header)
        eval_csv.flush()
    else:
        eval_csv = eval_csv_writer = None

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
    from algorithms.mappo.adapter_utils import make_mappo_model_for_adapter
    model = make_mappo_model_for_adapter(adapter, device, actor_arch=args.actor_arch)
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
    trainer = PPOTrainer(model, lr_actor=args.actor_lr, lr_critic=args.critic_lr,
                         clip_param=args.clip_param, entropy_coef=args.entropy_coef,
                         max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
                         gamma=args.gamma, gae_lambda=args.gae_lambda)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)

    num_red = env.max_num_red
    obs, info = env.reset(seed=args.seed)

    if args.debug:
        print(f'num_red={num_red} actor_obs_dim={model.actor_obs_dim} '
              f'critic_state_dim={model.critic_state_dim}', flush=True)
        print(f'opponent_policy={args.opponent_policy}', flush=True)

    total_steps = 0
    episodes_completed = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_red_alive: list[float] = []
    episode_blue_alive: list[float] = []

    # Episode outcome buckets (recent window)
    recent_outcomes: deque[dict] = deque(maxlen=100)
    recent_missile_stats: dict[str, int] = {
        "red_fired": 0, "blue_fired": 0, "hit": 0, "total_episodes": 0,
    }

    current_ep_returns = np.zeros(num_red, dtype=np.float32)
    current_ep_length = 0
    nan_detected = False
    best_model_path = f'{args.output_dir}/latest/model.pt'
    best_score = -float("inf")
    avg_ret = 0.0
    avg_len = 0.0
    avg_red_alive = float(_alive_counts(env)[0])
    avg_blue_alive = float(_alive_counts(env)[1])
    iterations_completed = 0
    last_eval_step = -999999

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

        iter_actions = []

        for step in range(current_rollout_len):
            result = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

            actor_obs_list = []
            for rid in env.red_ids:
                actor_obs_list.append(result['actor_obs'].get(
                    rid, np.zeros(actor_obs_dim, dtype=np.float32)))
            actor_obs_np = np.stack(actor_obs_list)

            critic_state_np = result['critic_state']
            # Adapter red_valid_mask is a padded-slot mask. PPO needs an
            # active-agent mask, so dead red agents do not contribute to actor
            # loss or team reward after individual death.
            red_alive_mask = _build_red_alive_mask(info, env, env.red_ids)

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
            episode_done = _team_episode_done(terminated, truncated)

            rewards_np = np.array(
                [float(rewards_dict.get(rid, 0.0)) for rid in env.red_ids],
                dtype=np.float32)
            # Team done is repeated for each red agent because centralized GAE
            # should reset only on episode-level termination/truncation.
            dones_np = np.full((num_red,), float(episode_done), dtype=np.float32)

            current_ep_returns += rewards_np
            current_ep_length += 1

            buffer.store(actor_obs_np, critic_state_np, action_np, log_prob_np,
                         rewards_np, dones_np, value_np, red_alive_mask)
            total_steps += 1

            # Missile fire stats from info
            for aid in env.agent_ids:
                agent_info = info.get(aid, {})
                if isinstance(agent_info, dict):
                    fired = int(agent_info.get("missiles_fired_this_step", 0))
                    if fired > 0:
                        if aid.startswith("red_"):
                            recent_missile_stats["red_fired"] += fired
                        else:
                            recent_missile_stats["blue_fired"] += fired
            mt = info.get("__missile_term__", {})
            if isinstance(mt, dict):
                for team, reasons in mt.items():
                    if isinstance(reasons, dict):
                        recent_missile_stats["hit"] += int(reasons.get("hit", 0))

            if episode_done:
                episodes_completed += 1
                episode_returns.append(float(current_ep_returns.mean()))
                episode_lengths.append(current_ep_length)
                red_alive, blue_alive = _alive_counts(env)
                episode_red_alive.append(float(red_alive))
                episode_blue_alive.append(float(blue_alive))
                outcome = _episode_outcome(env, any(truncated.values()), current_ep_length)
                recent_outcomes.append({
                    "winner": outcome["winner"],
                    "end_reason": outcome["end_reason"],
                    "mav_survived": _mav_alive(env),
                    "length": current_ep_length,
                    "red_alive": red_alive,
                    "blue_alive": blue_alive,
                })
                recent_missile_stats["total_episodes"] += 1
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
        act_sat = float(np.mean(np.abs(all_acts) >= 0.999))

        # Per-role action stats (MAV = red_0 index 0, UAVs = rest)
        mav_acts = np.concatenate([a[0:1] for a in iter_actions], axis=0)
        uav_acts = np.concatenate([a[1:] for a in iter_actions], axis=0)
        mav_act_abs = float(np.mean(np.abs(mav_acts))) if mav_acts.size > 0 else 0.0
        uav_act_abs = float(np.mean(np.abs(uav_acts))) if uav_acts.size > 0 else 0.0
        mav_sat = float(np.mean(np.abs(mav_acts) >= 0.999)) if mav_acts.size > 0 else 0.0
        uav_sat = float(np.mean(np.abs(uav_acts) >= 0.999)) if uav_acts.size > 0 else 0.0

        # Value statistics
        buf_vals = buffer.values[:buffer.pos]
        val_mean = float(np.mean(buf_vals))
        val_std = float(np.std(buf_vals))

        # Rolling averages
        avg_ret = (np.mean(episode_returns[-10:])
                   if episode_returns else 0.0)
        avg_len = (np.mean(episode_lengths[-10:])
                   if episode_lengths else 0)
        avg_red_alive = (np.mean(episode_red_alive[-10:])
                         if episode_red_alive else float(_alive_counts(env)[0]))
        avg_blue_alive = (np.mean(episode_blue_alive[-10:])
                          if episode_blue_alive else float(_alive_counts(env)[1]))

        # Recent episode outcome stats
        recent_list = list(recent_outcomes)
        n_rec = len(recent_list)
        if n_rec > 0:
            red_win_rate = sum(1 for o in recent_list if o["winner"] == "red") / n_rec
            blue_win_rate = sum(1 for o in recent_list if o["winner"] == "blue") / n_rec
            draw_rate = sum(1 for o in recent_list if o["winner"] == "draw") / n_rec
            timeout_rate = sum(1 for o in recent_list if o["end_reason"] == "timeout") / n_rec
            mav_surv_rate = sum(1 for o in recent_list if o["mav_survived"]) / n_rec
            red_alive_final = np.mean([o["red_alive"] for o in recent_list])
            blue_alive_final = np.mean([o["blue_alive"] for o in recent_list])
        else:
            red_win_rate = blue_win_rate = draw_rate = timeout_rate = mav_surv_rate = 0.0
            red_alive_final = blue_alive_final = 0.0

        # Missile stats
        n_msl_ep = max(recent_missile_stats["total_episodes"], 1)
        red_msl_rate = recent_missile_stats["red_fired"] / n_msl_ep
        blue_msl_rate = recent_missile_stats["blue_fired"] / n_msl_ep
        msl_hit_count = recent_missile_stats["hit"]
        msl_hit_rate = msl_hit_count / n_msl_ep if n_msl_ep > 0 else 0.0

        csv_writer.writerow([
            iteration, total_steps, f'{avg_ret:.4f}',
            f'{avg_len:.1f}', f'{avg_red_alive:.2f}',
            f'{avg_blue_alive:.2f}',
            f'{stats["actor_loss"]:.6f}', f'{stats["critic_loss"]:.6f}',
            f'{stats["entropy"]:.6f}',
            f'{act_mean_abs:.6f}', f'{act_std:.6f}',
            f'{act_min:.6f}', f'{act_max:.6f}',
            f'{val_mean:.6f}', f'{val_std:.6f}',
            f'{act_sat:.6f}',
            f'{mav_act_abs:.6f}', f'{uav_act_abs:.6f}',
            f'{mav_sat:.6f}', f'{uav_sat:.6f}',
            f'{red_win_rate:.4f}', f'{blue_win_rate:.4f}',
            f'{draw_rate:.4f}', f'{timeout_rate:.4f}',
            f'{mav_surv_rate:.4f}',
            f'{red_alive_final:.2f}', f'{blue_alive_final:.2f}',
            f'{red_msl_rate:.2f}', f'{blue_msl_rate:.2f}',
            f'{msl_hit_count}', f'{msl_hit_rate:.4f}',
            int(nan_detected), args.opponent_policy,
            episodes_completed,
        ])
        csv_file.flush()

        if iteration % args.console_log_interval == 0:
            if args.total_env_steps is None:
                steps_text = str(total_steps)
                progress_text = ''
            else:
                steps_text = f'{total_steps}/{args.total_env_steps}'
                progress = 100.0 * min(total_steps, args.total_env_steps) / args.total_env_steps
                progress_text = f' progress={progress:.1f}%'
            print(
                f'[train] iter={iteration:04d}{progress_text} steps={steps_text} '
                f'ep={episodes_completed} ret={avg_ret:+.2f} len={avg_len:.0f} '
                f'red_alive={avg_red_alive:.1f} blue_alive={avg_blue_alive:.1f} '
                f'win_r={red_win_rate:.2f} win_b={blue_win_rate:.2f} '
                f'draw={draw_rate:.2f} mav_surv={mav_surv_rate:.2f} '
                f'act_abs={act_mean_abs:.2f} sat={act_sat:.2f} '
                f'ent={stats["entropy"]:.2f} nan={int(nan_detected)}',
                flush=True,
            )

        # Checkpoint saving
        if not args.no_save and iteration % args.save_interval == 0:
            ckpt_path = (f'{args.output_dir}/checkpoints/'
                         f'iter_{iteration:04d}.pt')
            torch.save(model.state_dict(), ckpt_path)

        # ---- Periodic eval during training ----
        if args.eval_during_training and eval_csv_writer is not None:
            if total_steps - last_eval_step >= args.eval_interval_steps:
                last_eval_step = total_steps
                # Save a temp checkpoint for eval
                tmp_ckpt = f'{args.output_dir}/_tmp_eval.pt'
                torch.save(model.state_dict(), tmp_ckpt)
                eval_json = f'{args.output_dir}/_tmp_eval.json'
                records = _run_eval(
                    tmp_ckpt, args.opponent_policy, args.obs_adapter_version,
                    args.train_eval_episodes, str(args.device), eval_configs,
                    eval_json,
                )
                if records:
                    for r in records:
                        eval_csv_writer.writerow([
                            total_steps, iteration, r.get("config", ""),
                            r.get("avg_return", 0.0), r.get("avg_length", 0.0),
                            r.get("red_win_rate", 0.0), r.get("blue_win_rate", 0.0),
                            r.get("draw_rate", 0.0), r.get("timeout_rate", 0.0),
                            r.get("mav_survival_rate", 0.0),
                            r.get("red_alive_final_mean", 0.0),
                            r.get("blue_alive_final_mean", 0.0),
                            r.get("nan_detected", True),
                            r.get("actor_dim_ok", False),
                            r.get("critic_dim_ok", False),
                        ])
                    eval_csv.flush()
                    score = _compute_best_score(records)
                    if score > best_score:
                        best_score = score
                        torch.save(model.state_dict(),
                                   f'{args.output_dir}/best/model.pt')
                        meta = {
                            'config': args.config, 'seed': args.seed,
                            'opponent_policy': args.opponent_policy,
                            'iterations_completed': iterations_completed,
                            'total_steps': total_steps,
                            'best_score': round(score, 6),
                            'obs_adapter_version': args.obs_adapter_version,
                            'actor_obs_dim': actor_obs_dim,
                            'critic_state_dim': critic_state_dim,
                            'actor_arch': args.actor_arch,
                            'observation_mode': obs_mode,
                            'ppo_epochs': args.ppo_epochs,
                            'entropy_coef': args.entropy_coef,
                            'actor_lr': args.actor_lr,
                            'critic_lr': args.critic_lr,
                            'rollout_length': args.rollout_length,
                        }
                        with open(f'{args.output_dir}/best/meta.json', 'w') as f:
                            json.dump(meta, f)
                        print(f'[train] best checkpoint saved (score={score:.4f})',
                              flush=True)
                # Clean up temp
                for p in [tmp_ckpt, eval_json]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

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
            'actor_arch': args.actor_arch,
            'observation_mode': obs_mode,
            'ppo_epochs': args.ppo_epochs,
            'entropy_coef': args.entropy_coef,
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'rollout_length': args.rollout_length,
            'best_score': round(best_score, 6) if best_score > -float("inf") else None,
        }
        with open(f'{args.output_dir}/latest/meta.json', 'w') as f:
            json.dump(meta, f)
        print(f'Saved {best_model_path}', flush=True)

    csv_file.close()
    if eval_csv is not None:
        eval_csv.close()
    env.close()


if __name__ == '__main__':
    main()
