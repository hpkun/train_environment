"""Strict Pure HAPPO runner for the isolated formal 3v2 contract."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ACTION_DIM, ENV_TYPE


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    parser.add_argument("--output-dir", default="outputs/formal_v1_smoke")
    parser.add_argument("--total-env-steps", type=int, default=2048)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _flat(obs, red_ids):
    return np.stack([obs[aid]["flat"] for aid in red_ids]).astype(np.float32)


def _save(policy, directory: Path, meta: dict):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), directory / "model.pt")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    args = _args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = Path(args.config)
    if not config.is_absolute(): config = ROOT / config
    env = make_env(str(config))
    if env.config.get("env_type") != ENV_TYPE or env.action_dim != ACTION_DIM:
        raise ValueError("formal runner accepts only hetero_3v2_pure_happo_v1 Box(3)")
    device = torch.device(args.device)
    policy = PureHAPPOPolicy(env.actor_obs_dim, env.critic_state_dim, ACTION_DIM,
                             len(env.red_ids), credit_mode="shared_alive_team_mean").to(device)
    trainer = PureHAPPOTrainer(policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
                               entropy_coef=0.01, max_grad_norm=10.0, ppo_epochs=5,
                               gamma=0.99, gae_lambda=0.95, seed=args.seed)
    output = Path(args.output_dir)
    if not output.is_absolute(): output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    meta = {"formal_contract": ENV_TYPE, "policy_arch": "pure_happo",
            "credit_mode": "shared_alive_team_mean", "actor_obs_dim": env.actor_obs_dim,
            "critic_state_dim": env.critic_state_dim, "action_dim": ACTION_DIM,
            "num_agents": len(env.red_ids), "config": str(config)}
    _save(policy, output / "initial", {**meta, "total_env_steps_actual": 0})
    log_path = output / "train_log.csv"
    fields = ["iteration", "total_steps", "avg_reward", "critic_loss", "finite"]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        obs, info = env.reset(seed=args.seed)
        total_steps = 0; iteration = 0
        while total_steps < args.total_env_steps:
            length = min(args.rollout_length, args.total_env_steps - total_steps)
            buffer = HAPPORolloutBuffer(length, len(env.red_ids), env.actor_obs_dim,
                                        env.critic_state_dim, ACTION_DIM, [0, 1, 1])
            rewards_seen = []
            for _ in range(length):
                actor_obs = _flat(obs, env.red_ids)
                critic = np.asarray(info["critic_state"], np.float32)
                active = np.asarray(info["active_mask"], np.float32)
                with torch.no_grad():
                    result = policy.act(actor_obs, critic_state=critic)
                actions = result["action"].detach().cpu().numpy().astype(np.float32)
                actions *= active[:, None]
                next_obs, rewards, terms, truncs, next_info = env.step(
                    {aid: actions[i] for i, aid in enumerate(env.red_ids)})
                done = float(next_info["team_done"])
                with torch.no_grad():
                    next_value = policy.value(next_info["critic_state"]).detach().cpu().numpy()
                reward_vec = np.asarray([rewards[aid] for aid in env.red_ids], np.float32)
                buffer.store(actor_obs, critic, actions,
                             result["log_prob"].detach().cpu().numpy(), reward_vec,
                             np.full(len(env.red_ids), done, np.float32),
                             result["value"].detach().cpu().numpy(), active,
                             next_value=next_value)
                rewards_seen.append(float(reward_vec.mean()))
                obs, info = next_obs, next_info
                total_steps += 1
                if done:
                    obs, info = env.reset(seed=args.seed + total_steps)
            metrics = trainer.update(buffer)
            iteration += 1
            row = {"iteration": iteration, "total_steps": total_steps,
                   "avg_reward": float(np.mean(rewards_seen)),
                   "critic_loss": float(metrics.get("critic_loss", 0.0)),
                   "finite": int(all(torch.isfinite(parameter).all().item()
                                     for parameter in policy.parameters()))}
            writer.writerow(row); handle.flush()
            print(f"[formal-v1] it={iteration:04d} steps={total_steps}/{args.total_env_steps} "
                  f"reward={row['avg_reward']:+.4f}", flush=True)
    _save(policy, output / "latest", {**meta, "total_env_steps_actual": total_steps})
    env.close()


if __name__ == "__main__":
    main()
