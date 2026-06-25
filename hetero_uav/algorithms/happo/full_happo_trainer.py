"""Full HAPPO trainer: sequential agent-level update with correction factor M.

Aligns with ICLR 2022 HAPPO algorithm:
  - Random agent update order each PPO epoch
  - Correction factor M = M * ratio_after for each agent
  - Shared V-value critic updated after actor sequential updates
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


class FullHAPPOTrainer:
    """Full HAPPO trainer with independent per-agent actors + shared critic."""

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4,
                 clip_param=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=5,
                 gamma=0.99, gae_lambda=0.95,
                 seed=None, critic_update_after_actors=True):
        self.policy = policy
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_update_after_actors = critic_update_after_actors
        self.rng = np.random.default_rng(seed)

        num_agents = policy.num_agents
        self.actor_opts = [
            optim.Adam(
                list(policy.actors[i].parameters()) + [policy.action_log_stds[i]],
                lr=actor_lr,
            )
            for i in range(num_agents)
        ]
        self.critic_opt = optim.Adam(policy.critic.parameters(), lr=critic_lr)

    def update(self, buffer) -> dict:
        data = buffer.get(next(self.policy.parameters()).device)
        actor_obs = data["actor_obs"]       # [T, N, D]
        critic_state = data["critic_state"]  # [T, Dc]
        actions = data["actions"]            # [T, N, A]
        old_log_probs = data["old_log_probs"]  # [T, N]
        rewards = data["rewards"]            # [T, N]
        dones = data["dones"]                # [T, N]
        values = data["values"]              # [T]
        active = data["active_masks"]        # [T, N]
        env_ids = data.get("env_ids", torch.zeros_like(dones[:, 0], dtype=torch.long))

        T, N = actor_obs.shape[:2]

        # Team reward
        valid_count = active.sum(dim=-1).clamp(min=1)
        team_reward = (rewards * active).sum(dim=-1) / valid_count
        team_dones = dones[:, 0].float() if dones.ndim >= 2 else dones.float()

        # GAE
        next_values = data.get("next_values")
        if next_values is not None and not torch.isnan(next_values).any():
            advantages, returns = _compute_grouped_gae(
                team_reward, values, next_values, team_dones,
                env_ids, self.gamma, self.gae_lambda)
        else:
            with torch.no_grad():
                nv = self.policy.value(critic_state[-1:])
            av = torch.cat([values, nv])
            advantages, returns = _compute_gae(
                team_reward, av, team_dones, self.gamma, self.gae_lambda)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "actor_loss_per_agent": [0.0] * N,
            "entropy_per_agent": [0.0] * N,
            "approx_kl_per_agent": [0.0] * N,
        }

        for epoch in range(self.ppo_epochs):
            # Random agent update order
            order = list(self.rng.permutation(N))
            metrics["last_update_order"] = order

            # Init correction factor M
            M = advantages.detach().clone()  # [T]

            for idx_in_order, i in enumerate(order):
                valid_i = active[:, i] > 0.5
                if valid_i.sum() < 1:
                    continue

                obs_i = actor_obs[:, i, :]  # [T, D]
                act_i = actions[:, i, :]    # [T, A]
                old_lp_i = old_log_probs[:, i]  # [T]

                self.actor_opts[i].zero_grad()

                new_lp_i, entropy_i, _mean_i = self.policy.evaluate_agent_actions(
                    i, obs_i, act_i)

                ratio_i = (new_lp_i - old_lp_i.detach()).exp()

                adv = M
                surr1 = ratio_i * adv
                surr2 = torch.clamp(ratio_i, 1 - self.clip_param, 1 + self.clip_param) * adv
                policy_loss = -(torch.min(surr1, surr2) * valid_i).sum() / valid_i.sum().clamp(min=1)
                ent_mean = (entropy_i * valid_i).sum() / valid_i.sum().clamp(min=1)
                loss = policy_loss - self.entropy_coef * ent_mean
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]],
                    self.max_grad_norm,
                )
                self.actor_opts[i].step()

                metrics["actor_loss_per_agent"][i] = float(policy_loss.item())
                metrics["entropy_per_agent"][i] = float(ent_mean.item())
                kl_i = (old_lp_i - new_lp_i.detach()) * valid_i
                metrics["approx_kl_per_agent"][i] = float((kl_i.sum() / valid_i.sum().clamp(min=1)).item())

                # Update correction factor M = M * ratio_after
                with torch.no_grad():
                    after_lp_i, _, _ = self.policy.evaluate_agent_actions(i, obs_i, act_i)
                    ratio_after = (after_lp_i - old_lp_i).exp()
                    ratio_after = torch.where(valid_i, ratio_after, torch.ones_like(ratio_after))
                    M = M * ratio_after

            # Critic update
            self.critic_opt.zero_grad()
            new_values = self.policy.value(critic_state)
            critic_loss = F.mse_loss(new_values, returns) * self.value_coef
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

        # ── Aggregate metrics ──
        alive = active.sum(dim=0) > 0
        al = [i for i in range(N) if alive[i]]
        actor_loss_mean = float(np.mean([metrics["actor_loss_per_agent"][i] for i in al])) if al else 0.0
        ent_mean = float(np.mean([metrics["entropy_per_agent"][i] for i in al])) if al else 0.0
        kl_mean = float(np.mean([metrics["approx_kl_per_agent"][i] for i in al])) if al else 0.0

        log_std_vals = torch.stack([p.data for p in self.policy.action_log_stds])
        return {
            "actor_loss_mean": actor_loss_mean,
            "actor_loss_per_agent": metrics["actor_loss_per_agent"],
            "entropy_mean": ent_mean,
            "entropy_per_agent": metrics["entropy_per_agent"],
            "approx_kl_mean": kl_mean,
            "approx_kl_per_agent": metrics["approx_kl_per_agent"],
            "critic_loss": float(critic_loss.item()),
            "last_update_order": list(order),
            "action_log_std_min": float(log_std_vals.min().item()),
            "action_log_std_max": float(log_std_vals.max().item()),
            "action_log_std_mean": float(log_std_vals.mean().item()),
            "actor_loss_mav": metrics["actor_loss_per_agent"][0] if N > 0 else 0.0,
            "actor_loss_uav": float(np.mean(metrics["actor_loss_per_agent"][1:])) if N > 1 else 0.0,
            "entropy_mav": metrics["entropy_per_agent"][0] if N > 0 else 0.0,
            "entropy_uav": float(np.mean(metrics["entropy_per_agent"][1:])) if N > 1 else 0.0,
            "approx_kl_mav": metrics["approx_kl_per_agent"][0] if N > 0 else 0.0,
            "approx_kl_uav": float(np.mean(metrics["approx_kl_per_agent"][1:])) if N > 1 else 0.0,
            "mav_active_sample_count": int(alive[0].item()) if N > 0 and alive[0] else 0,
            "uav_active_sample_count": int(sum(1 for i in al if i > 0)),
            "mav_action_saturation_rate": 0.0,
            "uav_action_saturation_rate": 0.0,
        }


def _compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


def _compute_grouped_gae(rewards, values, next_values, dones, env_ids, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for eid in env_ids.unique():
        mask = env_ids == eid
        if mask.sum() == 0:
            continue
        r = rewards[mask]
        d = dones[mask]
        v = values[mask]
        nv = next_values[mask[-1:]] if mask[-1] < len(next_values) else torch.zeros(1, device=rewards.device)
        av, rt = _compute_gae(r, torch.cat([v, nv]), d, gamma, lam)
        advantages[mask] = av
        returns[mask] = rt
    return advantages, returns
