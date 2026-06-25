"""Paper-aligned HAPPO trainer: sequential per-agent update + correction factor M.

ICLR 2022 HAPPO algorithm:
  - Random agent update order each PPO epoch
  - Correction factor M = M * ratio_after for each agent
  - Shared V-value critic updated after actor sequential updates
  - Grouped GAE by env_id for parallel environments

Does NOT include: TAM-HAPPO, GRU, attention, masks, BRMA-MAPPO.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


def _compute_grouped_gae(rewards, values, next_values, dones, env_ids, gamma, lam):
    """Compute GAE grouped by env_id for parallel envs.

    Each env trajectory gets its own backward pass; next_values[pos] is the
    bootstrap value for the step after the trajectory ends.
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for eid in torch.unique(env_ids):
        idx = torch.nonzero(env_ids == eid, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        gae = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        pos_list = idx.tolist()
        for j, pos in enumerate(reversed(pos_list)):
            # Bootstrap: V(s_{t+1}) = values[t+1] if t+1 is in trajectory, else next_value
            if pos + 1 < len(values) and pos + 1 in pos_list:
                nv = values[pos + 1]
            else:
                nv = next_values[0]
            nonterminal = 1.0 - dones[pos]
            delta = rewards[pos] + gamma * nv * nonterminal - values[pos]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[pos] = gae
            returns[pos] = gae + values[pos]
    return advantages, returns


class FullHAPPOTrainer:
    """Paper-aligned full HAPPO trainer."""

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4,
                 clip_param=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=5,
                 gamma=0.99, gae_lambda=0.95,
                 seed=None):
        self.policy = policy
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rng = np.random.default_rng(seed)

        N = policy.num_agents
        self.actor_opts = [
            optim.Adam(
                list(policy.actors[i].parameters()) + [policy.action_log_stds[i]],
                lr=actor_lr,
            )
            for i in range(N)
        ]
        self.critic_opt = optim.Adam(policy.critic.parameters(), lr=critic_lr)

    def update(self, buffer) -> dict:
        data = buffer.get(next(self.policy.parameters()).device)
        actor_obs = data["actor_obs"]          # [T, N, D]
        critic_state = data["critic_state"]    # [T, Dc]
        actions = data["actions"]              # [T, N, A]
        old_log_probs = data["old_log_probs"]  # [T, N]
        rewards = data["rewards"]              # [T, N]
        dones = data["dones"]                  # [T, N]
        values = data["values"]                # [T]
        active = data["active_masks"]          # [T, N]

        T, N = actor_obs.shape[:2]

        # Numerical safety: check inputs are finite
        for name, tensor in [("actor_obs", actor_obs), ("critic_state", critic_state),
                              ("actions", actions), ("old_log_probs", old_log_probs),
                              ("rewards", rewards), ("dones", dones), ("values", values)]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"HAPPO trainer: non-finite {name} in buffer")

        # Team reward = average over active agents
        valid_count = active.sum(dim=-1).clamp(min=1)
        team_reward = (rewards * active).sum(dim=-1) / valid_count
        team_dones = dones[:, 0].float()

        # GAE with grouped env_id
        env_ids = data.get("env_ids", torch.zeros(T, dtype=torch.long, device=rewards.device))
        next_values = data.get("next_values")
        if next_values is not None and not torch.isnan(next_values).any():
            nv_ok = next_values
        else:
            with torch.no_grad():
                nv_ok = self.policy.value(critic_state[-1:])
        advantages, returns = _compute_grouped_gae(
            team_reward, values, nv_ok, team_dones, env_ids, self.gamma, self.gae_lambda)

        if not torch.isfinite(advantages).all():
            raise ValueError("HAPPO trainer: non-finite advantages after GAE")
        if not torch.isfinite(returns).all():
            raise ValueError("HAPPO trainer: non-finite returns after GAE")

        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "actor_loss_per_agent": [0.0] * N,
            "entropy_per_agent": [0.0] * N,
            "approx_kl_per_agent": [0.0] * N,
            "ratio_after_mean_per_agent": [0.0] * N,
            "m_abs_mean_after_each_agent": [],
            "valid_sample_count_per_agent": [0] * N,
        }

        for epoch in range(self.ppo_epochs):
            order = list(self.rng.permutation(N))
            metrics["last_update_order"] = order

            M = advantages.detach().clone()

            for idx_in_order, i in enumerate(order):
                valid_i = active[:, i] > 0.5
                n_valid = int(valid_i.sum().item())
                metrics["valid_sample_count_per_agent"][i] = n_valid
                if n_valid < 1:
                    metrics["ratio_after_mean_per_agent"][i] = 0.0
                    continue

                obs_i = actor_obs[:, i, :]
                act_i = actions[:, i, :]
                old_lp_i = old_log_probs[:, i]

                self.actor_opts[i].zero_grad()

                new_lp_i, entropy_i, _mean_i = self.policy.evaluate_agent_actions(i, obs_i, act_i)

                if not torch.isfinite(new_lp_i).all():
                    raise ValueError(f"HAPPO: non-finite new_lp for agent {i} epoch {epoch}")
                if not torch.isfinite(entropy_i).all():
                    raise ValueError(f"HAPPO: non-finite entropy for agent {i} epoch {epoch}")

                ratio_i = (new_lp_i - old_lp_i.detach()).exp()
                if not torch.isfinite(ratio_i).all():
                    raise ValueError(f"HAPPO: non-finite ratio_i for agent {i} epoch {epoch}")

                adv = M
                surr1 = ratio_i * adv
                surr2 = torch.clamp(ratio_i, 1 - self.clip_param, 1 + self.clip_param) * adv
                valid_f = valid_i.float()
                policy_loss = -(torch.min(surr1, surr2) * valid_f).sum() / valid_f.sum().clamp(min=1)
                ent_mean = (entropy_i * valid_f).sum() / valid_f.sum().clamp(min=1)
                loss = policy_loss - self.entropy_coef * ent_mean
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]],
                    self.max_grad_norm,
                )
                self.actor_opts[i].step()

                metrics["actor_loss_per_agent"][i] = float(policy_loss.item())
                metrics["entropy_per_agent"][i] = float(ent_mean.item())
                kl_i = (old_lp_i - new_lp_i.detach()) * valid_f
                metrics["approx_kl_per_agent"][i] = float((kl_i.sum() / valid_f.sum().clamp(min=1)).item())

                # Update correction factor M = M * ratio_after
                with torch.no_grad():
                    after_lp_i, _, _ = self.policy.evaluate_agent_actions(i, obs_i, act_i)
                    ratio_after = (after_lp_i - old_lp_i).exp()
                    if not torch.isfinite(ratio_after).all():
                        raise ValueError(f"HAPPO: non-finite ratio_after for agent {i} epoch {epoch}")
                    ratio_after = torch.where(valid_i, ratio_after, torch.ones_like(ratio_after))
                    metrics["ratio_after_mean_per_agent"][i] = float(ratio_after[valid_i].mean().item())
                    M = (M * ratio_after).detach()

                metrics["m_abs_mean_after_each_agent"].append(float(M.abs().mean().item()))

        # Critic update
        self.critic_opt.zero_grad()
        new_values = self.policy.value(critic_state)
        critic_loss = F.mse_loss(new_values, returns) * self.value_coef
        if not torch.isfinite(critic_loss):
            raise ValueError("HAPPO: non-finite critic_loss")
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # Action saturation
        with torch.no_grad():
            _, _, _, means_all = self.policy.evaluate_actions(actor_obs, critic_state, actions)
            mav_sat = float((means_all[:, 0, :].abs() >= 0.999).float().mean().item()) if N > 0 else 0.0
            uav_sat = float((means_all[:, 1:, :].abs() >= 0.999).float().mean().item()) if N > 1 else 0.0

        log_std_vals = torch.stack([p.data for p in self.policy.action_log_stds])
        vsc = metrics["valid_sample_count_per_agent"]
        return {
            "actor_loss_mean": float(np.mean([metrics["actor_loss_per_agent"][i] for i in range(N) if vsc[i] > 0])) if any(v > 0 for v in vsc) else 0.0,
            "actor_loss_per_agent": metrics["actor_loss_per_agent"],
            "entropy_mean": float(np.mean([metrics["entropy_per_agent"][i] for i in range(N) if vsc[i] > 0])) if any(v > 0 for v in vsc) else 0.0,
            "entropy_per_agent": metrics["entropy_per_agent"],
            "approx_kl_mean": float(np.mean([metrics["approx_kl_per_agent"][i] for i in range(N) if vsc[i] > 0])) if any(v > 0 for v in vsc) else 0.0,
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
            "mav_active_sample_count": vsc[0] if N > 0 else 0,
            "uav_active_sample_count": sum(vsc[1:]) if N > 1 else 0,
            "mav_action_saturation_rate": mav_sat,
            "uav_action_saturation_rate": uav_sat,
            "ratio_after_mean_per_agent": metrics["ratio_after_mean_per_agent"],
            "m_abs_mean_after_each_agent": metrics["m_abs_mean_after_each_agent"],
            "valid_sample_count_per_agent": vsc,
            "mask_keep_ratio": 1.0,
            "mask_entropy": 0.0,
            "masked_entity_count": 0.0,
        }
