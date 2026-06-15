"""HAPPO-style sequential update for reference v0."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mappo.utils import compute_gae
from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


def _compute_grouped_gae(team_reward, values, next_values, team_dones,
                         env_ids, gamma: float, gae_lambda: float):
    advantages = torch.zeros_like(team_reward)
    returns = torch.zeros_like(team_reward)
    for env_id in torch.unique(env_ids):
        idx = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        last_gae = torch.zeros((), device=team_reward.device)
        for pos in reversed(idx.tolist()):
            nonterminal = 1.0 - team_dones[pos]
            delta = team_reward[pos] + gamma * next_values[pos] * nonterminal - values[pos]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[pos] = last_gae
            returns[pos] = advantages[pos] + values[pos]
    return advantages, returns


def _wrapped_heading_error(pred_heading: torch.Tensor, target_heading: torch.Tensor) -> torch.Tensor:
    return torch.remainder(pred_heading - target_heading + 1.0, 2.0) - 1.0


def _uav_imitation_loss(policy, actor_obs: torch.Tensor, oracle_actions: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "uav_imitation_loss_from_flat"):
        return policy.uav_imitation_loss_from_flat(actor_obs, oracle_actions)
    pred = torch.clamp(policy.uav_actor(actor_obs), -0.999, 0.999)
    error = pred - oracle_actions
    error = error.clone()
    error[..., 1] = _wrapped_heading_error(pred[..., 1], oracle_actions[..., 1])
    return torch.mean(error ** 2)


class HAPPOReferenceTrainer:
    """Simplified HAPPO-style trainer.

    This keeps the HAPPO sequential role-wise update structure but uses a v0
    simplified correction factor. It is not a full TAM-HAPPO implementation.
    """

    def __init__(self, policy, actor_lr=2e-4, critic_lr=5e-4,
                 clip_param=0.2, entropy_coef=0.02, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=4, gamma=0.99,
                 gae_lambda=0.95):
        self.policy = policy
        self.actor_shared_params = (
            list(policy.actor_shared_parameters())
            if hasattr(policy, "actor_shared_parameters")
            else []
        )
        self.mav_opt = torch.optim.Adam(
            self.actor_shared_params
            + list(policy.mav_actor.parameters())
            + [policy.action_log_std_mav],
            lr=actor_lr,
        )
        self.uav_opt = torch.optim.Adam(
            self.actor_shared_params
            + list(policy.uav_actor.parameters())
            + [policy.action_log_std_uav],
            lr=actor_lr,
        )
        self.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _actor_update(self, data, advantages, role_id: int, optimizer):
        actor_obs = data["actor_obs"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        active_masks = data["active_masks"]
        role_ids = data["role_ids"]
        T, N = active_masks.shape
        role_mask = (role_ids.view(1, N).expand(T, N) == role_id).float()
        valid = active_masks * role_mask
        valid_sample_count = float(valid.sum().item())
        if valid_sample_count <= 0:
            return 0.0, 0.0, 0.0, 0.0

        optimizer.zero_grad()
        repeated_roles = role_ids.view(1, N).expand(T, N)
        eval_kwargs = {}
        if "rnn_hidden" in data and data["rnn_hidden"] is not None:
            eval_kwargs["rnn_hidden"] = data["rnn_hidden"]
        log_prob, entropy, _values, _mean, _roles = self.policy.evaluate_actions(
            actor_obs, repeated_roles, data["critic_state"], actions, **eval_kwargs)
        ratio = torch.exp(log_prob - old_log_probs)
        adv = advantages.unsqueeze(-1)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -(torch.min(surr1, surr2) * valid).sum() / valid.sum().clamp(min=1)
        entropy_mean = (entropy * valid).sum() / valid.sum().clamp(min=1)
        loss = policy_loss - self.entropy_coef * entropy_mean
        loss.backward()
        params = (self.actor_shared_params + list(self.policy.mav_actor.parameters()) + [self.policy.action_log_std_mav]
                  if role_id == MAV_ROLE_ID
                  else self.actor_shared_params + list(self.policy.uav_actor.parameters()) + [self.policy.action_log_std_uav])
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        approx_kl = ((old_log_probs - log_prob) * valid).sum() / valid.sum().clamp(min=1)
        return (
            float(policy_loss.item()),
            float(entropy_mean.item()),
            float(approx_kl.item()),
            valid_sample_count,
        )

    def update(self, buffer, uav_imitation_batch=None, uav_imitation_coef: float = 0.0):
        data = buffer.get(next(self.policy.parameters()).device)
        rewards = data["rewards"]
        active = data["active_masks"]
        values = data["values"]
        dones = data["dones"]
        valid_count = active.sum(dim=-1).clamp(min=1)
        team_reward = (rewards * active).sum(dim=-1) / valid_count
        team_dones = dones[:, 0].float()
        next_values = data.get("next_values")
        if next_values is not None and not torch.isnan(next_values).any():
            advantages, returns = _compute_grouped_gae(
                team_reward, values, next_values, team_dones,
                data["env_ids"], self.gamma, self.gae_lambda)
        else:
            with torch.no_grad():
                next_val = self.policy.value(data["critic_state"][-1:])
            all_values = torch.cat([values, next_val])
            advantages, returns = compute_gae(
                team_reward, all_values, team_dones, self.gamma, self.gae_lambda)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_mav, actor_loss_uav = [], []
        entropy_mav, entropy_uav = [], []
        kl_mav, kl_uav = [], []
        valid_mav, valid_uav = [], []
        critic_losses = []
        imitation_losses = []

        for _ in range(self.ppo_epochs):
            self.critic_opt.zero_grad()
            new_values = self.policy.value(data["critic_state"])
            critic_loss = F.mse_loss(new_values, returns) * self.value_coef
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.item()))

            m_loss, m_ent, m_kl, m_valid = self._actor_update(
                data, advantages, MAV_ROLE_ID, self.mav_opt)
            u_loss, u_ent, u_kl, u_valid = self._actor_update(
                data, advantages, UAV_ROLE_ID, self.uav_opt)
            if uav_imitation_batch is not None and uav_imitation_coef > 0.0:
                obs_batch, action_batch = uav_imitation_batch
                self.uav_opt.zero_grad()
                imitation_loss = _uav_imitation_loss(
                    self.policy,
                    obs_batch.to(next(self.policy.parameters()).device),
                    action_batch.to(next(self.policy.parameters()).device),
                )
                (float(uav_imitation_coef) * imitation_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_shared_params
                    + list(self.policy.uav_actor.parameters())
                    + [self.policy.action_log_std_uav],
                    self.max_grad_norm,
                )
                self.uav_opt.step()
                imitation_losses.append(float(imitation_loss.item()))
            actor_loss_mav.append(m_loss)
            actor_loss_uav.append(u_loss)
            entropy_mav.append(m_ent)
            entropy_uav.append(u_ent)
            kl_mav.append(m_kl)
            kl_uav.append(u_kl)
            valid_mav.append(m_valid)
            valid_uav.append(u_valid)

        actions = data["actions"].detach()
        roles = data["role_ids"].detach().cpu().numpy()
        abs_actions = torch.abs(actions)
        sat = (abs_actions >= 0.999).float()
        mav_mask = torch.as_tensor(roles == MAV_ROLE_ID, device=actions.device).view(1, -1, 1)
        uav_mask = torch.as_tensor(roles == UAV_ROLE_ID, device=actions.device).view(1, -1, 1)
        mav_sat = float(sat.masked_select(mav_mask.expand_as(sat)).mean().item()) if mav_mask.any() else 0.0
        uav_sat = float(sat.masked_select(uav_mask.expand_as(sat)).mean().item()) if uav_mask.any() else 0.0

        mav_log_std = self.policy.action_log_std_mav.detach()
        uav_log_std = self.policy.action_log_std_uav.detach()
        return {
            "actor_loss_mav": float(np.mean(actor_loss_mav)),
            "actor_loss_uav": float(np.mean(actor_loss_uav)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy_mav": float(np.mean(entropy_mav)),
            "entropy_uav": float(np.mean(entropy_uav)),
            "entropy_mav_valid_count": float(np.mean(valid_mav)),
            "entropy_uav_valid_count": float(np.mean(valid_uav)),
            "mav_active_sample_count": float(np.mean(valid_mav)),
            "uav_active_sample_count": float(np.mean(valid_uav)),
            "action_log_std_mav_min": float(mav_log_std.min().item()),
            "action_log_std_mav_max": float(mav_log_std.max().item()),
            "action_log_std_mav_mean": float(mav_log_std.mean().item()),
            "action_log_std_uav_min": float(uav_log_std.min().item()),
            "action_log_std_uav_max": float(uav_log_std.max().item()),
            "action_log_std_uav_mean": float(uav_log_std.mean().item()),
            "approx_kl_mav": float(np.mean(kl_mav)),
            "approx_kl_uav": float(np.mean(kl_uav)),
            "action_saturation_rate": float(sat.mean().item()),
            "mav_action_saturation_rate": mav_sat,
            "uav_action_saturation_rate": uav_sat,
            "uav_imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
        }
