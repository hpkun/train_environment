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
        self.mav_opt = torch.optim.Adam(
            list(policy.mav_actor.parameters()) + [policy.action_log_std_mav],
            lr=actor_lr,
        )
        self.uav_opt = torch.optim.Adam(
            list(policy.uav_actor.parameters()) + [policy.action_log_std_uav],
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
        if valid.sum().item() <= 0:
            return 0.0, 0.0, 0.0

        optimizer.zero_grad()
        repeated_roles = role_ids.view(1, N).expand(T, N)
        log_prob, entropy, _values, _mean, _roles = self.policy.evaluate_actions(
            actor_obs, repeated_roles, data["critic_state"], actions)
        ratio = torch.exp(log_prob - old_log_probs)
        adv = advantages.unsqueeze(-1)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -(torch.min(surr1, surr2) * valid).sum() / valid.sum().clamp(min=1)
        entropy_mean = (entropy * valid).sum() / valid.sum().clamp(min=1)
        loss = policy_loss - self.entropy_coef * entropy_mean
        loss.backward()
        params = (list(self.policy.mav_actor.parameters()) + [self.policy.action_log_std_mav]
                  if role_id == MAV_ROLE_ID
                  else list(self.policy.uav_actor.parameters()) + [self.policy.action_log_std_uav])
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        approx_kl = ((old_log_probs - log_prob) * valid).sum() / valid.sum().clamp(min=1)
        return float(policy_loss.item()), float(entropy_mean.item()), float(approx_kl.item())

    def update(self, buffer):
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
        critic_losses = []

        for _ in range(self.ppo_epochs):
            self.critic_opt.zero_grad()
            new_values = self.policy.value(data["critic_state"])
            critic_loss = F.mse_loss(new_values, returns) * self.value_coef
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.item()))

            m_loss, m_ent, m_kl = self._actor_update(data, advantages, MAV_ROLE_ID, self.mav_opt)
            u_loss, u_ent, u_kl = self._actor_update(data, advantages, UAV_ROLE_ID, self.uav_opt)
            actor_loss_mav.append(m_loss)
            actor_loss_uav.append(u_loss)
            entropy_mav.append(m_ent)
            entropy_uav.append(u_ent)
            kl_mav.append(m_kl)
            kl_uav.append(u_kl)

        actions = data["actions"].detach()
        roles = data["role_ids"].detach().cpu().numpy()
        abs_actions = torch.abs(actions)
        sat = (abs_actions >= 0.999).float()
        mav_mask = torch.as_tensor(roles == MAV_ROLE_ID, device=actions.device).view(1, -1, 1)
        uav_mask = torch.as_tensor(roles == UAV_ROLE_ID, device=actions.device).view(1, -1, 1)
        mav_sat = float(sat.masked_select(mav_mask.expand_as(sat)).mean().item()) if mav_mask.any() else 0.0
        uav_sat = float(sat.masked_select(uav_mask.expand_as(sat)).mean().item()) if uav_mask.any() else 0.0

        return {
            "actor_loss_mav": float(np.mean(actor_loss_mav)),
            "actor_loss_uav": float(np.mean(actor_loss_uav)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy_mav": float(np.mean(entropy_mav)),
            "entropy_uav": float(np.mean(entropy_uav)),
            "approx_kl_mav": float(np.mean(kl_mav)),
            "approx_kl_uav": float(np.mean(kl_uav)),
            "action_saturation_rate": float(sat.mean().item()),
            "mav_action_saturation_rate": mav_sat,
            "uav_action_saturation_rate": uav_sat,
        }
