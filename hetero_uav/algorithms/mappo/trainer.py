"""PPO update logic for shared-policy MAPPO baseline."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .utils import compute_gae


class PPOTrainer:
    """Train MAPPO actor-critic with simple batch PPO."""

    def __init__(self, model, lr_actor=5e-4, lr_critic=5e-4,
                 clip_param=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=4,
                 gamma=0.99, gae_lambda=0.95):
        self.model = model
        self.actor_opt = torch.optim.Adam(model.actor.parameters(), lr=lr_actor)
        self.actor_opt.add_param_group(
            {'params': [model.action_log_std], 'lr': lr_actor})
        self.critic_opt = torch.optim.Adam(model.critic.parameters(), lr=lr_critic)

        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def update(self, buffer):
        (actor_obs, critic_state, actions, old_log_probs,
         rewards, dones, values, red_valid) = buffer.get(
            next(self.model.parameters()).device)

        T, num_red = rewards.shape
        # Use mean reward over valid agents as team reward
        valid_count = red_valid[:, :num_red].sum(dim=-1).clamp(min=1)
        team_reward = (rewards * red_valid[:, :num_red]).sum(dim=-1) / valid_count

        # team value + bootstrap
        with torch.no_grad():
            next_val = self.model.critic(critic_state[-1:]).squeeze(-1)
        team_values = values  # scalar per step
        all_values = torch.cat([team_values, next_val]).to(rewards.device)

        team_dones = (dones.sum(dim=-1) > 0).float()

        advantages, returns = compute_gae(
            team_reward, all_values, team_dones,
            self.gamma, self.gae_lambda)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.ppo_epochs):
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()

            new_log_prob, entropy, new_values = self.model.evaluate_actions(
                actor_obs.view(-1, self.model.actor_obs_dim),
                critic_state,
                actions.view(-1, self.model.action_dim),
            )
            new_log_prob = new_log_prob.view(T, num_red)
            entropy = entropy.view(T, num_red)
            new_values = new_values.squeeze(-1)

            # Mask invalid agents
            valid = red_valid[:, :num_red]
            ratio = torch.exp(new_log_prob - old_log_probs)
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)                     * advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * valid).sum() / valid.sum().clamp(min=1)

            value_loss = F.mse_loss(new_values, returns) * self.value_coef

            ent_loss = -(entropy * valid).sum() / valid.sum().clamp(min=1)

            loss = policy_loss + value_loss + self.entropy_coef * ent_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()

            actor_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
            entropies.append(-ent_loss.item())

        return {
            'actor_loss': float(np.mean(actor_losses)),
            'critic_loss': float(np.mean(critic_losses)),
            'entropy': float(np.mean(entropies)),
        }
