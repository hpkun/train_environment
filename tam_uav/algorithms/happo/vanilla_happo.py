"""Vanilla non-recurrent HAPPO with independent categorical actors."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, output_dim))


class VanillaHAPPOPolicy(nn.Module):
    """Local independent actors and one centralized state-value critic."""

    def __init__(self, agent_ids, role_by_agent, actor_obs_dim, critic_state_dim,
                 action_dim=4, action_levels=40, hidden_dim=128,
                 actor_sharing="independent"):
        super().__init__()
        self.agent_ids = list(agent_ids)
        self.role_by_agent = dict(role_by_agent)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.action_levels = int(action_levels)
        self.actor_sharing = actor_sharing
        if actor_sharing not in {"independent", "role_shared_ablation"}:
            raise ValueError("actor_sharing must be independent or role_shared_ablation")
        keys = self.agent_ids if actor_sharing == "independent" else sorted(set(role_by_agent.values()))
        self.actors = nn.ModuleDict({key: _mlp(
            self.actor_obs_dim, self.action_dim * self.action_levels, hidden_dim)
            for key in keys})
        self.critic = _mlp(self.critic_state_dim, 1, hidden_dim)

    def actor_key(self, agent_id):
        return agent_id if self.actor_sharing == "independent" else self.role_by_agent[agent_id]

    def logits(self, agent_id, obs):
        shape = obs.shape[:-1]
        raw = self.actors[self.actor_key(agent_id)](obs.reshape(-1, obs.shape[-1]))
        return raw.reshape(*shape, self.action_dim, self.action_levels)

    @staticmethod
    def _masked_logits(logits, available_actions=None):
        if available_actions is None:
            return logits
        mask = torch.as_tensor(available_actions, dtype=torch.bool, device=logits.device)
        if not mask.any(dim=-1).all():
            raise ValueError("each categorical head must have an available action")
        return logits.masked_fill(~mask, -1e9)

    def act(self, actor_obs, critic_state, available_actions=None, deterministic=False):
        device = next(self.parameters()).device
        obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        available = None if available_actions is None else torch.as_tensor(
            available_actions, dtype=torch.bool, device=device)
        actions, log_probs, entropies, logits_out = [], [], [], []
        for index, agent_id in enumerate(self.agent_ids):
            logits = self._masked_logits(
                self.logits(agent_id, obs[index]),
                None if available is None else available[index])
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action).sum(-1))
            entropies.append(dist.entropy().sum(-1))
            logits_out.append(logits)
        state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)
        return {
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "logits": torch.stack(logits_out),
            "value": self.value(state),
        }

    def evaluate_agent(self, agent_id, obs, actions, available_actions=None):
        logits = self._masked_logits(self.logits(agent_id, obs), available_actions)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()).sum(-1), dist.entropy().sum(-1), logits

    def value(self, states):
        return self.critic(states).squeeze(-1)


class VanillaHAPPORolloutBuffer:
    """Finite-horizon multi-agent buffer with explicit termination semantics."""

    def __init__(self, length, num_agents, obs_dim, state_dim,
                 action_dim=4, action_levels=40):
        shape = (length, num_agents)
        self.length, self.num_agents = int(length), int(num_agents)
        self.obs = np.zeros((*shape, obs_dim), np.float32)
        self.states = np.zeros((length, state_dim), np.float32)
        self.actions = np.zeros((*shape, action_dim), np.int64)
        self.old_log_probs = np.zeros(shape, np.float32)
        self.rewards = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)
        self.next_values = np.zeros(shape, np.float32)
        self.terminated = np.zeros(shape, np.float32)
        self.truncated = np.zeros(shape, np.float32)
        self.active_masks = np.zeros(shape, np.float32)
        self.available_actions = np.zeros((*shape, action_dim, action_levels), np.float32)
        self.agent_alive = np.zeros(shape, np.float32)
        self.episode_ids = np.zeros(length, np.int64)
        self.decision_steps = np.zeros(length, np.int64)
        self.policy_versions = np.zeros(length, np.int64)
        self.pos = 0

    def add(self, *, obs, state, actions, log_probs, rewards, value, next_value,
            terminated, truncated, active_masks, available_actions, agent_alive,
            episode_id, decision_step, policy_version):
        i = self.pos
        if i >= self.length:
            raise IndexError("rollout buffer is full")
        self.obs[i], self.states[i], self.actions[i] = obs, state, actions
        self.old_log_probs[i], self.rewards[i] = log_probs, rewards
        self.values[i] = np.broadcast_to(value, (self.num_agents,))
        self.next_values[i] = np.broadcast_to(next_value, (self.num_agents,))
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.active_masks[i], self.available_actions[i] = active_masks, available_actions
        self.agent_alive[i] = agent_alive
        self.episode_ids[i], self.decision_steps[i] = episode_id, decision_step
        self.policy_versions[i] = policy_version
        self.pos += 1

    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        n = self.pos
        advantages = np.zeros((n, self.num_agents), np.float32)
        last = np.zeros(self.num_agents, np.float32)
        for t in reversed(range(n)):
            bootstrap = 1.0 - self.terminated[t]
            continuation = (1.0 - np.maximum(self.terminated[t], self.truncated[t]))
            if t + 1 < n:
                continuation *= (self.episode_ids[t + 1] == self.episode_ids[t])
            delta = self.rewards[t] + gamma * self.next_values[t] * bootstrap - self.values[t]
            last = delta + gamma * gae_lambda * continuation * last
            last *= self.active_masks[t]
            advantages[t] = last
        returns = advantages + self.values[:n]
        return advantages, returns

    def tensors(self, device):
        n = self.pos
        return {name: torch.as_tensor(getattr(self, name)[:n], device=device)
                for name in ("obs", "states", "actions", "old_log_probs", "rewards",
                             "values", "next_values", "terminated", "truncated",
                             "active_masks", "available_actions", "agent_alive")}


@dataclass
class HAPPOUpdateResult:
    metrics: dict
    update_orders: list


class VanillaHAPPOTrainer:
    """Sequential HAPPO: A_i <- A_i * prod_{j<i}(pi_j/pi_j_old)."""

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=10.0,
                 ppo_epochs=2, minibatch_size=256, gamma=0.99,
                 gae_lambda=0.95, randomize_order=True, seed=0):
        self.policy = policy
        self.actor_optimizers = {
            key: torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for key, actor in policy.actors.items()}
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param, self.value_coef = clip_param, value_coef
        self.entropy_coef, self.max_grad_norm = entropy_coef, max_grad_norm
        self.ppo_epochs, self.minibatch_size = int(ppo_epochs), int(minibatch_size)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.randomize_order = randomize_order
        self.rng = np.random.default_rng(seed)
        self.update_count = 0

    @staticmethod
    def joint_ratio(new_log_prob, old_log_prob):
        return torch.exp(new_log_prob - old_log_prob)

    @staticmethod
    def update_factor(factor, new_log_prob, old_log_prob, active_mask):
        ratio = torch.exp(new_log_prob - old_log_prob).detach()
        return (factor * torch.where(active_mask > 0.5, ratio,
                                     torch.ones_like(ratio))).detach()

    def update(self, buffer):
        device = next(self.policy.parameters()).device
        data = buffer.tensors(device)
        advantages_np, returns_np = buffer.compute_gae(self.gamma, self.gae_lambda)
        advantages = torch.as_tensor(advantages_np, device=device)
        returns = torch.as_tensor(returns_np, device=device)
        active = data["active_masks"]
        valid_adv = advantages[active > 0.5]
        if valid_adv.numel() > 1:
            advantages = torch.where(active > 0.5,
                (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8),
                torch.zeros_like(advantages))
        metrics, orders = {}, []
        actor_losses = {aid: [] for aid in self.policy.agent_ids}
        entropies = {aid: [] for aid in self.policy.agent_ids}
        kls = {aid: [] for aid in self.policy.agent_ids}
        factors = []
        grad_norms = {aid: [] for aid in self.policy.agent_ids}
        clip_fractions = {aid: [] for aid in self.policy.agent_ids}
        critic_losses, critic_grad_norms = [], []

        for _ in range(self.ppo_epochs):
            order = list(range(len(self.policy.agent_ids)))
            if self.randomize_order:
                self.rng.shuffle(order)
            orders.append([self.policy.agent_ids[i] for i in order])
            factor = torch.ones(buffer.pos, device=device)
            for index in order:
                aid = self.policy.agent_ids[index]
                valid = active[:, index] > 0.5
                if not valid.any():
                    continue
                optimizer = self.actor_optimizers[self.policy.actor_key(aid)]
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                permutation = valid_indices[torch.as_tensor(
                    self.rng.permutation(len(valid_indices)), device=device)]
                for start in range(0, len(permutation), self.minibatch_size):
                    batch = permutation[start:start+self.minibatch_size]
                    optimizer.zero_grad()
                    new_logp, entropy, _ = self.policy.evaluate_agent(
                        aid, data["obs"][batch, index], data["actions"][batch, index],
                        data["available_actions"][batch, index])
                    ratio = self.joint_ratio(
                        new_logp, data["old_log_probs"][batch, index])
                    corrected_adv = advantages[batch, index] * factor[batch]
                    surr1 = ratio * corrected_adv
                    surr2 = torch.clamp(
                        ratio, 1-self.clip_param, 1+self.clip_param) * corrected_adv
                    loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(
                        self.policy.actors[self.policy.actor_key(aid)].parameters(),
                        self.max_grad_norm)
                    optimizer.step()
                    actor_losses[aid].append(float(loss.item()))
                    entropies[aid].append(float(entropy.mean().item()))
                    grad_norms[aid].append(float(grad))
                    clip_fractions[aid].append(float(
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()))
                with torch.no_grad():
                    post_logp, _, _ = self.policy.evaluate_agent(
                        aid, data["obs"][:, index], data["actions"][:, index],
                        data["available_actions"][:, index])
                factor = self.update_factor(
                    factor, post_logp, data["old_log_probs"][:, index], active[:, index])
                kls[aid].append(float((data["old_log_probs"][:, index][valid]
                                      - post_logp[valid]).mean().item()))
                factors.append(factor.detach().cpu())

            self.critic_optimizer.zero_grad()
            predicted = self.policy.value(data["states"])
            old_value = data["values"][:, 0]
            target = (returns * active).sum(-1) / active.sum(-1).clamp(min=1)
            clipped = old_value + (predicted - old_value).clamp(-self.clip_param, self.clip_param)
            value_loss = 0.5 * torch.maximum((predicted-target).pow(2),
                                              (clipped-target).pow(2)).mean()
            (self.value_coef * value_loss).backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(float(value_loss.item()))
            critic_grad_norms.append(float(critic_grad))

        for aid in self.policy.agent_ids:
            metrics[f"actor_loss/{aid}"] = float(np.mean(actor_losses[aid])) if actor_losses[aid] else 0.0
            metrics[f"entropy/{aid}"] = float(np.mean(entropies[aid])) if entropies[aid] else 0.0
            metrics[f"approx_kl/{aid}"] = float(np.mean(kls[aid])) if kls[aid] else 0.0
            metrics[f"gradient_norm/{aid}"] = float(np.mean(grad_norms[aid])) if grad_norms[aid] else 0.0
            metrics[f"clip_fraction/{aid}"] = float(np.mean(clip_fractions[aid])) if clip_fractions[aid] else 0.0
        all_factors = torch.cat(factors) if factors else torch.ones(1)
        metrics.update({
            "critic_loss": float(np.mean(critic_losses)),
            "critic_gradient_norm": float(np.mean(critic_grad_norms)),
            "advantage_mean": float(valid_adv.mean()) if valid_adv.numel() else 0.0,
            "advantage_std": float(valid_adv.std()) if valid_adv.numel() > 1 else 0.0,
            "importance_factor_mean": float(all_factors.mean()),
            "importance_factor_std": float(all_factors.std()),
            "importance_factor_min": float(all_factors.min()),
            "importance_factor_max": float(all_factors.max()),
            "value_mean": float(data["values"].mean()),
            "value_std": float(data["values"].std()),
        })
        self.update_count += 1
        return HAPPOUpdateResult(metrics, orders)
