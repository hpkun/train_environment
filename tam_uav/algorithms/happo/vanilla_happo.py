"""Heterogeneous-reward vanilla HAPPO with independent categorical actors."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


PAPER_ACTOR_HIDDEN_SIZES = (256, 128)
PAPER_CRITIC_HIDDEN_SIZES = (256, 128)


def _hidden_sizes(explicit, hidden_dim, published_default):
    sizes = (tuple(int(value) for value in explicit) if explicit is not None
             else (int(hidden_dim), int(hidden_dim)) if hidden_dim is not None
             else tuple(published_default))
    if len(sizes) != 2 or any(value <= 0 for value in sizes):
        raise ValueError("feedforward hidden sizes must contain two positive widths")
    return sizes


def _mlp(input_dim, output_dim, hidden_sizes):
    first, second = hidden_sizes
    return nn.Sequential(nn.Linear(input_dim, first), nn.Tanh(),
                         nn.Linear(first, second), nn.Tanh(),
                         nn.Linear(second, output_dim))


def huber_loss(error, delta):
    """Elementwise Huber loss with an explicit positive delta."""
    if delta <= 0:
        raise ValueError("huber delta must be positive")
    absolute = error.abs()
    delta_tensor = torch.as_tensor(delta, dtype=error.dtype, device=error.device)
    return torch.where(absolute <= delta_tensor,
                       0.5 * error.pow(2),
                       delta_tensor * (absolute - 0.5 * delta_tensor))


def clipped_value_loss(predicted, old_value, returns, clip_param,
                       value_loss_type="clipped_huber", huber_delta=10.0):
    """Elementwise maximum of unclipped and value-clipped critic losses."""
    clipped = old_value + (predicted - old_value).clamp(-clip_param, clip_param)
    errors = predicted - returns
    clipped_errors = clipped - returns
    if value_loss_type == "clipped_huber":
        return torch.maximum(huber_loss(errors, huber_delta),
                             huber_loss(clipped_errors, huber_delta))
    if value_loss_type == "legacy_clipped_mse":
        return 0.5 * torch.maximum(errors.pow(2), clipped_errors.pow(2))
    raise ValueError(f"unsupported value_loss_type {value_loss_type!r}")


class HeterogeneousCentralCritic(nn.Module):
    """Shared centralized backbone with one scalar value head per agent."""

    def __init__(self, state_dim, agent_ids, hidden_sizes):
        super().__init__()
        first, second = hidden_sizes
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, first), nn.Tanh(),
            nn.Linear(first, second), nn.Tanh())
        self.heads = nn.ModuleDict({aid: nn.Linear(second, 1) for aid in agent_ids})

    def forward(self, states):
        features = self.backbone(states)
        return torch.cat([head(features) for head in self.heads.values()], dim=-1)


class VanillaHAPPOPolicy(nn.Module):
    """Independent local actors and a heterogeneous centralized critic."""

    def __init__(self, agent_ids, role_by_agent, actor_obs_dim, critic_state_dim,
                 action_dim=4, action_levels=40, hidden_dim=None,
                 actor_sharing="independent", actor_hidden_sizes=None,
                 critic_hidden_sizes=None):
        super().__init__()
        if actor_sharing == "role_shared_ablation":
            warnings.warn(
                "role_shared_ablation is deprecated; use parameter_sharing_ppo_ablation",
                FutureWarning, stacklevel=2)
            actor_sharing = "parameter_sharing_ppo_ablation"
        if actor_sharing not in {"independent", "parameter_sharing_ppo_ablation"}:
            raise ValueError("unsupported actor sharing mode")
        self.agent_ids = list(agent_ids)
        self.role_by_agent = dict(role_by_agent)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.action_levels = int(action_levels)
        self.actor_hidden_sizes = _hidden_sizes(
            actor_hidden_sizes, hidden_dim, PAPER_ACTOR_HIDDEN_SIZES)
        self.critic_hidden_sizes = _hidden_sizes(
            critic_hidden_sizes, hidden_dim, PAPER_CRITIC_HIDDEN_SIZES)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None
                              else self.actor_hidden_sizes[-1])
        self.actor_sharing = actor_sharing
        keys = (self.agent_ids if actor_sharing == "independent"
                else sorted(set(self.role_by_agent.values())))
        self.actors = nn.ModuleDict({key: _mlp(
            self.actor_obs_dim, self.action_dim * self.action_levels,
            self.actor_hidden_sizes)
            for key in keys})
        self.critic = HeterogeneousCentralCritic(
            self.critic_state_dim, self.agent_ids, self.critic_hidden_sizes)

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
        return {"actions": torch.stack(actions),
                "log_probs": torch.stack(log_probs),
                "entropies": torch.stack(entropies),
                "logits": torch.stack(logits_out),
                "value": self.value(state)}

    def evaluate_agent(self, agent_id, obs, actions, available_actions=None):
        logits = self._masked_logits(self.logits(agent_id, obs), available_actions)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()).sum(-1), dist.entropy().sum(-1), logits

    def value(self, states):
        return self.critic(states)


class VanillaHAPPORolloutBuffer:
    """Finite-horizon buffer whose reward/value tensors are always ``[T, N]``."""

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
        value = np.asarray(value, dtype=np.float32)
        next_value = np.asarray(next_value, dtype=np.float32)
        if value.shape != (self.num_agents,) or next_value.shape != (self.num_agents,):
            raise ValueError("value and next_value must each have shape [num_agents]")
        self.obs[i], self.states[i], self.actions[i] = obs, state, actions
        self.old_log_probs[i], self.rewards[i] = log_probs, rewards
        self.values[i], self.next_values[i] = value, next_value
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
            continuation = 1.0 - np.maximum(self.terminated[t], self.truncated[t])
            if t + 1 < n:
                continuation *= self.episode_ids[t + 1] == self.episode_ids[t]
            delta = self.rewards[t] + gamma * self.next_values[t] * bootstrap - self.values[t]
            last = (delta + gamma * gae_lambda * continuation * last) * self.active_masks[t]
            advantages[t] = last
        return advantages, advantages + self.values[:n]

    def validated_policy_version(self):
        """Return the sole policy version among used rows with active samples."""
        used = self.policy_versions[:self.pos]
        valid_rows = self.active_masks[:self.pos].any(axis=1)
        versions = np.unique(used[valid_rows])
        if len(versions) == 0:
            raise ValueError("rollout contains no active samples")
        if len(versions) != 1:
            raise ValueError(
                f"rollout mixes policy versions: {versions.tolist()}")
        return int(versions[0])

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
    minibatch_permutations: list


class VanillaHAPPOTrainer:
    """Formal sequential HAPPO; shared actors are intentionally rejected."""

    algorithm_mode = "heterogeneous-reward vanilla HAPPO"

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=10.0,
                 ppo_epochs=2, minibatch_size=256, gamma=0.99,
                 gae_lambda=0.95, randomize_order=True, seed=0,
                 value_loss_type="clipped_huber", huber_delta=10.0):
        if policy.actor_sharing != "independent":
            raise ValueError("formal VanillaHAPPOTrainer requires independent actors")
        self.policy = policy
        self.actor_optimizers = {key: torch.optim.Adam(actor.parameters(), lr=actor_lr)
                                 for key, actor in policy.actors.items()}
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param, self.value_coef = clip_param, value_coef
        self.entropy_coef, self.max_grad_norm = entropy_coef, max_grad_norm
        self.ppo_epochs, self.minibatch_size = int(ppo_epochs), int(minibatch_size)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        if value_loss_type not in {"clipped_huber", "legacy_clipped_mse"}:
            raise ValueError("unsupported value_loss_type")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.value_loss_type = value_loss_type
        self.huber_delta = float(huber_delta)
        self.randomize_order = randomize_order
        self.rng = np.random.default_rng(seed)
        self.update_count = 0

    @staticmethod
    def joint_ratio(new_log_prob, old_log_prob):
        return torch.exp(new_log_prob - old_log_prob)

    @staticmethod
    def update_log_factor(log_factor, new_log_prob, old_log_prob, active_mask):
        increment = torch.where(active_mask > 0.5,
                                new_log_prob - old_log_prob,
                                torch.zeros_like(new_log_prob))
        result = (log_factor + increment).detach()
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite HAPPO log importance factor")
        return result

    @staticmethod
    def update_factor(factor, new_log_prob, old_log_prob, active_mask):
        result = factor * torch.where(
            active_mask > 0.5, torch.exp(new_log_prob - old_log_prob),
            torch.ones_like(new_log_prob))
        return result.detach()

    @staticmethod
    def _normalize_advantages(advantages, active):
        normalized = torch.zeros_like(advantages)
        for index in range(advantages.shape[1]):
            valid = active[:, index] > 0.5
            values = advantages[valid, index]
            if values.numel() > 1:
                normalized[valid, index] = (values - values.mean()) / (values.std() + 1e-8)
            elif values.numel() == 1:
                normalized[valid, index] = values
        return normalized

    def _permutation(self, indices, device, trace):
        order = self.rng.permutation(len(indices)).tolist()
        trace.append([int(indices[i]) for i in order])
        return indices[torch.as_tensor(order, device=device)]

    def update(self, buffer):
        policy_version = buffer.validated_policy_version()
        device = next(self.policy.parameters()).device
        data = buffer.tensors(device)
        advantages_np, returns_np = buffer.compute_gae(self.gamma, self.gae_lambda)
        raw_advantages = torch.as_tensor(advantages_np, device=device)
        returns = torch.as_tensor(returns_np, device=device)
        active = data["active_masks"]
        advantages = self._normalize_advantages(raw_advantages, active)
        order = list(range(len(self.policy.agent_ids)))
        if self.randomize_order:
            self.rng.shuffle(order)
        order_names = [self.policy.agent_ids[i] for i in order]
        permutations = []
        log_factor = torch.zeros(buffer.pos, device=device)
        actor_losses = {aid: [] for aid in self.policy.agent_ids}
        entropies = {aid: [] for aid in self.policy.agent_ids}
        grad_norms = {aid: [] for aid in self.policy.agent_ids}
        clip_fractions = {aid: [] for aid in self.policy.agent_ids}
        metrics = {"agent_order": ",".join(order_names),
                   "agent_order_count": 1,
                   "factor_initialization_count": 1,
                   "policy_version": policy_version}

        for index in order:
            aid = self.policy.agent_ids[index]
            valid = active[:, index] > 0.5
            fixed_factor = torch.exp(log_factor).detach()
            if not torch.isfinite(fixed_factor).all():
                raise FloatingPointError(f"non-finite HAPPO importance factor before {aid}")
            metrics[f"factor_before_mean/{aid}"] = float(fixed_factor.mean())
            metrics[f"factor_before_min/{aid}"] = float(fixed_factor.min())
            metrics[f"factor_before_max/{aid}"] = float(fixed_factor.max())
            if valid.any():
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                optimizer = self.actor_optimizers[aid]
                for _ in range(self.ppo_epochs):
                    permutation = self._permutation(valid_indices, device, permutations)
                    for start in range(0, len(permutation), self.minibatch_size):
                        batch = permutation[start:start + self.minibatch_size]
                        optimizer.zero_grad()
                        new_logp, entropy, _ = self.policy.evaluate_agent(
                            aid, data["obs"][batch, index], data["actions"][batch, index],
                            data["available_actions"][batch, index])
                        ratio = self.joint_ratio(new_logp, data["old_log_probs"][batch, index])
                        corrected = advantages[batch, index] * fixed_factor[batch]
                        clipped_ratio = ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
                        loss = -torch.minimum(ratio * corrected,
                                              clipped_ratio * corrected).mean()
                        loss -= self.entropy_coef * entropy.mean()
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(
                            self.policy.actors[aid].parameters(), self.max_grad_norm)
                        optimizer.step()
                        actor_losses[aid].append(float(loss.detach()))
                        entropies[aid].append(float(entropy.mean().detach()))
                        grad_norms[aid].append(float(grad.detach()))
                        clip_fractions[aid].append(float(
                            ((ratio - 1).abs() > self.clip_param).float().mean().detach()))
            with torch.no_grad():
                final_logp, _, _ = self.policy.evaluate_agent(
                    aid, data["obs"][:, index], data["actions"][:, index],
                    data["available_actions"][:, index])
            if valid.any():
                approx_kl = (data["old_log_probs"][valid, index]
                             - final_logp[valid]).mean()
                if not torch.isfinite(approx_kl):
                    raise FloatingPointError(f"non-finite approximate KL for {aid}")
                metrics[f"approx_kl/{aid}"] = float(approx_kl)
            else:
                metrics[f"approx_kl/{aid}"] = 0.0
            log_factor = self.update_log_factor(
                log_factor, final_logp, data["old_log_probs"][:, index], active[:, index])
            factor_after = torch.exp(log_factor)
            if not torch.isfinite(factor_after).all():
                raise FloatingPointError(f"non-finite HAPPO importance factor after {aid}")
            metrics[f"factor_after_mean/{aid}"] = float(factor_after.mean())
            metrics[f"factor_after_min/{aid}"] = float(factor_after.min())
            metrics[f"factor_after_max/{aid}"] = float(factor_after.max())

        critic_losses = {aid: [] for aid in self.policy.agent_ids}
        critic_head_grad_norms = {aid: [] for aid in self.policy.agent_ids}
        critic_grad_norms = []
        all_indices = torch.arange(buffer.pos, device=device)
        for _ in range(self.ppo_epochs):
            permutation = self._permutation(all_indices, device, permutations)
            for start in range(0, len(permutation), self.minibatch_size):
                batch = permutation[start:start + self.minibatch_size]
                predicted = self.policy.value(data["states"][batch])
                old_value = data["values"][batch]
                losses = []
                for index, aid in enumerate(self.policy.agent_ids):
                    valid = active[batch, index] > 0.5
                    if not valid.any():
                        continue
                    per_sample = clipped_value_loss(
                        predicted[:, index], old_value[:, index],
                        returns[batch, index], self.clip_param,
                        self.value_loss_type, self.huber_delta)
                    agent_loss = per_sample[valid].mean()
                    losses.append(agent_loss)
                    critic_losses[aid].append(float(agent_loss.detach()))
                if losses:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss = torch.stack(losses).mean()
                    (self.value_coef * loss).backward()
                    # Isolate inactive critic heads: set grad=None so Adam momentum
                    # does not drift parameters that have no active samples this minibatch.
                    active_head_ids = set()
                    for index, aid in enumerate(self.policy.agent_ids):
                        valid = active[batch, index] > 0.5
                        if valid.any():
                            active_head_ids.add(aid)
                    for aid in self.policy.agent_ids:
                        if aid not in active_head_ids:
                            for param in self.policy.critic.heads[aid].parameters():
                                param.grad = None
                    for aid in self.policy.agent_ids:
                        squared = [parameter.grad.detach().pow(2).sum()
                                   for parameter in self.policy.critic.heads[aid].parameters()
                                   if parameter.grad is not None]
                        if squared:
                            critic_head_grad_norms[aid].append(
                                float(torch.sqrt(torch.stack(squared).sum())))
                    grad = torch.nn.utils.clip_grad_norm_(
                        self.policy.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    critic_grad_norms.append(float(grad))

        with torch.no_grad():
            final_values = self.policy.value(data["states"])
        for index, aid in enumerate(self.policy.agent_ids):
            valid = active[:, index] > 0.5
            value = final_values[valid, index]
            target = returns[valid, index]
            adv = raw_advantages[valid, index]
            metrics[f"actor_loss/{aid}"] = float(np.mean(actor_losses[aid])) if actor_losses[aid] else 0.0
            metrics[f"entropy/{aid}"] = float(np.mean(entropies[aid])) if entropies[aid] else 0.0
            metrics[f"gradient_norm/{aid}"] = float(np.mean(grad_norms[aid])) if grad_norms[aid] else 0.0
            metrics[f"clip_fraction/{aid}"] = float(np.mean(clip_fractions[aid])) if clip_fractions[aid] else 0.0
            metrics[f"critic_loss/{aid}"] = float(np.mean(critic_losses[aid])) if critic_losses[aid] else 0.0
            metrics[f"active_sample_count/{aid}"] = int(valid.sum())
            metrics[f"actor_zero_gradient/{aid}"] = bool(
                valid.any() and grad_norms[aid] and max(grad_norms[aid]) == 0.0)
            metrics[f"critic_head_gradient_norm/{aid}"] = float(
                np.mean(critic_head_grad_norms[aid])) if critic_head_grad_norms[aid] else 0.0
            metrics[f"critic_head_zero_gradient/{aid}"] = bool(
                valid.any() and critic_head_grad_norms[aid]
                and max(critic_head_grad_norms[aid]) == 0.0)
            for label, tensor in (("value", value), ("return", target), ("advantage", adv)):
                metrics[f"{label}_mean/{aid}"] = float(tensor.mean()) if tensor.numel() else 0.0
                metrics[f"{label}_std/{aid}"] = float(tensor.std()) if tensor.numel() > 1 else 0.0
            if target.numel() > 1 and float(target.var()) > 0:
                metrics[f"explained_variance/{aid}"] = float(1 - (target-value).var()/target.var())
            else:
                metrics[f"explained_variance/{aid}"] = 0.0
        final_factor = torch.exp(log_factor)
        metrics.update({
            "critic_loss": float(np.mean([x for values in critic_losses.values() for x in values]))
                if any(critic_losses.values()) else 0.0,
            "critic_gradient_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
            "importance_factor_mean": float(final_factor.mean()),
            "importance_factor_std": float(final_factor.std()) if final_factor.numel() > 1 else 0.0,
            "importance_factor_min": float(final_factor.min()),
            "importance_factor_max": float(final_factor.max()),
            "log_factor_min": float(log_factor.min()),
            "log_factor_max": float(log_factor.max()),
            "runtime_invariants_valid": True,
        })
        numeric = [value for value in metrics.values() if isinstance(value, (int, float, bool))]
        if not np.isfinite(numeric).all():
            raise FloatingPointError("non-finite HAPPO update metric")
        self.update_count += 1
        return HAPPOUpdateResult(metrics, [order_names], permutations)


class ParameterSharingPPOTrainer:
    """Ordinary PPO ablation for shared actors; it has no sequential factor."""

    algorithm_mode = "parameter_sharing_ppo_ablation"

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4,
                 clip_param=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=10.0, ppo_epochs=2, minibatch_size=256,
                 gamma=0.99, gae_lambda=0.95, seed=0, **_):
        if policy.actor_sharing != "parameter_sharing_ppo_ablation":
            raise ValueError("parameter-sharing PPO requires shared actors")
        self.policy = policy
        self.actor_optimizers = {key: torch.optim.Adam(actor.parameters(), lr=actor_lr)
                                 for key, actor in policy.actors.items()}
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param, self.value_coef = clip_param, value_coef
        self.entropy_coef, self.max_grad_norm = entropy_coef, max_grad_norm
        self.ppo_epochs, self.minibatch_size = int(ppo_epochs), int(minibatch_size)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.rng = np.random.default_rng(seed)
        self.update_count = 0
        self.uses_sequential_factor = False

    def update(self, buffer):
        device = next(self.policy.parameters()).device
        data = buffer.tensors(device)
        advantages_np, returns_np = buffer.compute_gae(self.gamma, self.gae_lambda)
        advantages = VanillaHAPPOTrainer._normalize_advantages(
            torch.as_tensor(advantages_np, device=device), data["active_masks"])
        returns = torch.as_tensor(returns_np, device=device)
        permutations, losses = [], []
        for key, optimizer in self.actor_optimizers.items():
            samples = [(t, i) for i, aid in enumerate(self.policy.agent_ids)
                       if self.policy.actor_key(aid) == key
                       for t in range(buffer.pos) if data["active_masks"][t, i] > 0.5]
            for _ in range(self.ppo_epochs):
                order = self.rng.permutation(len(samples)).tolist()
                permutations.append(order)
                for start in range(0, len(order), self.minibatch_size):
                    selected = [samples[j] for j in order[start:start+self.minibatch_size]]
                    if not selected:
                        continue
                    terms, entropy_terms = [], []
                    for t, i in selected:
                        aid = self.policy.agent_ids[i]
                        new_logp, entropy, _ = self.policy.evaluate_agent(
                            aid, data["obs"][t:t+1, i], data["actions"][t:t+1, i],
                            data["available_actions"][t:t+1, i])
                        ratio = torch.exp(new_logp - data["old_log_probs"][t, i])
                        adv = advantages[t, i]
                        terms.append(torch.minimum(
                            ratio * adv,
                            ratio.clamp(1-self.clip_param, 1+self.clip_param) * adv))
                        entropy_terms.append(entropy)
                    loss = -torch.cat(terms).mean() - self.entropy_coef * torch.cat(entropy_terms).mean()
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.actors[key].parameters(),
                                                   self.max_grad_norm)
                    optimizer.step(); losses.append(float(loss.detach()))
        critic_losses = []
        for _ in range(self.ppo_epochs):
            predicted = self.policy.value(data["states"])
            valid = data["active_masks"] > 0.5
            old = data["values"]
            clipped = old + (predicted-old).clamp(-self.clip_param, self.clip_param)
            per_sample = 0.5 * torch.maximum((predicted-returns).pow(2),
                                              (clipped-returns).pow(2))
            critic_loss = per_sample[valid].mean()
            self.critic_optimizer.zero_grad(); (self.value_coef*critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step(); critic_losses.append(float(critic_loss.detach()))
        self.update_count += 1
        metrics = {"actor_loss": float(np.mean(losses)) if losses else 0.0,
                   "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                   "uses_sequential_factor": False,
                   "runtime_invariants_valid": False,
                   "agent_order_count": 0,
                   "factor_initialization_count": 0}
        return HAPPOUpdateResult(metrics, [], permutations)
