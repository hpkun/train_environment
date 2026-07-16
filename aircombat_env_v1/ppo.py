"""Small feed-forward PPO implementation for the 1v1 environment."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = math.log(0.05)
LOG_STD_MAX = math.log(0.6)


def _orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class SquashedNormal:
    def __init__(self, mean, std):
        self.base = Normal(mean, std)

    @property
    def mean(self):
        return torch.tanh(self.base.mean)

    def sample(self):
        return torch.tanh(self.base.sample())

    def rsample(self):
        return torch.tanh(self.base.rsample())

    def log_prob(self, action):
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(action)
        correction = 2.0 * (
            math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
        return (self.base.log_prob(latent) - correction).sum(-1)

    def entropy(self):
        return self.base.entropy().sum(-1)


class Actor(nn.Module):
    def __init__(self, observation_dim=20, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            _orthogonal(nn.Linear(observation_dim, 128), math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal(nn.Linear(128, 128), math.sqrt(2.0)),
            nn.Tanh(),
        )
        self.action_mean = _orthogonal(nn.Linear(128, action_dim), 0.01)
        self.action_log_std = _orthogonal(nn.Linear(128, action_dim), 0.01)
        initial_fraction = (
            math.log(0.3) - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN)
        initial_raw = math.log(initial_fraction / (1.0 - initial_fraction))
        nn.init.constant_(self.action_log_std.bias, initial_raw)

    def distribution(self, observation):
        hidden = self.network(observation)
        mean = self.action_mean(hidden)
        raw_log_std = self.action_log_std(hidden)
        fraction = torch.sigmoid(raw_log_std)
        log_std = LOG_STD_MIN + fraction * (LOG_STD_MAX - LOG_STD_MIN)
        return SquashedNormal(mean, log_std.exp())

    def forward(self, observation):
        distribution = self.distribution(observation)
        return distribution.mean, distribution.base.stddev


class Critic(nn.Module):
    def __init__(self, observation_dim=20):
        super().__init__()
        self.network = nn.Sequential(
            _orthogonal(nn.Linear(observation_dim, 128), math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal(nn.Linear(128, 128), math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal(nn.Linear(128, 1), 1.0),
        )

    def forward(self, observation):
        return self.network(observation).squeeze(-1)


def deterministic_action(actor, observation):
    with torch.no_grad():
        return actor.distribution(observation).mean


def stochastic_action(actor, observation):
    with torch.no_grad():
        distribution = actor.distribution(observation)
        action = distribution.sample()
        return action, distribution.log_prob(action)


def compute_gae(rewards, values, next_values, terminated, truncated,
                gamma=0.99, gae_lambda=0.95):
    """Compute GAE while bootstrapping truncations but not true terminals."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1:], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        bootstrap = 1.0 - terminated[step].astype(np.float32)
        continuation = 1.0 - np.logical_or(
            terminated[step], truncated[step]).astype(np.float32)
        delta = (
            rewards[step] + gamma * next_values[step] * bootstrap
            - values[step])
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[step] = gae
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, rollout_steps, num_envs, observation_dim=20,
                 action_dim=3):
        shape = (rollout_steps, num_envs)
        self.observations = np.zeros(
            shape + (observation_dim,), dtype=np.float32)
        self.actions = np.zeros(shape + (action_dim,), dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.next_values = np.zeros(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        self.position = 0

    def add(self, observations, actions, rewards, values, next_values,
            log_probs, terminated, truncated):
        index = self.position
        if index >= self.rewards.shape[0]:
            raise IndexError("rollout buffer is full")
        for name, value in (
            ("observations", observations), ("actions", actions),
            ("rewards", rewards), ("values", values),
            ("next_values", next_values), ("log_probs", log_probs),
            ("terminated", terminated), ("truncated", truncated),
        ):
            getattr(self, name)[index] = value
        self.position += 1

    def flatten(self, gamma=0.99, gae_lambda=0.95):
        if self.position != self.rewards.shape[0]:
            raise RuntimeError("rollout buffer is not full")
        advantages, returns = compute_gae(
            self.rewards, self.values, self.next_values,
            self.terminated, self.truncated, gamma, gae_lambda)
        result = {
            "observations": self.observations.reshape(-1, self.observations.shape[-1]),
            "actions": self.actions.reshape(-1, self.actions.shape[-1]),
            "old_log_probs": self.log_probs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        advantage = result["advantages"]
        result["advantages"] = (
            (advantage - advantage.mean()) / (advantage.std() + 1e-8))
        return result


def ppo_update(actor, critic, optimizer, batch, device,
               clip_epsilon=0.2, update_epochs=10, minibatch_size=256,
               entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
               target_kl=0.03):
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise FloatingPointError("non-finite rollout data")
    count = tensors["returns"].shape[0]
    metrics = []
    for _ in range(update_epochs):
        permutation = torch.randperm(count, device=device)
        epoch_kl = []
        for start in range(0, count, minibatch_size):
            indices = permutation[start:start + minibatch_size]
            observations = tensors["observations"][indices]
            actions = tensors["actions"][indices]
            old_log_prob = tensors["old_log_probs"][indices]
            advantages = tensors["advantages"][indices]
            returns = tensors["returns"][indices]
            distribution = actor.distribution(observations)
            new_log_prob = distribution.log_prob(actions)
            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            policy_loss = -torch.min(
                ratio * advantages,
                ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
                * advantages).mean()
            value_loss = 0.5 * (critic(observations) - returns).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                max_grad_norm)
            optimizer.step()
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (ratio - 1.0).abs() > clip_epsilon).float().mean()
            metrics.append({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "approximate_kl": approximate_kl.item(),
                "clip_fraction": clip_fraction.item(),
                "gradient_norm": float(gradient_norm),
            })
            epoch_kl.append(approximate_kl.item())
        if np.mean(epoch_kl) > target_kl:
            break
    if not all(torch.isfinite(parameter).all()
               for model in (actor, critic) for parameter in model.parameters()):
        raise FloatingPointError("non-finite model parameter")
    predictions = critic(tensors["observations"]).detach().cpu().numpy()
    returns = tensors["returns"].cpu().numpy()
    variance = np.var(returns)
    explained_variance = (
        float("nan") if variance == 0.0
        else float(1.0 - np.var(returns - predictions) / variance))
    output = {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }
    output["explained_variance"] = explained_variance
    with torch.no_grad():
        _, std = actor(tensors["observations"])
        output["action_std_mean"] = float(std.mean())
    return output
