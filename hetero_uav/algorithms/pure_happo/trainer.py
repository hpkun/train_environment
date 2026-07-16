"""Vanilla sequential HAPPO for independent actors and one shared critic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


ALGORITHM_CONTRACT = "pure_happo_sequential_v2"


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _param_vector(params) -> torch.Tensor:
    chunks = [parameter.detach().flatten().cpu() for parameter in params]
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {key: 0.0 for key in ("mean", "std", "min", "max")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _epoch_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "final": 0.0}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "final": float(values[-1]),
    }


def _ratio_stats(ratio: torch.Tensor, valid: torch.Tensor, clip_param: float,
                 old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> dict[str, float]:
    values = ratio[valid].detach().float()
    if values.numel() == 0:
        return {key: 0.0 for key in (
            "mean", "std", "min", "max", "p95", "p99", "clip_fraction",
            "approx_kl", "approx_kl_abs")}
    delta = old_log_prob[valid] - new_log_prob[valid].detach()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "clip_fraction": float(((values < 1.0 - clip_param) |
                                (values > 1.0 + clip_param)).float().mean()),
        "approx_kl": float(delta.mean()),
        "approx_kl_abs": float(delta.abs().mean()),
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    if values.shape != returns.shape:
        raise ValueError(f"value/return shape mismatch: {values.shape} != {returns.shape}")
    variance = torch.var(returns.detach().float(), unbiased=False)
    if float(variance) <= 1e-12:
        return 0.0
    return float(1.0 - torch.var(returns - values, unbiased=False) / variance)


def _alive_before_team_mean(rewards: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return (rewards * active).sum(dim=-1) / active.sum(dim=-1).clamp(min=1)


def _compute_grouped_gae(rewards, values, next_values, bootstrap_or_dones,
                         continuation_or_env_ids, env_ids_or_gamma=None,
                         gamma_or_lambda=None, gae_lambda=None, **legacy_kwargs):
    """GAE with separate bootstrap and episode-continuation masks.

    The seven-argument legacy form remains readable for old diagnostic callers;
    it interprets ``bootstrap_or_dones`` as done and derives both masks from it.
    Formal sequential-v2 callers use the eight-argument mask form.
    """
    if legacy_kwargs:
        unexpected = set(legacy_kwargs) - {"gamma", "lam"}
        if unexpected:
            raise TypeError(f"unexpected GAE keyword(s): {sorted(unexpected)}")
        if env_ids_or_gamma is not None or gamma_or_lambda is not None or gae_lambda is not None:
            raise TypeError("do not mix positional and keyword legacy GAE parameters")
        env_ids_or_gamma = legacy_kwargs.get("gamma")
        gamma_or_lambda = legacy_kwargs.get("lam")
    if gae_lambda is None:
        if env_ids_or_gamma is None or gamma_or_lambda is None:
            raise TypeError("legacy GAE requires gamma and lam")
        dones = bootstrap_or_dones
        bootstrap_masks = 1.0 - dones
        continuation_masks = 1.0 - dones
        env_ids = continuation_or_env_ids
        gamma = env_ids_or_gamma
        gae_lambda = gamma_or_lambda
    else:
        bootstrap_masks = bootstrap_or_dones
        continuation_masks = continuation_or_env_ids
        env_ids = env_ids_or_gamma
        gamma = gamma_or_lambda
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for env_id in torch.unique(env_ids):
        indices = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
        gae = rewards.new_tensor(0.0)
        for position in reversed(indices.tolist()):
            delta = (rewards[position]
                     + gamma * bootstrap_masks[position] * next_values[position]
                     - values[position])
            gae = (delta + gamma * gae_lambda * continuation_masks[position] * gae)
            advantages[position] = gae
            returns[position] = gae + values[position]
    return advantages, returns


class PureHAPPOTrainer:
    """One fixed random order per update; all epochs are contiguous per actor."""

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4,
                 clip_param=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=5, gamma=0.99,
                 gae_lambda=0.95, seed=None, critic_epochs: int = 1):
        if getattr(policy, "credit_mode", None) != "shared_alive_team_mean":
            raise ValueError("sequential-v2 requires shared_alive_team_mean")
        if int(critic_epochs) < 1:
            raise ValueError("critic_epochs must be >= 1")
        self.policy = policy
        self.clip_param = float(clip_param)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.critic_epochs = int(critic_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.rng = np.random.default_rng(seed)
        self.actor_opts = [optim.Adam(
            list(policy.actors[index].parameters()) + [policy.action_log_stds[index]],
            lr=actor_lr) for index in range(policy.num_agents)]
        self.critic_opt = optim.Adam(policy.critic.parameters(), lr=critic_lr)

    def update(self, buffer) -> dict:
        device = next(self.policy.parameters()).device
        data = buffer.get(device)
        actor_obs = data["actor_obs"]
        critic_state = data["critic_state"]
        actions = data["actions"]
        raw_actions = data["raw_actions"]
        raw_valid = data["raw_action_valid"] > 0.5
        old_log_probs = data["old_log_probs"]
        rewards = data["rewards"]
        values = data["values"]
        next_values = data["next_values"]
        active = data["active_masks"]
        actor_active = data.get("actor_update_masks", active)
        terminated = data["terminated"]
        episode_dones = data["episode_dones"]
        env_ids = data["env_ids"]
        tensors = {
            "actor_obs": actor_obs, "critic_state": critic_state, "actions": actions,
            "old_log_probs": old_log_probs, "rewards": rewards, "values": values,
            "next_values": next_values, "active": active,
            "terminated": terminated, "episode_dones": episode_dones,
        }
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"HAPPO: non-finite {name}")
        if values.ndim != 1 or next_values.shape != values.shape:
            raise ValueError("sequential-v2 critic values must be scalar [T]")

        team_reward = _alive_before_team_mean(rewards, active)
        bootstrap_masks = 1.0 - terminated
        continuation_masks = 1.0 - episode_dones
        advantages_raw, returns = _compute_grouped_gae(
            team_reward, values, next_values, bootstrap_masks, continuation_masks,
            env_ids, self.gamma, self.gae_lambda)
        advantages = advantages_raw.clone()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)
        advantages = advantages.detach()
        returns = returns.detach()

        count = self.policy.num_agents
        order = list(self.rng.permutation(count))
        factor = torch.ones_like(advantages)
        per_agent = [{
            "loss": [], "entropy": [], "kl": [], "kl_abs": [], "clip": [],
            "ratio_mean": [], "ratio_std": [], "ratio_p95": [], "ratio_p99": [],
            "grad_norm": [],
        } for _ in range(count)]
        factor_before = [{} for _ in range(count)]
        factor_after = [{} for _ in range(count)]
        final_stats = [{} for _ in range(count)]
        update_norm = [0.0] * count
        update_trace = []

        for agent_idx in order:
            valid = actor_active[:, agent_idx] > 0.5
            fixed_factor = factor.detach().clone()
            factor_before[agent_idx] = _stats(fixed_factor)
            params = (list(self.policy.actors[agent_idx].parameters())
                      + [self.policy.action_log_stds[agent_idx]])
            before = _param_vector(params)
            raw_i = raw_actions[:, agent_idx] if bool(raw_valid.all()) else None
            for epoch in range(self.ppo_epochs):
                update_trace.append([int(agent_idx), int(epoch)])
                if not bool(valid.any()):
                    continue
                self.actor_opts[agent_idx].zero_grad()
                new_log_prob, entropy, _ = self.policy.evaluate_agent_actions(
                    agent_idx, actor_obs[:, agent_idx], actions[:, agent_idx], raw_i)
                ratio = torch.exp(new_log_prob - old_log_probs[:, agent_idx].detach())
                if not torch.isfinite(ratio).all():
                    raise ValueError(f"HAPPO: non-finite ratio agent={agent_idx}")
                effective_advantage = (advantages * fixed_factor).detach()
                surrogate = ratio * effective_advantage
                clipped = torch.clamp(ratio, 1.0 - self.clip_param,
                                      1.0 + self.clip_param) * effective_advantage
                valid_float = valid.float()
                policy_loss = -(torch.minimum(surrogate, clipped) * valid_float).sum() / valid_float.sum()
                entropy_mean = (entropy * valid_float).sum() / valid_float.sum()
                loss = policy_loss - self.entropy_coef * entropy_mean
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise ValueError(f"HAPPO: non-finite actor gradient agent={agent_idx}")
                self.actor_opts[agent_idx].step()
                stats = _ratio_stats(ratio, valid, self.clip_param,
                                     old_log_probs[:, agent_idx], new_log_prob)
                target = per_agent[agent_idx]
                target["loss"].append(float(policy_loss.detach()))
                target["entropy"].append(float(entropy_mean.detach()))
                target["kl"].append(stats["approx_kl"])
                target["kl_abs"].append(stats["approx_kl_abs"])
                target["clip"].append(stats["clip_fraction"])
                for key in ("ratio_mean", "ratio_std", "ratio_p95", "ratio_p99"):
                    short_key = key[6:] if key.startswith("ratio_") else key
                    target[key].append(stats[short_key])
                target["grad_norm"].append(_safe_float(grad_norm))
            update_norm[agent_idx] = float(torch.linalg.vector_norm(
                _param_vector(params) - before))
            with torch.no_grad():
                final_log_prob, _, _ = self.policy.evaluate_agent_actions(
                    agent_idx, actor_obs[:, agent_idx], actions[:, agent_idx], raw_i)
                final_ratio = torch.exp(final_log_prob - old_log_probs[:, agent_idx])
                final_ratio = torch.where(valid, final_ratio, torch.ones_like(final_ratio))
                if not torch.isfinite(final_ratio).all():
                    raise ValueError(f"HAPPO: non-finite final ratio agent={agent_idx}")
                final_stats[agent_idx] = _ratio_stats(
                    final_ratio, valid, self.clip_param,
                    old_log_probs[:, agent_idx], final_log_prob)
                factor = (factor * final_ratio).detach()
                factor_after[agent_idx] = _stats(factor)

        critic_params = list(self.policy.critic.parameters())
        critic_before = _param_vector(critic_params)
        critic_losses = []
        critic_grad_norms = []
        with torch.no_grad():
            old_ev = _explained_variance(values, returns)
        for _ in range(self.critic_epochs):
            self.critic_opt.zero_grad()
            prediction = self.policy.value(critic_state)
            if prediction.shape != returns.shape:
                raise ValueError("critic prediction/return shape mismatch")
            critic_loss_unscaled = F.mse_loss(prediction, returns)
            critic_loss = self.value_coef * critic_loss_unscaled
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(critic_grad)):
                raise ValueError("HAPPO: non-finite critic gradient")
            self.critic_opt.step()
            critic_losses.append(float(critic_loss_unscaled.detach()))
            critic_grad_norms.append(_safe_float(critic_grad))
        with torch.no_grad():
            final_values = self.policy.value(critic_state)
            new_ev = _explained_variance(final_values, returns)
        critic_update_norm = float(torch.linalg.vector_norm(
            _param_vector(critic_params) - critic_before))

        valid_counts = [int((actor_active[:, i] > 0.5).sum()) for i in range(count)]
        average = lambda values: float(np.mean(values)) if values else 0.0
        maximum = lambda values: float(np.max(values)) if values else 0.0
        actor_loss_per_agent = [average(row["loss"]) for row in per_agent]
        entropy_per_agent = [average(row["entropy"]) for row in per_agent]
        kl_per_agent = [average(row["kl"]) for row in per_agent]
        kl_abs_per_agent = [average(row["kl_abs"]) for row in per_agent]
        clip_per_agent = [average(row["clip"]) for row in per_agent]
        ratio_p95 = [average(row["ratio_p95"]) for row in per_agent]
        ratio_p99 = [average(row["ratio_p99"]) for row in per_agent]
        ratio_mean = [average(row["ratio_mean"]) for row in per_agent]
        ratio_std = [average(row["ratio_std"]) for row in per_agent]
        grad_norms = [average(row["grad_norm"]) for row in per_agent]
        uav = [1, 2] if count >= 3 else list(range(1, count))
        role_mean = lambda values, indices: average([values[i] for i in indices])
        action_stats = self._action_stats(actions, active)
        log_stds = torch.stack([parameter.detach() for parameter in self.policy.action_log_stds])
        raw_adv_stats = _stats(advantages_raw)
        normalized_adv_stats = _stats(advantages)
        return_stats = _stats(returns)
        metrics = {
            "algorithm_contract": ALGORITHM_CONTRACT,
            "last_update_order": order,
            "agent_update_order": order,
            "actor_update_trace": update_trace,
            "actor_loss_epochs_per_agent": [row["loss"] for row in per_agent],
            "entropy_epochs_per_agent": [row["entropy"] for row in per_agent],
            "approx_kl_epochs_per_agent": [row["kl"] for row in per_agent],
            "approx_kl_abs_epochs_per_agent": [row["kl_abs"] for row in per_agent],
            "clip_fraction_epochs_per_agent": [row["clip"] for row in per_agent],
            "actor_loss_epoch_summary_per_agent": [
                _epoch_summary(row["loss"]) for row in per_agent],
            "entropy_epoch_summary_per_agent": [
                _epoch_summary(row["entropy"]) for row in per_agent],
            "approx_kl_epoch_summary_per_agent": [
                _epoch_summary(row["kl"]) for row in per_agent],
            "approx_kl_abs_epoch_summary_per_agent": [
                _epoch_summary(row["kl_abs"]) for row in per_agent],
            "clip_fraction_epoch_summary_per_agent": [
                _epoch_summary(row["clip"]) for row in per_agent],
            "factor_before_per_agent": factor_before,
            "factor_after_per_agent": factor_after,
            "final_factor": _stats(factor),
            "m_abs_mean_after_each_agent": [
                factor_after[index].get("mean", 0.0) for index in order],
            "actor_loss_per_agent": actor_loss_per_agent,
            "entropy_per_agent": entropy_per_agent,
            "approx_kl_per_agent": kl_per_agent,
            "approx_kl_abs_per_agent": kl_abs_per_agent,
            "clip_fraction_per_agent": clip_per_agent,
            "ratio_mean_per_agent": ratio_mean,
            "ratio_std_per_agent": ratio_std,
            "ratio_p95_per_agent": ratio_p95,
            "ratio_p99_per_agent": ratio_p99,
            "actor_grad_norm_per_agent": grad_norms,
            "policy_update_norm_per_agent": update_norm,
            "final_approx_kl_per_agent": [row.get("approx_kl", 0.0) for row in final_stats],
            "final_approx_kl_abs_per_agent": [row.get("approx_kl_abs", 0.0) for row in final_stats],
            "final_ratio_mean_per_agent": [row.get("mean", 0.0) for row in final_stats],
            "final_ratio_std_per_agent": [row.get("std", 0.0) for row in final_stats],
            "final_ratio_p95_per_agent": [row.get("p95", 0.0) for row in final_stats],
            "final_ratio_p99_per_agent": [row.get("p99", 0.0) for row in final_stats],
            "ratio_after_mean_per_agent": [row.get("mean", 0.0) for row in final_stats],
            "valid_sample_count_per_agent": valid_counts,
            "actor_loss_mean": role_mean(actor_loss_per_agent, list(range(count))),
            "entropy_mean": role_mean(entropy_per_agent, list(range(count))),
            "approx_kl_mean": role_mean(kl_per_agent, list(range(count))),
            "approx_kl_mav": kl_per_agent[0],
            "approx_kl_uav": role_mean(kl_per_agent, uav),
            "approx_kl_abs_mav": kl_abs_per_agent[0],
            "approx_kl_abs_uav": role_mean(kl_abs_per_agent, uav),
            "final_approx_kl_abs_mav": final_stats[0].get("approx_kl_abs", 0.0),
            "final_approx_kl_abs_uav": role_mean(
                [row.get("approx_kl_abs", 0.0) for row in final_stats], uav),
            "final_ratio_p95_mav": final_stats[0].get("p95", 0.0),
            "final_ratio_p95_uav": role_mean(
                [row.get("p95", 0.0) for row in final_stats], uav),
            "final_ratio_p99_mav": final_stats[0].get("p99", 0.0),
            "final_ratio_p99_uav": role_mean(
                [row.get("p99", 0.0) for row in final_stats], uav),
            "clip_fraction_mav": clip_per_agent[0],
            "clip_fraction_uav": role_mean(clip_per_agent, uav),
            "ratio_p95_mav": ratio_p95[0],
            "ratio_p95_uav": role_mean(ratio_p95, uav),
            "ratio_p99_mav": ratio_p99[0],
            "ratio_p99_uav": role_mean(ratio_p99, uav),
            "policy_update_norm_mav": update_norm[0],
            "policy_update_norm_uav": role_mean(update_norm, uav),
            "actor_grad_norm_mav": grad_norms[0],
            "actor_grad_norm_uav": role_mean(grad_norms, uav),
            "critic_loss": critic_losses[-1] if critic_losses else 0.0,
            "critic_loss_unscaled": critic_losses[-1] if critic_losses else 0.0,
            "critic_loss_scaled": (
                self.value_coef * critic_losses[-1] if critic_losses else 0.0),
            "critic_epochs": self.critic_epochs,
            "critic_loss_per_epoch": critic_losses,
            "critic_grad_norm_per_epoch": critic_grad_norms,
            "critic_loss_mean_over_epochs": average(critic_losses),
            "critic_loss_first_epoch": critic_losses[0] if critic_losses else 0.0,
            "critic_loss_last_epoch": critic_losses[-1] if critic_losses else 0.0,
            "critic_grad_norm_mean_over_epochs": average(critic_grad_norms),
            "critic_grad_norm_max_over_epochs": maximum(critic_grad_norms),
            "critic_grad_norm": critic_grad_norms[-1] if critic_grad_norms else 0.0,
            "critic_update_norm": critic_update_norm,
            "value_explained_variance_old": old_ev,
            "value_explained_variance_new": new_ev,
            "value_explained_variance": new_ev,
            "value_pred_old_mean": float(values.mean()),
            "value_pred_new_mean": float(final_values.mean()),
            "advantage_raw_mean": raw_adv_stats["mean"],
            "advantage_raw_std": raw_adv_stats["std"],
            "advantage_raw_min": raw_adv_stats["min"],
            "advantage_raw_max": raw_adv_stats["max"],
            "advantage_norm_mean": normalized_adv_stats["mean"],
            "advantage_norm_std": normalized_adv_stats["std"],
            "return_mean": return_stats["mean"],
            "return_std": return_stats["std"],
            "return_min": return_stats["min"],
            "return_max": return_stats["max"],
            "terminated_count": int(terminated.sum()),
            "truncation_count": int(((episode_dones > 0.5) & (terminated < 0.5)).sum()),
            "episode_boundary_count": int(episode_dones.sum()),
            "action_log_std_mean": float(log_stds.mean()),
            "action_log_std_mav_mean": float(log_stds[0].mean()),
            "action_log_std_uav_mean": float(log_stds[1:].mean()) if count > 1 else 0.0,
            "credit_mode": self.policy.credit_mode,
            "active_sample_ratio_per_agent": [
                value / max(int(actor_active.shape[0]), 1) for value in valid_counts],
            "m_mean_after_each_agent": [
                factor_after[index].get("mean", 0.0) for index in order],
            "m_std_after_each_agent": [
                factor_after[index].get("std", 0.0) for index in order],
            "m_abs_max_after_each_agent": [
                max(abs(factor_after[index].get("min", 0.0)),
                    abs(factor_after[index].get("max", 0.0))) for index in order],
            **action_stats,
        }
        for key, value in action_stats.items():
            if key.endswith("_active"):
                metrics[key[:-7]] = value
        for prefix, values in (("actor_loss", actor_loss_per_agent),
                               ("entropy", entropy_per_agent)):
            metrics[f"{prefix}_mav"] = values[0]
            metrics[f"{prefix}_uav"] = role_mean(values, uav)
        metrics["mav_action_saturation_rate"] = action_stats["mav_action_saturation_rate"]
        metrics["uav_action_saturation_rate"] = action_stats["uav_action_saturation_rate"]
        for row in per_agent:
            row["loss_max_abs"] = maximum([abs(value) for value in row["loss"]])
        if not self._all_finite(metrics):
            raise ValueError("HAPPO: non-finite training metrics")
        return metrics

    @staticmethod
    def _action_stats(actions: torch.Tensor, active: torch.Tensor) -> dict[str, float]:
        names = ("pitch", "heading", "speed")
        result = {}
        for role, indices in (("mav", [0]), ("uav", list(range(1, actions.shape[1])))):
            values = actions[:, indices].reshape(-1, actions.shape[-1])
            mask = active[:, indices].reshape(-1) > 0.5
            for dimension, name in enumerate(names[:actions.shape[-1]]):
                valid = values[mask, dimension]
                result[f"{role}_action_mean_{name}_active"] = float(valid.mean()) if valid.numel() else 0.0
                result[f"{role}_action_std_{name}_active"] = (
                    float(valid.std(unbiased=False)) if valid.numel() > 1 else 0.0)
                result[f"{role}_action_saturation_{name}_active"] = (
                    float((valid.abs() >= 0.999).float().mean()) if valid.numel() else 0.0)
        result["mav_action_saturation_rate"] = float(
            (actions[:, 0].abs() >= 0.999).float().mean())
        result["uav_action_saturation_rate"] = float(
            (actions[:, 1:].abs() >= 0.999).float().mean()) if actions.shape[1] > 1 else 0.0
        return result

    @classmethod
    def _all_finite(cls, value) -> bool:
        if isinstance(value, dict):
            return all(cls._all_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(cls._all_finite(item) for item in value)
        if isinstance(value, (int, float, np.number)):
            return bool(np.isfinite(value))
        return True
