"""Paper-aligned HAPPO baseline: independent per-agent actors + shared V critic.

ICLR 2022 HAPPO algorithm: random sequential agent update with
PPO-clip and correction factor M. Grouped GAE by env_id.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


def _safe_float(value) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _param_vector(params) -> torch.Tensor:
    chunks = [p.detach().flatten().cpu() for p in params]
    return torch.cat(chunks) if chunks else torch.zeros(1)


def _stats_from_tensor(x: torch.Tensor) -> dict:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _valid_ratio_stats(ratio: torch.Tensor, valid: torch.Tensor,
                       clip_param: float, old_lp: torch.Tensor,
                       new_lp: torch.Tensor) -> dict:
    vals = ratio[valid].detach().float()
    if vals.numel() == 0:
        return {
            "ratio_mean": 0.0, "ratio_std": 0.0, "ratio_min": 0.0,
            "ratio_max": 0.0, "ratio_p95": 0.0, "ratio_p99": 0.0,
            "clip_fraction": 0.0, "approx_kl_abs": 0.0,
        }
    clip = ((vals < 1.0 - clip_param) | (vals > 1.0 + clip_param)).float().mean()
    kl_abs = (old_lp[valid] - new_lp[valid].detach()).abs().float().mean()
    return {
        "ratio_mean": float(vals.mean().item()),
        "ratio_std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "ratio_min": float(vals.min().item()),
        "ratio_max": float(vals.max().item()),
        "ratio_p95": float(torch.quantile(vals, 0.95).item()),
        "ratio_p99": float(torch.quantile(vals, 0.99).item()),
        "clip_fraction": float(clip.item()),
        "approx_kl_abs": float(kl_abs.item()),
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    y = returns.detach().float()
    pred = values.detach().float()
    if y.numel() == 0:
        return 0.0
    var_y = torch.var(y, unbiased=False)
    if float(var_y.item()) <= 1e-12:
        return 0.0
    err = y - pred
    return float((1.0 - torch.var(err, unbiased=False) / var_y).item())


def _mean_for_indices(values: list[float], indices: list[int]) -> float:
    vals = [_safe_float(values[i]) for i in indices if i < len(values)]
    return float(np.mean(vals)) if vals else 0.0


def _alive_before_team_mean(rewards: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Per-step team reward using the rollout's pre-action alive mask."""
    return (rewards * active).sum(dim=-1) / active.sum(dim=-1).clamp(min=1)


def _compute_grouped_gae(rewards, values, next_values, dones, env_ids, gamma, lam):
    """GAE grouped by env_id.  next_values[pos] is the bootstrap value
    for transition pos, already stored by the rollout buffer."""
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    for eid in torch.unique(env_ids):
        idx = torch.nonzero(env_ids == eid, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        gae = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        for pos in reversed(idx.tolist()):
            nv = next_values[pos]
            nonterminal = 1.0 - dones[pos]
            delta = rewards[pos] + gamma * nv * nonterminal - values[pos]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[pos] = gae
            returns[pos] = gae + values[pos]
    return advantages, returns


def _compute_role_local_gae(rewards, values, next_values, dones, env_ids, gamma, lam):
    """Compute independent per-agent GAE chains with per-agent done masks."""
    if rewards.shape != values.shape or values.shape != next_values.shape:
        raise ValueError(
            "role-local rewards, values, and next_values must have identical [T,N] shapes")
    if dones.shape != rewards.shape:
        raise ValueError("role-local dones must have shape [T,N]")
    advantage_cols = []
    return_cols = []
    for i in range(rewards.shape[1]):
        adv_i, ret_i = _compute_grouped_gae(
            rewards[:, i], values[:, i], next_values[:, i], dones[:, i],
            env_ids, gamma, lam)
        advantage_cols.append(adv_i)
        return_cols.append(ret_i)
    return torch.stack(advantage_cols, dim=1), torch.stack(return_cols, dim=1)


def _normalize_role_local_advantages(
        advantages: torch.Tensor, actor_update_masks: torch.Tensor) -> torch.Tensor:
    """Normalize each agent using only samples eligible for its actor update."""
    normalized = torch.zeros_like(advantages)
    for i in range(advantages.shape[1]):
        valid = actor_update_masks[:, i] > 0.5
        values = advantages[valid, i]
        if values.numel() == 0:
            continue
        mean = values.mean()
        std = values.std(unbiased=False) if values.numel() > 1 else values.new_tensor(0.0)
        normalized[valid, i] = (values - mean) / (std + 1e-8)
    return normalized


class PureHAPPOTrainer:
    """Paper-aligned HAPPO trainer."""

    def __init__(self, policy, actor_lr=5e-4, critic_lr=5e-4,
                 clip_param=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=10.0, ppo_epochs=5,
                 gamma=0.99, gae_lambda=0.95,
                 seed=None, critic_epochs: int = 1):
        if int(critic_epochs) < 1:
            raise ValueError("critic_epochs must be >= 1")
        self.policy = policy
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.critic_epochs = int(critic_epochs)
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
        actor_obs = data["actor_obs"]; critic_state = data["critic_state"]
        actions = data["actions"]; old_log_probs = data["old_log_probs"]
        rewards = data["rewards"]; dones = data["dones"]
        values = data["values"]; active = data["active_masks"]
        actor_active = data.get("actor_update_masks", active)
        local_credit = getattr(self.policy, "credit_mode", "shared_alive_team_mean") == "role_local_vector_critic"

        T, N = actor_obs.shape[:2]
        for name, tensor in [("actor_obs", actor_obs), ("critic_state", critic_state),
                              ("actions", actions), ("old_log_probs", old_log_probs),
                              ("rewards", rewards), ("dones", dones),
                              ("values", values), ("active_masks", active),
                              ("actor_update_masks", actor_active)]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"HAPPO: non-finite {name} in buffer")

        env_ids = data.get("env_ids", torch.zeros(T, dtype=torch.long, device=rewards.device))
        nv_data = data.get("next_values")
        if (nv_data is None or nv_data.shape != values.shape
                or not torch.isfinite(nv_data).all()):
            raise ValueError(
                "HAPPO next_values must be finite and match values shape; "
                f"got next_values={None if nv_data is None else tuple(nv_data.shape)} "
                f"values={tuple(values.shape)}")
        nv = nv_data
        if local_credit:
            advantages_raw, returns = _compute_role_local_gae(
                rewards, values, nv, dones, env_ids, self.gamma, self.gae_lambda)
        else:
            team_dones = dones[:, 0].float()
            team_reward = _alive_before_team_mean(rewards, active)
            advantages_raw, returns = _compute_grouped_gae(
                team_reward, values, nv, team_dones, env_ids, self.gamma, self.gae_lambda)
        if not torch.isfinite(advantages_raw).all() or not torch.isfinite(returns).all():
            raise ValueError("HAPPO: non-finite GAE output")
        advantages = advantages_raw
        if T > 1:
            if local_credit:
                advantages = _normalize_role_local_advantages(advantages, actor_active)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "actor_loss_per_agent": [0.0]*N, "entropy_per_agent": [0.0]*N,
            "approx_kl_per_agent": [0.0]*N,
            "approx_kl_abs_per_agent": [0.0]*N,
            "ratio_mean_per_agent": [0.0]*N,
            "ratio_std_per_agent": [0.0]*N,
            "ratio_min_per_agent": [0.0]*N,
            "ratio_max_per_agent": [0.0]*N,
            "ratio_p95_per_agent": [0.0]*N,
            "ratio_p99_per_agent": [0.0]*N,
            "clip_fraction_per_agent": [0.0]*N,
            "actor_grad_norm_per_agent": [0.0]*N,
            "policy_update_norm_per_agent": [0.0]*N,
            "ratio_after_mean_per_agent": [0.0]*N,
            "m_abs_mean_after_each_agent": [],
            "m_mean_after_each_agent": [],
            "m_std_after_each_agent": [],
            "m_abs_max_after_each_agent": [],
            "valid_sample_count_per_agent": [
                int((actor_active[:, i] > 0.5).sum().item()) for i in range(N)],
        }

        # Save initial actor params for post-update delta computation
        initial_actor_params = []
        for i in range(N):
            params_i = list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]]
            initial_actor_params.append(_param_vector(params_i))

        for epoch in range(self.ppo_epochs):
            order = list(self.rng.permutation(N))
            metrics["last_update_order"] = order
            M = advantages.detach().clone()
            correction = torch.ones(T, dtype=advantages.dtype, device=advantages.device)

            for idx_in_order, i in enumerate(order):
                valid_i = actor_active[:, i] > 0.5
                n_valid = metrics["valid_sample_count_per_agent"][i]
                if n_valid < 1:
                    metrics["ratio_after_mean_per_agent"][i] = 0.0
                    continue

                obs_i = actor_obs[:, i, :]; act_i = actions[:, i, :]
                old_lp_i = old_log_probs[:, i]
                self.actor_opts[i].zero_grad()
                new_lp_i, entropy_i, _ = self.policy.evaluate_agent_actions(i, obs_i, act_i)
                if not torch.isfinite(new_lp_i).all() or not torch.isfinite(entropy_i).all():
                    raise ValueError(f"HAPPO: non-finite lp/ent agent {i} epoch {epoch}")
                ratio_i = (new_lp_i - old_lp_i.detach()).exp()
                if not torch.isfinite(ratio_i).all():
                    raise ValueError(f"HAPPO: non-finite ratio_i agent {i} epoch {epoch}")
                ratio_stats = _valid_ratio_stats(
                    ratio_i, valid_i, self.clip_param, old_lp_i, new_lp_i)

                valid_f = valid_i.float()
                actor_advantage = (advantages[:, i] * correction).detach() if local_credit else M
                surr1 = ratio_i * actor_advantage
                surr2 = torch.clamp(ratio_i, 1 - self.clip_param, 1 + self.clip_param) * actor_advantage
                policy_loss = -(torch.min(surr1, surr2) * valid_f).sum() / valid_f.sum().clamp(min=1)
                ent_mean = (entropy_i * valid_f).sum() / valid_f.sum().clamp(min=1)
                loss = policy_loss - self.entropy_coef * ent_mean
                actor_params = list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]]
                with torch.no_grad():
                    actor_before = _param_vector(actor_params)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise ValueError(f"HAPPO: non-finite actor gradient agent {i} epoch {epoch}")
                self.actor_opts[i].step()
                with torch.no_grad():
                    actor_after = _param_vector(actor_params)
                    update_norm = float(torch.linalg.vector_norm(actor_after - actor_before).item())
                metrics["actor_loss_per_agent"][i] = float(policy_loss.item())
                metrics["entropy_per_agent"][i] = float(ent_mean.item())
                metrics["approx_kl_per_agent"][i] = float(
                    ((old_lp_i - new_lp_i.detach()) * valid_f).sum() / valid_f.sum().clamp(min=1))
                for key, value in ratio_stats.items():
                    if key == "ratio_mean":
                        metrics["ratio_mean_per_agent"][i] = value
                    elif key == "ratio_std":
                        metrics["ratio_std_per_agent"][i] = value
                    elif key == "ratio_min":
                        metrics["ratio_min_per_agent"][i] = value
                    elif key == "ratio_max":
                        metrics["ratio_max_per_agent"][i] = value
                    elif key == "ratio_p95":
                        metrics["ratio_p95_per_agent"][i] = value
                    elif key == "ratio_p99":
                        metrics["ratio_p99_per_agent"][i] = value
                    elif key == "clip_fraction":
                        metrics["clip_fraction_per_agent"][i] = value
                    elif key == "approx_kl_abs":
                        metrics["approx_kl_abs_per_agent"][i] = value
                metrics["actor_grad_norm_per_agent"][i] = _safe_float(grad_norm.item())
                metrics["policy_update_norm_per_agent"][i] = update_norm
                with torch.no_grad():
                    after_lp_i, _, _ = self.policy.evaluate_agent_actions(i, obs_i, act_i)
                    ratio_after = (after_lp_i - old_lp_i).exp()
                    ratio_after = torch.where(valid_i, ratio_after, torch.ones_like(ratio_after))
                    metrics["ratio_after_mean_per_agent"][i] = float(ratio_after[valid_i].mean().item())
                    if local_credit:
                        correction = (correction * ratio_after).detach()
                        diagnostic_m = advantages[:, i] * correction
                    else:
                        M = (M * ratio_after).detach()
                        diagnostic_m = M
                metrics["m_abs_mean_after_each_agent"].append(float(diagnostic_m.abs().mean().item()))
                metrics["m_mean_after_each_agent"].append(float(diagnostic_m.mean().item()))
                metrics["m_std_after_each_agent"].append(
                    float(diagnostic_m.float().std(unbiased=False).item()) if diagnostic_m.numel() > 1 else 0.0)
                metrics["m_abs_max_after_each_agent"].append(float(diagnostic_m.abs().max().item()))

        critic_params = list(self.policy.critic.parameters())
        with torch.no_grad():
            critic_before = _param_vector(critic_params)
        critic_loss_per_epoch = []
        critic_grad_norm_per_epoch = []
        critic_loss_unscaled = torch.tensor(0.0, device=critic_state.device)
        critic_loss = torch.tensor(0.0, device=critic_state.device)
        for critic_epoch in range(self.critic_epochs):
            self.critic_opt.zero_grad()
            new_values = self.policy.value(critic_state)
            if local_credit:
                critic_mask = active.float()
                critic_loss_unscaled = (((new_values - returns) ** 2) * critic_mask).sum() / critic_mask.sum().clamp(min=1)
            else:
                critic_loss_unscaled = F.mse_loss(new_values, returns)
            critic_loss = critic_loss_unscaled * self.value_coef
            if not torch.isfinite(critic_loss):
                raise ValueError(f"HAPPO: non-finite critic_loss epoch {critic_epoch}")
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(critic_grad_norm)):
                raise ValueError(f"HAPPO: non-finite critic gradient epoch {critic_epoch}")
            self.critic_opt.step()
            critic_loss_per_epoch.append(float(critic_loss_unscaled.item()))
            critic_grad_norm_per_epoch.append(_safe_float(critic_grad_norm.item()))
        with torch.no_grad():
            critic_after = _param_vector(critic_params)
            critic_update_norm = float(torch.linalg.vector_norm(critic_after - critic_before).item())
            new_values_final = self.policy.value(critic_state)

        # ---- Post-update final metrics (full policy on same rollout) ----
        final_ratio_mean = [0.0] * N
        final_ratio_std = [0.0] * N
        final_ratio_p95 = [0.0] * N
        final_ratio_p99 = [0.0] * N
        final_clip_fraction = [0.0] * N
        final_approx_kl = [0.0] * N
        final_approx_kl_abs = [0.0] * N
        final_actor_delta = [0.0] * N
        with torch.no_grad():
            final_lp, final_ent, _, _ = self.policy.evaluate_actions(actor_obs, critic_state, actions)
            for i in range(N):
                valid_i = actor_active[:, i] > 0.5
                nv = int(valid_i.sum().item())
                if nv < 1:
                    continue
                old_lp_i = old_log_probs[:, i]
                new_lp_i = final_lp[:, i]
                ratio_i = (new_lp_i - old_lp_i).exp()
                ratio_i_valid = ratio_i[valid_i]
                final_ratio_mean[i] = float(ratio_i_valid.mean().item())
                final_ratio_std[i] = float(ratio_i_valid.std(unbiased=False).item()) if nv > 1 else 0.0
                final_ratio_p95[i] = float(torch.quantile(ratio_i_valid, 0.95).item()) if nv > 0 else 0.0
                final_ratio_p99[i] = float(torch.quantile(ratio_i_valid, 0.99).item()) if nv > 0 else 0.0
                final_clip_fraction[i] = float(
                    ((ratio_i_valid < 1 - self.clip_param) | (ratio_i_valid > 1 + self.clip_param)).float().mean().item())
                final_approx_kl[i] = float(((old_lp_i[valid_i] - new_lp_i[valid_i]).float().mean().item()))
                final_approx_kl_abs[i] = float(((old_lp_i[valid_i] - new_lp_i[valid_i]).abs().float().mean().item()))
                params_i = list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]]
                final_actor_delta[i] = float(
                    torch.linalg.vector_norm(_param_vector(params_i) - initial_actor_params[i]).item())

        with torch.no_grad():
            self.policy.evaluate_actions(actor_obs, critic_state, actions)
            action_values = actions
            mav_sat = float((action_values[:, 0, :].abs() >= 0.999).float().mean()) if N > 0 else 0.0
            uav_sat = float((action_values[:, 1:, :].abs() >= 0.999).float().mean()) if N > 1 else 0.0

        log_std_vals = torch.stack([p.data for p in self.policy.action_log_stds])
        v = metrics["valid_sample_count_per_agent"]
        alive_counts = [int((active[:, i] > 0.5).sum().item()) for i in range(N)]
        active_sample_ratio = [float(count) / float(max(T, 1)) for count in v]
        ls_mav = log_std_vals[0] if N > 0 else torch.zeros(3)
        ls_uav = log_std_vals[1:].flatten() if N > 1 else torch.zeros(3)
        raw_adv_stats = _stats_from_tensor(advantages_raw)
        norm_adv_stats = _stats_from_tensor(advantages)
        return_stats = _stats_from_tensor(returns)
        value_old_stats = _stats_from_tensor(values)
        value_new_stats = _stats_from_tensor(new_values_final)
        value_ev_old = _explained_variance(values, returns)
        value_ev_new = _explained_variance(new_values_final, returns)
        mav_indices = [0] if N > 0 else []
        uav_indices = list(range(1, N))
        action_names = ("pitch", "heading", "speed")
        action_stats = {}
        for role_name, tensor, mask in (
            (
                "mav",
                action_values[:, 0:1, :] if N > 0 else action_values[:, 0:0, :],
                active[:, 0:1] if N > 0 else active[:, 0:0],
            ),
            (
                "uav",
                action_values[:, 1:, :] if N > 1 else action_values[:, 0:0, :],
                active[:, 1:] if N > 1 else active[:, 0:0],
            ),
        ):
            flat = tensor.reshape(-1, action_values.shape[-1]) if tensor.numel() else torch.zeros((0, action_values.shape[-1]), device=action_values.device)
            flat_mask = mask.reshape(-1) > 0.5 if mask.numel() else torch.zeros((0,), dtype=torch.bool, device=action_values.device)
            for dim, name in enumerate(action_names):
                if dim >= action_values.shape[-1] or flat.numel() == 0:
                    vals = torch.zeros(0, device=action_values.device)
                else:
                    vals = flat[:, dim]
                active_vals = vals[flat_mask] if vals.numel() and flat_mask.numel() else torch.zeros(0, device=action_values.device)
                action_stats[f"{role_name}_action_mean_{name}"] = float(vals.mean().item()) if vals.numel() else 0.0
                action_stats[f"{role_name}_action_std_{name}"] = (
                    float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0)
                action_stats[f"{role_name}_action_mean_abs_{name}"] = float(vals.abs().mean().item()) if vals.numel() else 0.0
                action_stats[f"{role_name}_action_saturation_{name}"] = (
                    float((vals.abs() >= 0.999).float().mean().item()) if vals.numel() else 0.0)
                action_stats[f"{role_name}_action_mean_{name}_active"] = (
                    float(active_vals.mean().item()) if active_vals.numel() else 0.0)
                action_stats[f"{role_name}_action_std_{name}_active"] = (
                    float(active_vals.std(unbiased=False).item()) if active_vals.numel() > 1 else 0.0)
                action_stats[f"{role_name}_action_mean_abs_{name}_active"] = (
                    float(active_vals.abs().mean().item()) if active_vals.numel() else 0.0)
                action_stats[f"{role_name}_action_saturation_{name}_active"] = (
                    float((active_vals.abs() >= 0.999).float().mean().item()) if active_vals.numel() else 0.0)
        critic_loss_mean = float(np.mean(critic_loss_per_epoch)) if critic_loss_per_epoch else 0.0
        critic_grad_mean = float(np.mean(critic_grad_norm_per_epoch)) if critic_grad_norm_per_epoch else 0.0
        critic_grad_max = float(np.max(critic_grad_norm_per_epoch)) if critic_grad_norm_per_epoch else 0.0
        return {
            "actor_loss_mean": float(np.mean([metrics["actor_loss_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "actor_loss_per_agent": metrics["actor_loss_per_agent"],
            "entropy_mean": float(np.mean([metrics["entropy_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "entropy_per_agent": metrics["entropy_per_agent"],
            "approx_kl_mean": float(np.mean([metrics["approx_kl_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "approx_kl_per_agent": metrics["approx_kl_per_agent"],
            "critic_loss": float(critic_loss.item()),
            "critic_loss_unscaled": float(critic_loss_unscaled.item()),
            "critic_loss_scaled": float(critic_loss.item()),
            "critic_loss_mean_over_epochs": critic_loss_mean,
            "critic_loss_first_epoch": critic_loss_per_epoch[0] if critic_loss_per_epoch else 0.0,
            "critic_loss_last_epoch": critic_loss_per_epoch[-1] if critic_loss_per_epoch else 0.0,
            "critic_loss_per_epoch": critic_loss_per_epoch,
            "critic_grad_norm": critic_grad_norm_per_epoch[-1] if critic_grad_norm_per_epoch else 0.0,
            "critic_grad_norm_mean_over_epochs": critic_grad_mean,
            "critic_grad_norm_max_over_epochs": critic_grad_max,
            "critic_grad_norm_per_epoch": critic_grad_norm_per_epoch,
            "critic_update_norm": critic_update_norm,
            "advantage_raw_mean": raw_adv_stats["mean"],
            "advantage_raw_std": raw_adv_stats["std"],
            "advantage_raw_min": raw_adv_stats["min"],
            "advantage_raw_max": raw_adv_stats["max"],
            "advantage_norm_mean": norm_adv_stats["mean"],
            "advantage_norm_std": norm_adv_stats["std"],
            "advantage_norm_min": norm_adv_stats["min"],
            "advantage_norm_max": norm_adv_stats["max"],
            "return_mean": return_stats["mean"],
            "return_std": return_stats["std"],
            "return_min": return_stats["min"],
            "return_max": return_stats["max"],
            "value_pred_old_mean": value_old_stats["mean"],
            "value_pred_old_std": value_old_stats["std"],
            "value_pred_old_min": value_old_stats["min"],
            "value_pred_old_max": value_old_stats["max"],
            "value_pred_new_mean": value_new_stats["mean"],
            "value_pred_new_std": value_new_stats["std"],
            "value_pred_new_min": value_new_stats["min"],
            "value_pred_new_max": value_new_stats["max"],
            "value_explained_variance_old": value_ev_old,
            "value_explained_variance_new": value_ev_new,
            "value_pred_mean": value_new_stats["mean"],
            "value_pred_std": value_new_stats["std"],
            "value_pred_min": value_new_stats["min"],
            "value_pred_max": value_new_stats["max"],
            "value_explained_variance": value_ev_new,
            "critic_epochs": self.critic_epochs,
            "last_update_order": list(order),
            "action_log_std_min": float(log_std_vals.min().item()),
            "action_log_std_max": float(log_std_vals.max().item()),
            "action_log_std_mean": float(log_std_vals.mean().item()),
            "action_log_std_mav_min": float(ls_mav.min().item()) if ls_mav.numel() else 0.0,
            "action_log_std_mav_max": float(ls_mav.max().item()) if ls_mav.numel() else 0.0,
            "action_log_std_mav_mean": float(ls_mav.mean().item()) if ls_mav.numel() else 0.0,
            "action_log_std_uav_min": float(ls_uav.min().item()) if ls_uav.numel() else 0.0,
            "action_log_std_uav_max": float(ls_uav.max().item()) if ls_uav.numel() else 0.0,
            "action_log_std_uav_mean": float(ls_uav.mean().item()) if ls_uav.numel() else 0.0,
            "actor_loss_mav": metrics["actor_loss_per_agent"][0] if N > 0 else 0.0,
            "actor_loss_uav": float(np.mean(metrics["actor_loss_per_agent"][1:])) if N > 1 else 0.0,
            "entropy_mav": metrics["entropy_per_agent"][0] if N > 0 else 0.0,
            "entropy_uav": float(np.mean(metrics["entropy_per_agent"][1:])) if N > 1 else 0.0,
            "approx_kl_mav": metrics["approx_kl_per_agent"][0] if N > 0 else 0.0,
            "approx_kl_uav": float(np.mean(metrics["approx_kl_per_agent"][1:])) if N > 1 else 0.0,
            "approx_kl_abs_mav": metrics["approx_kl_abs_per_agent"][0] if N > 0 else 0.0,
            "approx_kl_abs_uav": _mean_for_indices(metrics["approx_kl_abs_per_agent"], uav_indices),
            "clip_fraction_mav": metrics["clip_fraction_per_agent"][0] if N > 0 else 0.0,
            "clip_fraction_uav": _mean_for_indices(metrics["clip_fraction_per_agent"], uav_indices),
            "ratio_mean_mav": metrics["ratio_mean_per_agent"][0] if N > 0 else 0.0,
            "ratio_mean_uav": _mean_for_indices(metrics["ratio_mean_per_agent"], uav_indices),
            "ratio_std_mav": metrics["ratio_std_per_agent"][0] if N > 0 else 0.0,
            "ratio_std_uav": _mean_for_indices(metrics["ratio_std_per_agent"], uav_indices),
            "ratio_p95_mav": metrics["ratio_p95_per_agent"][0] if N > 0 else 0.0,
            "ratio_p95_uav": _mean_for_indices(metrics["ratio_p95_per_agent"], uav_indices),
            "ratio_p99_mav": metrics["ratio_p99_per_agent"][0] if N > 0 else 0.0,
            "ratio_p99_uav": _mean_for_indices(metrics["ratio_p99_per_agent"], uav_indices),
            "actor_grad_norm_mav": metrics["actor_grad_norm_per_agent"][0] if N > 0 else 0.0,
            "actor_grad_norm_uav": _mean_for_indices(metrics["actor_grad_norm_per_agent"], uav_indices),
            "policy_update_norm_mav": metrics["policy_update_norm_per_agent"][0] if N > 0 else 0.0,
            "policy_update_norm_uav": _mean_for_indices(metrics["policy_update_norm_per_agent"], uav_indices),
            "mav_active_sample_count": v[0] if N > 0 else 0,
            "uav_active_sample_count": sum(v[1:]) if N > 1 else 0,
            "mav_alive_sample_count": alive_counts[0] if N > 0 else 0,
            "uav_alive_sample_count": sum(alive_counts[1:]) if N > 1 else 0,
            "mav_actor_effective_fraction": float(v[0]) / max(float(alive_counts[0]), 1.0) if N > 0 else 0.0,
            "uav_actor_effective_fraction": float(sum(v[1:])) / max(float(sum(alive_counts[1:])), 1.0) if N > 1 else 0.0,
            "credit_mode": getattr(self.policy, "credit_mode", "shared_alive_team_mean"),
            "entropy_mav_valid_count": v[0] if N > 0 else 0,
            "entropy_uav_valid_count": sum(v[1:]) if N > 1 else 0,
            "mav_action_saturation_rate": mav_sat,
            "uav_action_saturation_rate": uav_sat,
            **action_stats,
            "ratio_after_mean_per_agent": metrics["ratio_after_mean_per_agent"],
            "m_abs_mean_after_each_agent": metrics["m_abs_mean_after_each_agent"],
            "m_mean_after_each_agent": metrics["m_mean_after_each_agent"],
            "m_std_after_each_agent": metrics["m_std_after_each_agent"],
            "m_abs_max_after_each_agent": metrics["m_abs_max_after_each_agent"],
            "valid_sample_count_per_agent": v,
            "active_sample_ratio_per_agent": active_sample_ratio,
            "final_ratio_mean_per_agent": final_ratio_mean,
            "final_ratio_std_per_agent": final_ratio_std,
            "final_ratio_p95_per_agent": final_ratio_p95,
            "final_ratio_p99_per_agent": final_ratio_p99,
            "final_clip_fraction_per_agent": final_clip_fraction,
            "final_approx_kl_per_agent": final_approx_kl,
            "final_approx_kl_abs_per_agent": final_approx_kl_abs,
            "final_actor_parameter_delta_per_agent": final_actor_delta,
            "final_ratio_mean_mav": final_ratio_mean[0] if N > 0 else 0.0,
            "final_ratio_mean_uav": _mean_for_indices(final_ratio_mean, uav_indices),
            "final_ratio_std_mav": final_ratio_std[0] if N > 0 else 0.0,
            "final_ratio_std_uav": _mean_for_indices(final_ratio_std, uav_indices),
            "final_ratio_p95_mav": final_ratio_p95[0] if N > 0 else 0.0,
            "final_ratio_p95_uav": _mean_for_indices(final_ratio_p95, uav_indices),
            "final_ratio_p99_mav": final_ratio_p99[0] if N > 0 else 0.0,
            "final_ratio_p99_uav": _mean_for_indices(final_ratio_p99, uav_indices),
            "final_clip_fraction_mav": final_clip_fraction[0] if N > 0 else 0.0,
            "final_clip_fraction_uav": _mean_for_indices(final_clip_fraction, uav_indices),
            "final_approx_kl_mav": final_approx_kl[0] if N > 0 else 0.0,
            "final_approx_kl_uav": _mean_for_indices(final_approx_kl, uav_indices),
            "final_approx_kl_abs_mav": final_approx_kl_abs[0] if N > 0 else 0.0,
            "final_approx_kl_abs_uav": _mean_for_indices(final_approx_kl_abs, uav_indices),
            "final_actor_parameter_delta_mav": final_actor_delta[0] if N > 0 else 0.0,
            "final_actor_parameter_delta_uav": _mean_for_indices(final_actor_delta, uav_indices),
            "ratio_mean_per_agent": metrics["ratio_mean_per_agent"],
            "ratio_std_per_agent": metrics["ratio_std_per_agent"],
            "ratio_min_per_agent": metrics["ratio_min_per_agent"],
            "ratio_max_per_agent": metrics["ratio_max_per_agent"],
            "ratio_p95_per_agent": metrics["ratio_p95_per_agent"],
            "ratio_p99_per_agent": metrics["ratio_p99_per_agent"],
            "clip_fraction_per_agent": metrics["clip_fraction_per_agent"],
            "approx_kl_abs_per_agent": metrics["approx_kl_abs_per_agent"],
            "actor_grad_norm_per_agent": metrics["actor_grad_norm_per_agent"],
            "policy_update_norm_per_agent": metrics["policy_update_norm_per_agent"],
            "reward_nan_count": int((~torch.isfinite(rewards)).sum().item()),
            "action_nan_count": int((~torch.isfinite(actions)).sum().item()),
            "value_nan_count": int((~torch.isfinite(values)).sum().item()),
            "log_prob_nan_count": int((~torch.isfinite(old_log_probs)).sum().item()),
            "gradient_nonfinite_count": 0,
        }
