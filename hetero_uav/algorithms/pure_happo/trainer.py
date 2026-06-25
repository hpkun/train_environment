"""Paper-aligned HAPPO baseline: independent per-agent actors + shared V critic.

ICLR 2022 HAPPO algorithm: random sequential agent update with
PPO-clip and correction factor M. Grouped GAE by env_id.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


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


class PureHAPPOTrainer:
    """Paper-aligned HAPPO trainer."""

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
        actor_obs = data["actor_obs"]; critic_state = data["critic_state"]
        actions = data["actions"]; old_log_probs = data["old_log_probs"]
        rewards = data["rewards"]; dones = data["dones"]
        values = data["values"]; active = data["active_masks"]

        T, N = actor_obs.shape[:2]
        for name, tensor in [("actor_obs", actor_obs), ("critic_state", critic_state),
                              ("actions", actions), ("old_log_probs", old_log_probs),
                              ("rewards", rewards), ("dones", dones),
                              ("values", values), ("active_masks", active)]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"HAPPO: non-finite {name} in buffer")

        team_reward = (rewards * active).sum(dim=-1) / active.sum(dim=-1).clamp(min=1)
        team_dones = dones[:, 0].float()
        env_ids = data.get("env_ids", torch.zeros(T, dtype=torch.long, device=rewards.device))
        nv_data = data.get("next_values")
        if nv_data is not None and not torch.isnan(nv_data).any() and nv_data.numel() == T:
            nv = nv_data
        else:
            with torch.no_grad():
                nv_single = self.policy.value(critic_state[-1:])
            nv = nv_single.expand(T)
        advantages, returns = _compute_grouped_gae(
            team_reward, values, nv, team_dones, env_ids, self.gamma, self.gae_lambda)
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            raise ValueError("HAPPO: non-finite GAE output")
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "actor_loss_per_agent": [0.0]*N, "entropy_per_agent": [0.0]*N,
            "approx_kl_per_agent": [0.0]*N,
            "ratio_after_mean_per_agent": [0.0]*N,
            "m_abs_mean_after_each_agent": [],
            "valid_sample_count_per_agent": [
                int((active[:, i] > 0.5).sum().item()) for i in range(N)],
        }

        for epoch in range(self.ppo_epochs):
            order = list(self.rng.permutation(N))
            metrics["last_update_order"] = order
            M = advantages.detach().clone()

            for idx_in_order, i in enumerate(order):
                valid_i = active[:, i] > 0.5
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

                valid_f = valid_i.float()
                surr1 = ratio_i * M
                surr2 = torch.clamp(ratio_i, 1 - self.clip_param, 1 + self.clip_param) * M
                policy_loss = -(torch.min(surr1, surr2) * valid_f).sum() / valid_f.sum().clamp(min=1)
                ent_mean = (entropy_i * valid_f).sum() / valid_f.sum().clamp(min=1)
                loss = policy_loss - self.entropy_coef * ent_mean
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.actors[i].parameters()) + [self.policy.action_log_stds[i]],
                    self.max_grad_norm)
                self.actor_opts[i].step()
                metrics["actor_loss_per_agent"][i] = float(policy_loss.item())
                metrics["entropy_per_agent"][i] = float(ent_mean.item())
                metrics["approx_kl_per_agent"][i] = float(
                    ((old_lp_i - new_lp_i.detach()) * valid_f).sum() / valid_f.sum().clamp(min=1))
                with torch.no_grad():
                    after_lp_i, _, _ = self.policy.evaluate_agent_actions(i, obs_i, act_i)
                    ratio_after = (after_lp_i - old_lp_i).exp()
                    ratio_after = torch.where(valid_i, ratio_after, torch.ones_like(ratio_after))
                    metrics["ratio_after_mean_per_agent"][i] = float(ratio_after[valid_i].mean().item())
                    M = (M * ratio_after).detach()
                metrics["m_abs_mean_after_each_agent"].append(float(M.abs().mean().item()))

        self.critic_opt.zero_grad()
        new_values = self.policy.value(critic_state)
        critic_loss = F.mse_loss(new_values, returns) * self.value_coef
        if not torch.isfinite(critic_loss):
            raise ValueError("HAPPO: non-finite critic_loss")
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        with torch.no_grad():
            _, _, _, means_all = self.policy.evaluate_actions(actor_obs, critic_state, actions)
            mav_sat = float((means_all[:, 0, :].abs() >= 0.999).float().mean()) if N > 0 else 0.0
            uav_sat = float((means_all[:, 1:, :].abs() >= 0.999).float().mean()) if N > 1 else 0.0

        log_std_vals = torch.stack([p.data for p in self.policy.action_log_stds])
        v = metrics["valid_sample_count_per_agent"]
        ls_mav = log_std_vals[0] if N > 0 else torch.zeros(3)
        ls_uav = log_std_vals[1:].flatten() if N > 1 else torch.zeros(3)
        return {
            "actor_loss_mean": float(np.mean([metrics["actor_loss_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "actor_loss_per_agent": metrics["actor_loss_per_agent"],
            "entropy_mean": float(np.mean([metrics["entropy_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "entropy_per_agent": metrics["entropy_per_agent"],
            "approx_kl_mean": float(np.mean([metrics["approx_kl_per_agent"][i] for i in range(N) if v[i] > 0])) if any(x > 0 for x in v) else 0.0,
            "approx_kl_per_agent": metrics["approx_kl_per_agent"],
            "critic_loss": float(critic_loss.item()),
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
            "mav_active_sample_count": v[0] if N > 0 else 0,
            "uav_active_sample_count": sum(v[1:]) if N > 1 else 0,
            "mav_action_saturation_rate": mav_sat,
            "uav_action_saturation_rate": uav_sat,
            "ratio_after_mean_per_agent": metrics["ratio_after_mean_per_agent"],
            "m_abs_mean_after_each_agent": metrics["m_abs_mean_after_each_agent"],
            "valid_sample_count_per_agent": v,
        }
