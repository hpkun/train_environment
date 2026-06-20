"""Sequence-replay role-level HAPPO trainer for categorical TAM policies."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID
from .happo_trainer import _compute_grouped_gae


def _unique_parameters(parameters):
    seen = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


class TAMCategoricalHAPPOTrainer:
    """Time-ordered recurrent PPO with MAV->UAV role-level correction."""

    def __init__(
        self, policy, actor_lr=2e-4, critic_lr=5e-4, clip_param=0.2,
        entropy_coef=0.02, value_coef=0.5, max_grad_norm=10.0,
        ppo_epochs=4, gamma=0.99, gae_lambda=0.95,
    ):
        if getattr(policy, "action_distribution", None) != "multidiscrete_categorical":
            raise ValueError("TAMCategoricalHAPPOTrainer requires categorical policy")
        self.policy = policy
        shared = list(policy.actor_shared_parameters())
        self.mav_params = _unique_parameters(shared + list(policy.mav_actor.parameters()))
        self.uav_params = _unique_parameters(shared + list(policy.uav_actor.parameters()))
        self.mav_opt = torch.optim.Adam(self.mav_params, lr=actor_lr)
        self.uav_opt = torch.optim.Adam(self.uav_params, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param = float(clip_param)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.role_update_order = (MAV_ROLE_ID, UAV_ROLE_ID)
        self.happo_correction = "role_level"
        self.recurrent_update = "sequence_replay"

    @staticmethod
    def role_importance_ratio(new_log_prob, old_log_prob):
        return torch.exp(new_log_prob - old_log_prob).detach()

    def _advantages_and_returns(self, data):
        active = data["active_masks"]
        valid_count = active.sum(-1).clamp(min=1.0)
        team_reward = (data["rewards"] * active).sum(-1) / valid_count
        team_dones = data["dones"][:, 0].float()
        next_values = data["next_values"]
        if torch.isnan(next_values).any():
            raise ValueError("categorical sequence trainer requires per-step next_values")
        advantages, returns = _compute_grouped_gae(
            team_reward, data["values"], next_values, team_dones,
            data["env_ids"], self.gamma, self.gae_lambda,
        )
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.detach(), returns.detach()

    def _evaluate_sequence(self, sequence):
        return self.policy.evaluate_action_sequence(
            sequence["actor_obs"], sequence["role_ids"],
            None, sequence["actions"],
            initial_hidden=sequence["rnn_hidden_initial"],
            episode_start_masks=sequence["episode_start_masks"],
            active_masks=sequence["agent_alive_masks"],
        )

    def _update_role(self, sequences, advantages, correction, role_id, optimizer, params):
        optimizer.zero_grad()
        loss_sum = torch.zeros((), device=advantages.device)
        entropy_sum = torch.zeros((), device=advantages.device)
        kl_sum = torch.zeros((), device=advantages.device)
        valid_sum = torch.zeros((), device=advantages.device)
        for sequence in sequences:
            out = self._evaluate_sequence(sequence)
            indices = sequence["buffer_indices"]
            role_mask = (sequence["role_ids"].view(1, -1) == role_id).float()
            valid = sequence["agent_alive_masks"] * role_mask
            old = sequence["old_log_probs"]
            ratio = torch.exp(out["log_prob"] - old)
            corrected_advantage = advantages[indices].unsqueeze(-1) * correction[indices]
            surrogate1 = ratio * corrected_advantage
            surrogate2 = torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            ) * corrected_advantage
            loss_sum = loss_sum - (torch.minimum(surrogate1, surrogate2) * valid).sum()
            entropy_sum = entropy_sum + (out["entropy"] * valid).sum()
            kl_sum = kl_sum + ((old - out["log_prob"]) * valid).sum()
            valid_sum = valid_sum + valid.sum()
        if valid_sum.item() <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        policy_loss = loss_sum / valid_sum
        entropy_mean = entropy_sum / valid_sum
        (policy_loss - self.entropy_coef * entropy_mean).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            for sequence in sequences:
                out = self._evaluate_sequence(sequence)
                indices = sequence["buffer_indices"]
                role_mask = (sequence["role_ids"].view(1, -1) == role_id)
                valid = (sequence["agent_alive_masks"] > 0.5) & role_mask
                role_ratio = self.role_importance_ratio(
                    out["log_prob"], sequence["old_log_probs"]
                )
                updated = torch.where(valid, role_ratio, torch.ones_like(role_ratio))
                correction[indices] = correction[indices] * updated
        return (
            float(policy_loss.item()), float(entropy_mean.item()),
            float((kl_sum / valid_sum).item()), float(valid_sum.item()),
            float(torch.as_tensor(grad_norm).item()),
        )

    def _distribution_metrics(self, sequences):
        actions = torch.cat([sequence["actions"].reshape(-1, 4) for sequence in sequences])
        active = torch.cat([sequence["agent_alive_masks"].reshape(-1) for sequence in sequences]) > 0.5
        roles = torch.cat([
            sequence["role_ids"].view(1, -1).expand(sequence["actions"].shape[:2]).reshape(-1)
            for sequence in sequences
        ])
        with torch.no_grad():
            probabilities = torch.cat([
                self._evaluate_sequence(sequence)["action_probs"].reshape(-1, 4, self.policy.action_levels)
                for sequence in sequences
            ])
        levels = self.policy.action_levels
        selected = actions[active]

        def role_metrics(role_id):
            mask = active & (roles == role_id)
            if not mask.any():
                return 0.0, 0.0
            maximum = probabilities[mask].max(-1).values.mean()
            usage = torch.unique(actions[mask]).numel() / levels
            return float(maximum.item()), float(usage)

        mav_prob, mav_usage = role_metrics(MAV_ROLE_ID)
        uav_prob, uav_usage = role_metrics(UAV_ROLE_ID)
        return {
            "edge_bin_rate": float(((selected == 0) | (selected == levels - 1)).float().mean().item()),
            "low_bin_rate": float((selected == 0).float().mean().item()),
            "high_bin_rate": float((selected == levels - 1).float().mean().item()),
            "throttle_high_rate": float((selected[:, 0] >= levels - 4).float().mean().item()),
            "surface_edge_rate": float(((selected[:, 1:] == 0) | (selected[:, 1:] == levels - 1)).float().mean().item()),
            "max_action_prob_mav": mav_prob,
            "max_action_prob_uav": uav_prob,
            "action_bin_usage_mav": mav_usage,
            "action_bin_usage_uav": uav_usage,
        }

    def update(self, buffer):
        device = next(self.policy.parameters()).device
        data = buffer.get(device)
        sequences = buffer.get_sequences(device)
        advantages, returns = self._advantages_and_returns(data)
        role_stats = {MAV_ROLE_ID: [], UAV_ROLE_ID: []}
        critic_losses, critic_grad_norms, correction_values = [], [], []

        for _epoch in range(self.ppo_epochs):
            self.critic_opt.zero_grad()
            values = self.policy.value(data["critic_state"])
            critic_loss = F.mse_loss(values, returns) * self.value_coef
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.item()))
            critic_grad_norms.append(float(torch.as_tensor(critic_grad).item()))

            correction = torch.ones_like(data["old_log_probs"])
            for role_id, optimizer, params in (
                (MAV_ROLE_ID, self.mav_opt, self.mav_params),
                (UAV_ROLE_ID, self.uav_opt, self.uav_params),
            ):
                role_stats[role_id].append(self._update_role(
                    sequences, advantages, correction, role_id, optimizer, params
                ))
            if not torch.isfinite(correction).all():
                raise ValueError("non-finite HAPPO correction factor")
            correction_values.append(correction.detach().reshape(-1))

        for name, parameter in self.policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"non-finite policy parameter after update: {name}")

        mav = np.asarray(role_stats[MAV_ROLE_ID], dtype=np.float64)
        uav = np.asarray(role_stats[UAV_ROLE_ID], dtype=np.float64)
        correction = torch.cat(correction_values)
        metrics = {
            "actor_loss_mav": float(mav[:, 0].mean()),
            "actor_loss_uav": float(uav[:, 0].mean()),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy_mav": float(mav[:, 1].mean()),
            "entropy_uav": float(uav[:, 1].mean()),
            "approx_kl_mav": float(mav[:, 2].mean()),
            "approx_kl_uav": float(uav[:, 2].mean()),
            "mav_active_sample_count": float(mav[:, 3].mean()),
            "uav_active_sample_count": float(uav[:, 3].mean()),
            "grad_norm_actor": float(np.mean(np.concatenate([mav[:, 4], uav[:, 4]]))),
            "grad_norm_critic": float(np.mean(critic_grad_norms)),
            "correction_factor_mean": float(correction.mean().item()),
            "correction_factor_max": float(correction.max().item()),
            "correction_factor_min": float(correction.min().item()),
        }
        metrics.update(self._distribution_metrics(sequences))
        return metrics
