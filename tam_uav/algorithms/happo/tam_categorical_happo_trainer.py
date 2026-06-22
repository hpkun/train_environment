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
        mav_actor_lr_scale=1.0, uav_actor_lr_scale=1.0,
        mav_entropy_coef=None, uav_entropy_coef=None,
        mav_clip_param=None, uav_clip_param=None,
        mav_target_kl=0.0, uav_target_kl=0.0,
        role_kl_early_stop=False,
        mav_shared_update_mode="full",
        happo_update_granularity: str = "role",
        agent_ids: list[str] | None = None,
    ):
        if happo_update_granularity not in ("role", "agent"):
            raise ValueError("happo_update_granularity must be 'role' or 'agent'")
        if getattr(policy, "action_distribution", None) != "multidiscrete_categorical":
            raise ValueError("TAMCategoricalHAPPOTrainer requires categorical policy")
        self.policy = policy
        self.shared_actor_params = _unique_parameters(policy.actor_shared_parameters())
        self.mav_params = _unique_parameters(policy.mav_actor.parameters())
        self.uav_params = _unique_parameters(policy.uav_actor.parameters())
        self.shared_actor_opt = (
            torch.optim.Adam(self.shared_actor_params, lr=actor_lr)
            if self.shared_actor_params else None
        )
        self.mav_actor_lr_effective = float(actor_lr) * float(mav_actor_lr_scale)
        self.uav_actor_lr_effective = float(actor_lr) * float(uav_actor_lr_scale)
        self.mav_opt = torch.optim.Adam(
            self.mav_params, lr=self.mav_actor_lr_effective
        )
        self.uav_opt = torch.optim.Adam(
            self.uav_params, lr=self.uav_actor_lr_effective
        )
        self.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.clip_param = float(clip_param)
        self.entropy_coef = float(entropy_coef)
        self.role_entropy_coef = {
            MAV_ROLE_ID: float(entropy_coef if mav_entropy_coef is None else mav_entropy_coef),
            UAV_ROLE_ID: float(entropy_coef if uav_entropy_coef is None else uav_entropy_coef),
        }
        self.role_clip_param = {
            MAV_ROLE_ID: float(clip_param if mav_clip_param is None else mav_clip_param),
            UAV_ROLE_ID: float(clip_param if uav_clip_param is None else uav_clip_param),
        }
        self.role_target_kl = {
            MAV_ROLE_ID: float(mav_target_kl),
            UAV_ROLE_ID: float(uav_target_kl),
        }
        self.role_kl_early_stop = bool(role_kl_early_stop)
        if mav_shared_update_mode not in {"full", "head_only"}:
            raise ValueError("mav_shared_update_mode must be full or head_only")
        self.mav_shared_update_mode = mav_shared_update_mode
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.happo_update_granularity = happo_update_granularity
        self.agent_ids = list(agent_ids) if agent_ids else ["red_0", "red_1", "red_2"]
        self.role_update_order = (MAV_ROLE_ID, UAV_ROLE_ID)
        self.happo_correction = "agent_sequential" if happo_update_granularity == "agent" else "role_level"
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

    def _update_role(
        self, sequences, advantages, correction, role_id, optimizer, params,
        force_skip_by_kl=False, agent_index=None,
    ):
        if self.shared_actor_opt is not None:
            self.shared_actor_opt.zero_grad()
        else:
            for parameter in self.shared_actor_params:
                parameter.grad = None
        optimizer.zero_grad()
        loss_sum = torch.zeros((), device=advantages.device)
        entropy_sum = torch.zeros((), device=advantages.device)
        kl_sum = torch.zeros((), device=advantages.device)
        valid_sum = torch.zeros((), device=advantages.device)
        for sequence in sequences:
            out = self._evaluate_sequence(sequence)
            indices = sequence["buffer_indices"]
            role_mask = (sequence["role_ids"].view(1, -1) == role_id).float()
            if agent_index is not None:
                agent_mask = torch.zeros_like(role_mask)
                agent_mask[:, agent_index] = 1.0
                role_mask = role_mask * agent_mask
            valid = sequence["agent_alive_masks"] * role_mask
            old = sequence["old_log_probs"]
            ratio = torch.exp(out["log_prob"] - old)
            corrected_advantage = advantages[indices].unsqueeze(-1) * correction[indices]
            surrogate1 = ratio * corrected_advantage
            surrogate2 = torch.clamp(
                ratio,
                1.0 - self.role_clip_param[role_id],
                1.0 + self.role_clip_param[role_id],
            ) * corrected_advantage
            loss_sum = loss_sum - (torch.minimum(surrogate1, surrogate2) * valid).sum()
            entropy_sum = entropy_sum + (out["entropy"] * valid).sum()
            log_ratio = out["log_prob"] - old
            approx_kl = (ratio - 1.0) - log_ratio
            kl_sum = kl_sum + (approx_kl * valid).sum()
            valid_sum = valid_sum + valid.sum()
        if valid_sum.item() <= 0:
            return (0.0,) * 9
        policy_loss = loss_sum / valid_sum
        entropy_mean = entropy_sum / valid_sum
        approx_kl_mean = kl_sum / valid_sum
        target_kl = self.role_target_kl[role_id]
        kl_triggered = bool(
            self.role_kl_early_stop and target_kl > 0.0
            and approx_kl_mean.item() > target_kl
        )
        if force_skip_by_kl or kl_triggered:
            for parameter in self.shared_actor_params:
                parameter.grad = None
            return (
                float(policy_loss.item()), float(entropy_mean.item()),
                float(approx_kl_mean.item()), float(valid_sum.item()),
                0.0, 0.0, 0.0, float(kl_triggered), 1.0,
            )
        (policy_loss - self.role_entropy_coef[role_id] * entropy_mean).backward()
        shared_grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.shared_actor_params, self.max_grad_norm
            ) if self.shared_actor_params else torch.zeros(())
        )
        head_grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        shared_step_enabled = (
            self.shared_actor_opt is not None
            and (role_id != MAV_ROLE_ID or self.mav_shared_update_mode == "full")
        )
        if shared_step_enabled:
            self.shared_actor_opt.step()
        for parameter in self.shared_actor_params:
            parameter.grad = None

        shared_grad_value = float(torch.as_tensor(shared_grad_norm).item())
        head_grad_value = float(torch.as_tensor(head_grad_norm).item())
        actor_grad_value = float(np.hypot(shared_grad_value, head_grad_value))

        with torch.no_grad():
            for sequence in sequences:
                out = self._evaluate_sequence(sequence)
                indices = sequence["buffer_indices"]
                role_mask = (sequence["role_ids"].view(1, -1) == role_id)
                if agent_index is not None:
                    agent_mask = torch.zeros_like(role_mask)
                    agent_mask[:, agent_index] = True
                    role_mask = role_mask & agent_mask
                valid = (sequence["agent_alive_masks"] > 0.5) & role_mask
                role_ratio = self.role_importance_ratio(
                    out["log_prob"], sequence["old_log_probs"]
                )
                updated = torch.where(valid, role_ratio, torch.ones_like(role_ratio))
                correction[indices] = correction[indices] * updated
        return (
            float(policy_loss.item()), float(entropy_mean.item()),
            float(approx_kl_mean.item()), float(valid_sum.item()),
            actor_grad_value, shared_grad_value, head_grad_value, 0.0, 0.0,
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

        axis_names = ("throttle", "aileron", "elevator", "rudder")

        def role_metrics(role_id):
            mask = active & (roles == role_id)
            if not mask.any():
                return 0.0, 0.0, 0.0, [0.0] * 4, [0] * 4
            role_probs = probabilities[mask]
            maximum = role_probs.max(-1).values.mean()
            usage = torch.unique(actions[mask]).numel() / levels
            prior = self.policy.neutral_prior_probabilities(role_id).to(role_probs)
            safe_probs = role_probs.clamp_min(1e-8)
            safe_prior = prior.clamp_min(1e-8)
            per_sample_axis_kl = (
                safe_probs * (safe_probs.log() - safe_prior.log())
            ).sum(-1)
            per_axis_kl = per_sample_axis_kl.mean(0)
            dominant = role_probs.mean(0).argmax(-1)
            return (
                float(maximum.item()), float(usage),
                float(per_axis_kl.sum().item()),
                [float(value) for value in per_axis_kl.cpu()],
                [int(value) for value in dominant.cpu()],
            )

        mav_prob, mav_usage, mav_kl, mav_axis_kl, mav_dominant = role_metrics(MAV_ROLE_ID)
        uav_prob, uav_usage, uav_kl, uav_axis_kl, uav_dominant = role_metrics(UAV_ROLE_ID)
        metrics = {
            "edge_bin_rate": float(((selected == 0) | (selected == levels - 1)).float().mean().item()),
            "low_bin_rate": float((selected == 0).float().mean().item()),
            "high_bin_rate": float((selected == levels - 1).float().mean().item()),
            "throttle_high_rate": float((selected[:, 0] >= levels - 4).float().mean().item()),
            "surface_edge_rate": float(((selected[:, 1:] == 0) | (selected[:, 1:] == levels - 1)).float().mean().item()),
            "max_action_prob_mav": mav_prob,
            "max_action_prob_uav": uav_prob,
            "action_bin_usage_mav": mav_usage,
            "action_bin_usage_uav": uav_usage,
            "neutral_prior_probs_mav": self.policy.neutral_prior_probabilities(
                MAV_ROLE_ID
            ).cpu().tolist(),
            "neutral_prior_probs_uav": self.policy.neutral_prior_probabilities(
                UAV_ROLE_ID
            ).cpu().tolist(),
            "kl_to_neutral_mav": mav_kl,
            "kl_to_neutral_uav": uav_kl,
            "per_axis_kl_to_neutral_mav": mav_axis_kl,
            "per_axis_kl_to_neutral_uav": uav_axis_kl,
        }
        for index, name in enumerate(axis_names):
            metrics[f"dominant_bin_mav_{name}"] = mav_dominant[index]
            metrics[f"dominant_bin_uav_{name}"] = uav_dominant[index]
        return metrics

    def _prob_direction_telemetry(self, data, role_id: int, correction: torch.Tensor, advantages: torch.Tensor, prefix: str) -> dict:
        """Measure whether negative-adv actions decrease in probability after update."""
        role_mask = (data["role_ids"].view(1, -1) == role_id).float()
        active = data["agent_alive_masks"]
        valid = active * role_mask
        valid_flat = valid.reshape(-1) > 0.5
        if valid_flat.sum() < 2:
            return {}

        actions_flat = data["actions"].reshape(-1, data["actions"].shape[-1])
        adv_flat = advantages.unsqueeze(-1).expand_as(data["old_log_probs"]).reshape(-1)
        neg_mask = valid_flat & (adv_flat < 0.0)
        pos_mask = valid_flat & (adv_flat > 0.0)

        obs = data["actor_obs"]  # [T, N, D]
        T, N = obs.shape[:2]

        with torch.no_grad():
            # Re-use existing evaluate_sequence to get before-update probs
            sequences_before = [{
                "actor_obs": obs,
                "role_ids": data["role_ids"],
                "actions": data["actions"],
                "old_log_probs": data["old_log_probs"],
                "agent_alive_masks": active,
                "rnn_hidden_initial": (data.get("rnn_hidden_initial", torch.zeros(N, self.policy.rnn_hidden_size, device=obs.device))
                                       [0] if data.get("rnn_hidden_initial") is not None and data["rnn_hidden_initial"].ndim == 3
                                       else data.get("rnn_hidden_initial", torch.zeros(N, self.policy.rnn_hidden_size, device=obs.device))),
                "episode_start_masks": data.get("episode_start_masks", torch.ones(T, N, device=obs.device)),
                "active_masks": active,
                "buffer_indices": torch.arange(T, device=obs.device),
            }]
            before_seq = sequences_before[0]
            before_out = self._evaluate_sequence(before_seq)
            probs_before = before_out["action_probs"]  # [T, N, 4, 40]
            probs_before_flat = probs_before.reshape(-1, 4, self.policy.action_levels)
            # Gather selected probabilities: [B, 4, 40] indexed by actions [B, 4] → [B, 4]
            selected_before = torch.gather(
                probs_before_flat, 2, actions_flat.long().unsqueeze(-1)
            ).squeeze(-1)

        return {
            f"{prefix}_neg_adv_sample_count": float(neg_mask.sum().item()),
            f"{prefix}_pos_adv_sample_count": float(pos_mask.sum().item()),
            f"{prefix}_neg_adv_selected_prob_before_mean": float(selected_before[neg_mask].mean().item()) if neg_mask.any() else 0.0,
            f"{prefix}_pos_adv_selected_prob_before_mean": float(selected_before[pos_mask].mean().item()) if pos_mask.any() else 0.0,
            f"{prefix}_death_action_prob_before_mean": 0.0,
        }

    def _prob_direction_after(self, data, role_id, prefix, before_telemetry):
        """Capture after-update probabilities and compute deltas."""
        role_mask = (data["role_ids"].view(1, -1) == role_id).float()
        active = data["agent_alive_masks"]
        valid = active * role_mask
        valid_flat = valid.reshape(-1) > 0.5
        if valid_flat.sum() < 2:
            return {}

        actions_flat = data["actions"].reshape(-1, data["actions"].shape[-1])
        adv_flat = data["advantages"].unsqueeze(-1).expand_as(data["old_log_probs"]).reshape(-1)
        neg_mask = valid_flat & (adv_flat < 0.0)
        pos_mask = valid_flat & (adv_flat > 0.0)

        obs = data["actor_obs"]
        T, N = obs.shape[:2]

        with torch.no_grad():
            # Compute post-update probabilities (policy params already updated)
            seq_in = {
                "actor_obs": obs,
                "role_ids": data["role_ids"],
                "actions": data["actions"],
                "old_log_probs": data["old_log_probs"],
                "agent_alive_masks": active,
                "rnn_hidden_initial": (data.get("rnn_hidden_initial", torch.zeros(N, self.policy.rnn_hidden_size, device=obs.device))
                                       [0] if data.get("rnn_hidden_initial") is not None and data["rnn_hidden_initial"].ndim == 3
                                       else data.get("rnn_hidden_initial", torch.zeros(N, self.policy.rnn_hidden_size, device=obs.device))),
                "episode_start_masks": data.get("episode_start_masks", torch.ones(T, N, device=obs.device)),
                "active_masks": active,
                "buffer_indices": torch.arange(T, device=obs.device),
            }
            after_out = self._evaluate_sequence(seq_in)
            probs_after = after_out["action_probs"].reshape(-1, 4, self.policy.action_levels)
            selected_after = torch.gather(
                probs_after, 2, actions_flat.long().unsqueeze(-1)
            ).squeeze(-1)

        neg_before = before_telemetry.get(f"{prefix}_neg_adv_selected_prob_before_mean", 0.0)
        neg_after = float(selected_after[neg_mask].mean().item()) if neg_mask.any() else 0.0
        pos_before = before_telemetry.get(f"{prefix}_pos_adv_selected_prob_before_mean", 0.0)
        pos_after = float(selected_after[pos_mask].mean().item()) if pos_mask.any() else 0.0

        return {
            f"{prefix}_neg_adv_selected_prob_after_mean": neg_after,
            f"{prefix}_neg_adv_selected_prob_delta_mean": neg_after - neg_before,
            f"{prefix}_pos_adv_selected_prob_after_mean": pos_after,
            f"{prefix}_pos_adv_selected_prob_delta_mean": pos_after - pos_before,
            f"{prefix}_death_action_prob_after_mean": 0.0,
            f"{prefix}_death_action_prob_delta_mean": 0.0,
        }

    def update(self, buffer):
        device = next(self.policy.parameters()).device
        data = buffer.get(device)
        sequences = buffer.get_sequences(device)
        advantages, returns = self._advantages_and_returns(data)
        data["advantages"] = advantages.detach()
        role_stats = {MAV_ROLE_ID: [], UAV_ROLE_ID: []}
        kl_stopped_roles = set()
        critic_losses, critic_grad_norms, correction_values = [], [], []

        # Prob-direction telemetry BEFORE update
        prob_before_mav = self._prob_direction_telemetry(
            data, MAV_ROLE_ID, torch.ones_like(data["old_log_probs"]),
            data["advantages"], "mav")
        prob_before_uav = self._prob_direction_telemetry(
            data, UAV_ROLE_ID, torch.ones_like(data["old_log_probs"]),
            data["advantages"], "uav")

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
            if self.happo_update_granularity == "agent":
                # Agent-sequential: iterate over individual agents (red_ids order)
                agent_stats = {}
                for agent_index, agent_id in enumerate(self.agent_ids):
                    role_id = MAV_ROLE_ID if agent_index == 0 else UAV_ROLE_ID
                    optimizer = self.mav_opt if agent_index == 0 else self.uav_opt
                    params = self.mav_params if agent_index == 0 else self.uav_params
                    if agent_id not in agent_stats:
                        agent_stats[agent_id] = []
                    agent_stats[agent_id].append(self._update_role(
                        sequences, advantages, correction, role_id, optimizer, params,
                        force_skip_by_kl=role_id in kl_stopped_roles,
                        agent_index=agent_index,
                    ))
                    if agent_stats[agent_id][-1][7] > 0.0:
                        kl_stopped_roles.add(role_id)
                for role_id in (MAV_ROLE_ID, UAV_ROLE_ID):
                    role_stats[role_id] = []
            else:
                # Role-level: iterate MAV then UAV
                for role_id, optimizer, params in (
                    (MAV_ROLE_ID, self.mav_opt, self.mav_params),
                    (UAV_ROLE_ID, self.uav_opt, self.uav_params),
                ):
                    role_stats[role_id].append(self._update_role(
                        sequences, advantages, correction, role_id, optimizer, params,
                        force_skip_by_kl=role_id in kl_stopped_roles,
                    ))
                    if role_stats[role_id][-1][7] > 0.0:
                        kl_stopped_roles.add(role_id)
            if not torch.isfinite(correction).all():
                raise ValueError("non-finite HAPPO correction factor")
            correction_values.append(correction.detach().reshape(-1))

        for name, parameter in self.policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"non-finite policy parameter after update: {name}")

        # Prob-direction telemetry AFTER update
        prob_after_mav = self._prob_direction_after(data, MAV_ROLE_ID, "mav", prob_before_mav)
        prob_after_uav = self._prob_direction_after(data, UAV_ROLE_ID, "uav", prob_before_uav)

        # In agent mode, aggregate per-agent stats into MAV/UAV role stats
        if self.happo_update_granularity == "agent":
            agent_ids_in_stats = set(self.agent_ids) & set(agent_stats.keys())
            for agent_id in agent_ids_in_stats:
                agent_index = self.agent_ids.index(agent_id)
                role_id = MAV_ROLE_ID if agent_index == 0 else UAV_ROLE_ID
                role_stats[role_id].extend(agent_stats[agent_id])

        mav = np.asarray(role_stats[MAV_ROLE_ID], dtype=np.float64)
        uav = np.asarray(role_stats[UAV_ROLE_ID], dtype=np.float64)
        correction = torch.cat(correction_values)
        mav_entropy = float(mav[:, 1].mean()) if len(mav) else 0.0
        uav_entropy = float(uav[:, 1].mean()) if len(uav) else 0.0
        mav_actor_loss = float(mav[:, 0].mean()) if len(mav) else 0.0
        uav_actor_loss = float(uav[:, 0].mean()) if len(uav) else 0.0
        red0_active = data["active_masks"][:, 0] > 0.5
        red0_advantages = advantages[red0_active]
        red0_deaths = data.get(
            "death_transition_masks", torch.zeros_like(data["active_masks"])
        )[:, 0] > 0.5

        def _finite_ratio(numerator, denominator):
            return float(numerator / max(abs(denominator), 1e-8))

        metrics = {
            "happo_update_granularity": self.happo_update_granularity,
            "agent_update_order": self.agent_ids if self.happo_update_granularity == "agent" else "role_level",
            "actor_loss_mav": mav_actor_loss,
            "actor_loss_uav": uav_actor_loss,
            "critic_loss": float(np.mean(critic_losses)),
            "entropy_mav": mav_entropy,
            "entropy_uav": uav_entropy,
            "entropy_mav_raw": mav_entropy,
            "entropy_uav_raw": uav_entropy,
            "entropy_mav_per_axis_mean": mav_entropy / 4.0,
            "entropy_uav_per_axis_mean": uav_entropy / 4.0,
            "entropy_bonus_mav": self.role_entropy_coef[MAV_ROLE_ID] * mav_entropy,
            "entropy_bonus_uav": self.role_entropy_coef[UAV_ROLE_ID] * uav_entropy,
            "actor_surrogate_loss_mav_abs": abs(mav_actor_loss),
            "actor_surrogate_loss_uav_abs": abs(uav_actor_loss),
            "entropy_to_policy_loss_ratio_mav": _finite_ratio(
                self.role_entropy_coef[MAV_ROLE_ID] * mav_entropy, mav_actor_loss
            ),
            "entropy_to_policy_loss_ratio_uav": _finite_ratio(
                self.role_entropy_coef[UAV_ROLE_ID] * uav_entropy, uav_actor_loss
            ),
            "advantage_mean_red_0": float(red0_advantages.mean().item()) if red0_advantages.numel() else 0.0,
            "advantage_std_red_0": float(red0_advantages.std(unbiased=False).item()) if red0_advantages.numel() else 0.0,
            "advantage_min_red_0": float(red0_advantages.min().item()) if red0_advantages.numel() else 0.0,
            "advantage_max_red_0": float(red0_advantages.max().item()) if red0_advantages.numel() else 0.0,
            "active_sample_count_red_0": float(red0_active.sum().item()),
            "death_transition_count_red_0": float(red0_deaths.sum().item()),
            "death_transition_used_for_actor_red_0": float(
                (red0_deaths & red0_active).sum().item()
            ),
            "approx_kl_mav": float(mav[:, 2].mean()),
            "approx_kl_uav": float(uav[:, 2].mean()),
            "mav_actor_lr_effective": self.mav_actor_lr_effective,
            "uav_actor_lr_effective": self.uav_actor_lr_effective,
            "mav_entropy_coef_effective": self.role_entropy_coef[MAV_ROLE_ID],
            "uav_entropy_coef_effective": self.role_entropy_coef[UAV_ROLE_ID],
            "mav_clip_param_effective": self.role_clip_param[MAV_ROLE_ID],
            "uav_clip_param_effective": self.role_clip_param[UAV_ROLE_ID],
            "mav_target_kl": self.role_target_kl[MAV_ROLE_ID],
            "uav_target_kl": self.role_target_kl[UAV_ROLE_ID],
            "mav_active_sample_count": float(mav[:, 3].mean()),
            "uav_active_sample_count": float(uav[:, 3].mean()),
            "grad_norm_actor": float(np.mean(np.concatenate([mav[:, 4], uav[:, 4]]))),
            "grad_norm_shared": float(np.mean(np.concatenate([mav[:, 5], uav[:, 5]]))),
            "grad_norm_mav_head": float(mav[:, 6].mean()),
            "grad_norm_uav_head": float(uav[:, 6].mean()),
            "mav_shared_update_mode": self.mav_shared_update_mode,
            "mav_shared_step_enabled": int(
                self.shared_actor_opt is not None
                and self.mav_shared_update_mode == "full"
            ),
            "uav_shared_step_enabled": int(self.shared_actor_opt is not None),
            "mav_shared_grad_norm_before_clear": float(mav[:, 5].mean()),
            "grad_norm_shared_from_mav": float(mav[:, 5].mean()),
            "grad_norm_shared_from_uav": float(uav[:, 5].mean()),
            "mav_kl_early_stop_count": int(mav[:, 7].sum()),
            "uav_kl_early_stop_count": int(uav[:, 7].sum()),
            "mav_update_skipped_by_kl": int(mav[:, 8].sum()),
            "uav_update_skipped_by_kl": int(uav[:, 8].sum()),
            "grad_norm_critic": float(np.mean(critic_grad_norms)),
            "correction_factor_mean": float(correction.mean().item()),
            "correction_factor_max": float(correction.max().item()),
            "correction_factor_min": float(correction.min().item()),
        }
        metrics.update(prob_before_mav)
        metrics.update(prob_before_uav)
        metrics.update(prob_after_mav)
        metrics.update(prob_after_uav)
        metrics.update(self._distribution_metrics(sequences))
        # Agent-level metrics
        if self.happo_update_granularity == "agent":
            for agent_id in self.agent_ids:
                if agent_id in agent_stats and agent_stats[agent_id]:
                    arr = np.asarray(agent_stats[agent_id], dtype=np.float64)
                    metrics[f"agent_loss_{agent_id}"] = float(arr[:, 0].mean())
                    metrics[f"agent_entropy_{agent_id}"] = float(arr[:, 1].mean())
                    metrics[f"agent_approx_kl_{agent_id}"] = float(arr[:, 2].mean())
                    metrics[f"agent_active_sample_count_{agent_id}"] = float(arr[:, 3].mean())
            metrics["shared_step_count"] = int(
                len([a for a in agent_stats.values() if len(a) > 0]))
            metrics["mav_head_step_count"] = int(
                len(agent_stats.get(self.agent_ids[0], [])) if self.agent_ids else 0)
            metrics["uav_head_step_count"] = int(
                sum(len(agent_stats.get(aid, [])) for aid in self.agent_ids[1:]) if len(self.agent_ids) > 1 else 0)
        return metrics
