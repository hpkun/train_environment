"""Read-only audit for the current MAPPO baseline implementation.

The audit checks implementation logic against plain shared-policy MAPPO /
BRMA-MAPPO baseline expectations. It reports issues only; it does not change
the environment, reward, missile logic, PID, aircraft XML, or algorithm code.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.policy import MAPPOActorCritic
from algorithms.mappo.trainer import PPOTrainer
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


FILES = {
    "policy": ROOT / "algorithms" / "mappo" / "policy.py",
    "trainer": ROOT / "algorithms" / "mappo" / "trainer.py",
    "storage": ROOT / "algorithms" / "mappo" / "storage.py",
    "utils": ROOT / "algorithms" / "mappo" / "utils.py",
    "train": ROOT / "scripts" / "train_mappo_baseline.py",
    "eval_baseline": ROOT / "scripts" / "eval_mappo_baseline.py",
    "eval_zero_shot": ROOT / "scripts" / "eval_mappo_zero_shot.py",
    "adapter_v2": ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _linear_layers(module) -> list[int]:
    return [
        layer.out_features
        for layer in module
        if layer.__class__.__name__ == "Linear"
    ]


def _source_has(path_key: str, pattern: str) -> bool:
    return re.search(pattern, _read(FILES[path_key]), flags=re.MULTILINE) is not None


def _latest_action_saturation_hint() -> dict[str, Any]:
    """Read known training logs if present; absence is not an audit failure."""
    candidates = [
        ROOT / "outputs" / "main_mappo_experiment_f22_200k_rule_nearest_online_eval" / "train_log.csv",
        ROOT / "outputs" / "main_mappo_experiment_f22_500k_rule_nearest" / "train_log.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split(",")
        last = lines[-1].split(",")
        row = dict(zip(header, last))
        if "action_saturation_rate" in row:
            try:
                return {
                    "source": str(path.relative_to(ROOT)),
                    "last_action_saturation_rate": float(row["action_saturation_rate"]),
                    "last_action_mean_abs": float(row.get("action_mean_abs", 0.0)),
                    "high_action_saturation_observed": float(row["action_saturation_rate"]) >= 0.4,
                }
            except ValueError:
                pass
        try:
            return {
                "source": str(path.relative_to(ROOT)),
                "last_action_mean_abs": float(row.get("action_mean_abs", 0.0)),
                "high_action_saturation_observed": float(row.get("action_mean_abs", 0.0)) >= 0.8,
            }
        except ValueError:
            continue
    return {
        "source": None,
        "high_action_saturation_observed": False,
        "note": "No recent train_log.csv with action saturation fields found.",
    }


def build_audit() -> dict[str, Any]:
    adapter = HeteroObsAdapterV2()
    model = MAPPOActorCritic(
        actor_obs_dim=adapter.flat_actor_obs_dim,
        critic_state_dim=adapter.critic_state_dim,
        action_dim=3,
    )
    trainer = PPOTrainer(model)

    policy_src = _read(FILES["policy"])
    trainer_src = _read(FILES["trainer"])
    train_src = _read(FILES["train"])
    eval_zero_src = _read(FILES["eval_zero_shot"])
    adapter_src = _read(FILES["adapter_v2"])

    actor_mlp = [model.actor_obs_dim] + _linear_layers(model.actor)
    critic_mlp = [model.critic_state_dim] + _linear_layers(model.critic)

    uses_normal = "torch.distributions.Normal" in policy_src
    samples_and_clamps = bool(re.search(r"dist\.sample\(\)\.clamp", policy_src))
    logprob_on_action = "dist.log_prob(action)" in policy_src
    evaluate_logprob_on_actions = "dist.log_prob(actions)" in policy_src

    red_valid_slot_valid = (
        "red_valid_mask[:min(len(red_ids), self.max_red)] = 1.0" in adapter_src
    )
    train_stores_red_valid = (
        "red_valid_np = result['red_valid_mask']" in train_src
        and "buffer.store(actor_obs_np, critic_state_np, action_np, log_prob_np" in train_src
    )
    trainer_uses_red_valid_mask = (
        "valid = red_valid[:, :num_red]" in trainer_src
        and "policy_loss = (policy_loss * valid).sum()" in trainer_src
    )
    dead_agent_policy_loss_issue = (
        red_valid_slot_valid and train_stores_red_valid and trainer_uses_red_valid_mask
    )

    per_agent_dones = "dones_np = np.array" in train_src and "for rid in env.red_ids" in train_src
    single_agent_done_truncates_gae = "team_dones = (dones.sum(dim=-1) > 0).float()" in trainer_src

    team_reward_uses_red_valid = (
        "team_reward = (rewards * red_valid[:, :num_red]).sum(dim=-1) / valid_count"
        in trainer_src
    )
    value_loss_team = "F.mse_loss(new_values, returns)" in trainer_src

    critic_concat_red_actor_obs = (
        "self.critic_state_dim = self.flat_actor_obs_dim * self.max_red" in adapter_src
        and "critic_parts.append(actor_obs[red_ids[i]])" in adapter_src
    )

    train_stochastic = "deterministic=False" in train_src
    eval_deterministic = "deterministic=True" in eval_zero_src
    obs_adapter_meta_validation = (
        "resolve_obs_adapter_version" in eval_zero_src
        and "validate_model_dims" in eval_zero_src
    )

    saturation_hint = _latest_action_saturation_hint()

    blocking_issues: list[str] = []
    warnings: list[str] = []
    if model.actor_obs_dim != 96 or model.critic_state_dim != 480:
        blocking_issues.append("actor/critic dim with V2 is inconsistent")
    if model.action_dim != 3:
        blocking_issues.append("action_dim is not 3")
    if dead_agent_policy_loss_issue:
        blocking_issues.append("dead_agents_may_contribute_to_policy_loss")
    if single_agent_done_truncates_gae:
        blocking_issues.append("single_agent_death_may_truncate_team_gae")
    if not obs_adapter_meta_validation:
        blocking_issues.append("eval/train obs_adapter consistency is not validated")

    if samples_and_clamps and logprob_on_action:
        warnings.append("clipped_gaussian_logprob_mismatch_risk")
    warnings.extend([
        "no attention / no temporal / no HAPPO",
        "critic is simple concatenation rather than attention/global state",
        "MAPPO baseline may be insufficient for heterogeneous zero-shot transfer",
    ])
    if saturation_hint.get("high_action_saturation_observed"):
        warnings.append("high action saturation risk")

    network_architecture = {
        "actor_obs_dim": model.actor_obs_dim,
        "critic_state_dim": model.critic_state_dim,
        "action_dim": model.action_dim,
        "actor_mlp": actor_mlp,
        "critic_mlp": critic_mlp,
        "uses_gaussian_policy": uses_normal,
        "learnable_log_std": hasattr(model, "action_log_std"),
        "no_attention": not any(k in policy_src.lower() for k in ["attention", "transformer"]),
        "no_gru": "gru" not in policy_src.lower(),
        "no_temporal_feature": "temporal" not in policy_src.lower(),
        "no_happo_sequential_update": "happo" not in trainer_src.lower(),
        "conclusion": "MAPPO baseline only; not TAM-HAPPO temporal attention method.",
    }

    audit = {
        "summary": {
            "audit_passed": len(blocking_issues) == 0,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "recommended_next_fix": (
                "Fix alive mask first, then team done / GAE, then clipped Gaussian log_prob."
                if blocking_issues else
                "Baseline implementation has no blocking issue from this static audit."
            ),
            "baseline_status": "not_ready" if blocking_issues else "usable_after_fixes",
        },
        "network_architecture": network_architecture,
        "action_distribution_and_logprob": {
            "action_sampled_from_normal": uses_normal,
            "sample_then_clamp": samples_and_clamps,
            "log_prob_computed_on_clamped_action": samples_and_clamps and logprob_on_action,
            "evaluate_actions_uses_stored_actions": evaluate_logprob_on_actions,
            "action_saturation_hint": saturation_hint,
            "warning": (
                "clipped_gaussian_logprob_mismatch_risk"
                if samples_and_clamps and logprob_on_action else None
            ),
        },
        "rollout_mask_logic": {
            "adapter_red_valid_mask_semantics": "slot-valid, not alive-valid",
            "train_passes_red_valid_mask_to_buffer": train_stores_red_valid,
            "trainer_policy_loss_mask": "red_valid[:, :num_red]",
            "dead_agents_may_contribute_to_policy_loss": dead_agent_policy_loss_issue,
        },
        "done_and_gae_logic": {
            "train_dones_np_semantics": "per-agent terminated or truncated for red_ids",
            "per_agent_dones_detected": per_agent_dones,
            "trainer_team_dones_expression": "team_dones = (dones.sum(dim=-1) > 0).float()",
            "single_agent_death_may_truncate_team_gae": single_agent_done_truncates_gae,
            "centralized_critic_should_use_episode_done": True,
        },
        "reward_and_value_target_logic": {
            "team_reward_uses_valid_agents_mean": team_reward_uses_red_valid,
            "valid_mask_is_alive_mask": False,
            "dead_agent_rewards_may_enter_team_reward": team_reward_uses_red_valid,
            "critic_value_shape": "scalar team value per timestep",
            "value_loss_team_value_only": value_loss_team,
        },
        "centralized_critic_input": {
            "critic_state_dim": adapter.critic_state_dim,
            "critic_state_is_red_actor_obs_concat": critic_concat_red_actor_obs,
            "padding_fixed": True,
            "contains_only_red_actor_perspective_concat": critic_concat_red_actor_obs,
            "ctde_baseline_status": "simple centralized critic",
            "opponent_global_info_missing_risk": True,
            "not_attention_critic": True,
        },
        "advantage_and_ppo_update": {
            "gamma": trainer.gamma,
            "gae_lambda": trainer.gae_lambda,
            "advantage_normalization": "advantages are normalized when numel > 1",
            "ppo_ratio": "exp(new_log_prob - old_log_probs)",
            "clip_param": trainer.clip_param,
            "entropy_bonus_sign": "loss = policy + value + entropy_coef * negative_entropy",
            "grad_clipping": True,
            "max_grad_norm": trainer.max_grad_norm,
            "ppo_epochs": trainer.ppo_epochs,
            "actor_lr": trainer.actor_opt.param_groups[0]["lr"],
            "critic_lr": trainer.critic_opt.param_groups[0]["lr"],
            "max_grad_norm_large_risk": trainer.max_grad_norm >= 10.0,
        },
        "multiagent_parameter_sharing": {
            "red_mav_and_uav_share_actor": True,
            "same_actor_obs_dim": model.actor_obs_dim,
            "role_one_hot_present_in_v2_ego_feature": True,
            "fits_mappo_baseline": True,
            "not_happo_heterogeneous_sequential_policy": True,
        },
        "evaluation_consistency": {
            "train_uses_stochastic_action": train_stochastic,
            "eval_uses_deterministic_action": eval_deterministic,
            "obs_adapter_version_resolved_from_cli_or_meta": "resolve_obs_adapter_version" in eval_zero_src,
            "actor_critic_dim_validation": obs_adapter_meta_validation,
            "latest_checkpoint_meta_exists_in_training": "latest/meta.json" in train_src,
            "best_checkpoint_tracking_exists": "best/model.pt" in train_src,
        },
        "paper_alignment": {
            "brma_mappo_baseline": (
                "Aligned as a plain shared-policy MAPPO baseline with centralized critic."
            ),
            "tam_happo_attention_temporal": (
                "Not aligned: no temporal feature, no attention, no GRU, and no HAPPO "
                "sequential update."
            ),
            "next_after_baseline_fixes": (
                "If baseline still fails after alive mask / team done / log_prob fixes, "
                "prioritize entity attention rather than more training steps."
            ),
        },
    }
    return audit


def write_markdown(audit: dict[str, Any], output_md: Path) -> None:
    summary = audit["summary"]
    md = f"""# MAPPO Baseline Implementation Audit

## Purpose

This audit checks whether the current MAPPO baseline training logic is suitable
as a plain MAPPO baseline for the main F-22 MAV experiments. It does not modify
the environment or the algorithm.

## Baseline Scope

The current implementation is a MAPPO baseline related to BRMA-MAPPO in the
sense that it uses a shared actor, centralized critic, GAE, and PPO clipping.

It is not the final method for the heterogeneous TAM-HAPPO paper. It has no
temporal feature module, no attention module, no GRU, and no HAPPO sequential
agent update.

## Blocking Issues

{chr(10).join(f"- {issue}" for issue in summary["blocking_issues"]) or "- None"}

## Warnings

{chr(10).join(f"- {warning}" for warning in summary["warnings"]) or "- None"}

## Key Findings

- The V2 actor dim is {audit['network_architecture']['actor_obs_dim']} and critic
  dim is {audit['network_architecture']['critic_state_dim']}.
- The actor is an MLP {audit['network_architecture']['actor_mlp']}.
- The critic is an MLP {audit['network_architecture']['critic_mlp']}.
- The current PPO mask uses slot-valid red masks, not an alive mask.
- The current team done logic can let one red agent death truncate team done /
  GAE.
- The action path has clipped Gaussian log_prob mismatch risk:
  sample Normal action, clamp it, then compute log_prob on the clipped action.

## Recommended Fix Order

1. alive mask
2. team done / GAE
3. action distribution / log_prob
4. if the MAPPO baseline still fails, then consider entity attention

This audit is not the final method module.
"""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/mappo_baseline_audit/mappo_baseline_implementation_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/mappo_baseline_audit/mappo_baseline_implementation_audit.md",
    )
    args = parser.parse_args()

    audit = build_audit()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit, output_md)

    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"blocking_issues: {audit['summary']['blocking_issues']}", flush=True)
    print(f"warnings: {audit['summary']['warnings']}", flush=True)


if __name__ == "__main__":
    main()
