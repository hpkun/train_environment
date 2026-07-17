"""Formal independent-actor vanilla HAPPO trainer for tam_paper_env_v1."""

from __future__ import annotations

import argparse, csv, importlib.metadata, json, platform, random, sys, warnings
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo import (ParameterSharingPPOTrainer,
                                            VanillaHAPPORolloutBuffer,
                                            VanillaHAPPOTrainer)
from algorithms.happo.vanilla_happo_checkpoint import (load_vanilla_happo_checkpoint,
                                                       save_vanilla_happo_checkpoint)
from scripts.vanilla_happo_runtime import (deterministic_evaluate, flattened_obs,
                                           infer_policy, make_paper_env, seed_all)
from scripts.tam_output_paths import resolve_tam_output


def is_episode_boundary(terminated, truncated):
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated shapes must match")
    return bool(np.logical_or(terminated, truncated).all())


def resume_seed_state(requested_seed, episodes, loaded):
    if loaded.get("resume_semantics") == "episode_boundary":
        schedule = loaded.get("seed_schedule") or {}
        if "episode_seed_base" not in schedule:
            raise ValueError("strict resume checkpoint is missing episode seed schedule")
        base = int(schedule["episode_seed_base"])
        next_seed = int(schedule.get("next_episode_seed", base + int(episodes)))
        return base, next_seed
    base = int(requested_seed)
    return base, base + int(episodes)


def agent_update_contract(active_count, actor_changed, critic_head_changed,
                          actor_zero_gradient=False,
                          critic_head_zero_gradient=False):
    expected = int(active_count) > 0
    if not expected:
        return False, bool(not actor_changed and not critic_head_changed)
    actor_valid = actor_changed or actor_zero_gradient
    critic_valid = critic_head_changed or critic_head_zero_gradient
    return expected, bool(actor_valid and critic_valid)


def should_run_evaluation(interval, episodes, steps, total_steps):
    return bool(interval > 0 and episodes > 0
                and (steps % interval == 0 or steps == total_steps))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=("2v2", "3v2", "5v4"), default="2v2")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--total-environment-steps", type=int, default=5000)
    p.add_argument("--rollout-length", type=int, default=256)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-gradient-norm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--evaluation-interval", type=int, default=1000)
    p.add_argument("--evaluation-episodes", type=int, default=2)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--output-directory", default="outputs/tam_paper_vanilla_happo")
    p.add_argument("--resume-checkpoint")
    p.add_argument("--allow-episode-restart-resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--actor-sharing", choices=("independent",
                                                "parameter_sharing_ppo_ablation",
                                                "role_shared_ablation"),
                   default="independent")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--evaluation-seed-base", type=int, default=None)
    p.add_argument("--evaluation-perturbation", choices=("none","low","medium","large"),
                   default="none",
                   help=("none: nominal Table 5-7 evaluation; "
                         "low/medium/large: 5v4 generalization evaluation"))
    return p.parse_args(argv)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    output = resolve_tam_output(ROOT, args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    env = make_paper_env(ROOT, args.scenario)
    policy, obs_dim, state_dim = infer_policy(
        env, args.actor_sharing, args.hidden_dim, device)
    if args.actor_sharing == "role_shared_ablation":
        warnings.warn("role_shared_ablation is a legacy alias for "
                      "parameter_sharing_ppo_ablation", FutureWarning)
    if policy.actor_sharing == "independent":
        trainer = VanillaHAPPOTrainer(
            policy, args.actor_lr, args.critic_lr, args.clip_param,
            args.value_loss_coef, args.entropy_coef, args.max_gradient_norm,
            args.ppo_epochs, args.minibatch_size, args.gamma, args.gae_lambda,
            seed=args.seed)
    else:
        trainer = ParameterSharingPPOTrainer(
            policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            clip_param=args.clip_param, value_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef, max_grad_norm=args.max_gradient_norm,
            ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
            gamma=args.gamma, gae_lambda=args.gae_lambda, seed=args.seed)
    steps = episodes = policy_version = 0
    episode_seed_base = int(args.seed)
    next_episode_seed = episode_seed_base
    evaluation_seed_base = (args.evaluation_seed_base if args.evaluation_seed_base is not None
                            else episode_seed_base + 100000)
    resumed_from_semantics = None
    if args.resume_checkpoint:
        loaded = load_vanilla_happo_checkpoint(
            resolve_tam_output(ROOT, args.resume_checkpoint), policy, trainer,
            numpy_rng=rng, for_resume=True,
            allow_episode_restart=args.allow_episode_restart_resume,
            expected_scenario=args.scenario)
        steps, episodes = loaded["environment_steps"], loaded["episodes"]
        policy_version = trainer.update_count
        resumed_from_semantics = loaded["resume_semantics"]
        episode_seed_base, next_episode_seed = resume_seed_state(
            args.seed, episodes, loaded)
        if args.evaluation_seed_base is None:
            evaluation_seed_base = episode_seed_base + 100000
    try:
        jsbsim_version = importlib.metadata.version("jsbsim")
    except importlib.metadata.PackageNotFoundError:
        jsbsim_version = None
    snapshot = vars(args) | {
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "jsbsim_version": jsbsim_version, "actual_device": str(device),
        "dynamics_backend": "jsbsim", "controlled_side": "red",
        "agent_ids": env.agent_ids, "num_agents": env.num_agents,
        "actor_observation_dim": obs_dim, "critic_state_dim": state_dim,
        "action_space": "MultiDiscrete([40,40,40,40])",
        "total_parameters": sum(p.numel() for p in policy.parameters()),
        "actor_parameters": {aid: sum(p.numel() for p in policy.actors[policy.actor_key(aid)].parameters())
                             for aid in env.agent_ids},
        "critic_parameters": sum(p.numel() for p in policy.critic.parameters()),
        "uses_tam": False, "uses_recurrence": False, "uses_attention": False,
        "algorithm_mode": trainer.algorithm_mode,
        "reward_semantics": "heterogeneous_per_agent",
        "theoretical_team_reward_monotonic_guarantee_claimed": False,
        "resumed_from_semantics": resumed_from_semantics,
        "requested_seed": int(args.seed),
        "restored_seed": episode_seed_base,
        "next_episode_reset_seed": next_episode_seed,
        "evaluation_seed_base": int(evaluation_seed_base),
        "evaluation_perturbation": args.evaluation_perturbation,
        "evaluation_protocol": (
            "fixed_seed_nominal" if args.evaluation_perturbation == "none"
            else f"fixed_seed_{args.evaluation_perturbation}_generalization"),
        "actor_sharing_label": ("formal_independent" if args.actor_sharing == "independent"
                                else "parameter_sharing_ablation"),
    }
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output / "tensorboard")
        snapshot["tensorboard_writer"] = "torch.utils.tensorboard"
    except Exception:
        from scripts.tensorboard_fallback import FallbackSummaryWriter
        writer = FallbackSummaryWriter(output / "tensorboard")
        snapshot["tensorboard_writer"] = "dependency_free_tfrecord_fallback"
    (output / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    csv_path = output / "training.csv"
    obs, _ = env.reset(seed=next_episode_seed)
    episode_return = np.zeros(env.num_agents)
    episode_id = episodes
    rows, latest_eval = [], None
    at_episode_boundary = False
    while (steps < args.total_environment_steps
           and (args.max_updates is None or trainer.update_count < args.max_updates)):
        horizon = min(args.rollout_length, args.total_environment_steps - steps)
        for interval in (args.evaluation_interval, args.checkpoint_interval):
            if interval:
                horizon = min(horizon, interval - steps % interval)
        buffer = VanillaHAPPORolloutBuffer(horizon, env.num_agents, obs_dim, state_dim)
        episode_returns, winners, episode_lengths = [], [], []
        launches, hits, structural, boundary, survivors, terminal_agents = 0, 0, 0, 0, 0, 0
        target_violations = nonfinite = 0
        rollout_ended_at_episode_boundary = False
        reward_component_sums, reward_component_count = {}, 0
        for local_step in range(horizon):
            rollout_ended_at_episode_boundary = False
            actor_obs, state = flattened_obs(env, obs), env.get_state()
            available_dict = env.get_avail_actions()
            available = np.stack([available_dict[aid] for aid in env.agent_ids])
            alive_start = np.array([float(next(a for a in env.task.agents if a.agent_id == aid).alive)
                                    for aid in env.agent_ids], np.float32)
            with torch.no_grad():
                action_out = policy.act(actor_obs, state, available)
            actions = action_out["actions"].cpu().numpy()
            next_obs, rewards, term, trunc, info = env.step(
                {aid: actions[i] for i, aid in enumerate(env.agent_ids)})
            reward_array = np.array([rewards[aid] for aid in env.agent_ids], np.float32)
            alive_end = info["alive_at_step_end"]
            global_truncated = all(trunc.values())
            global_terminated = all(term.values()) and not global_truncated
            terminated = np.array([
                float(not alive_end[aid] or global_terminated) for aid in env.agent_ids])
            truncated = np.array([
                float(global_truncated and alive_end[aid]) for aid in env.agent_ids])
            with torch.no_grad():
                next_value = policy.value(torch.as_tensor(env.get_state(), dtype=torch.float32,
                                                           device=device)).cpu().numpy()
            buffer.add(obs=actor_obs, state=state, actions=actions,
                       log_probs=action_out["log_probs"].cpu().numpy(), rewards=reward_array,
                       value=action_out["value"].cpu().numpy(), next_value=next_value,
                       terminated=terminated, truncated=truncated,
                       active_masks=alive_start, available_actions=available,
                       agent_alive=alive_start, episode_id=episode_id,
                       decision_step=info["episode_step"], policy_version=policy_version)
            episode_return += reward_array; steps += 1; obs = next_obs
            target_violations += len(info["target_consistency_violation"])
            nonfinite += int(not np.isfinite(reward_array).all())
            for components in info["reward_components"].values():
                for key, value in components.items():
                    reward_component_sums[key] = reward_component_sums.get(key, 0.0) + float(value)
                reward_component_count += 1
            episode_done = all(bool(term[aid] or trunc[aid]) for aid in env.agent_ids)
            if episode_done:
                rollout_ended_at_episode_boundary = True
                episode_returns.append(episode_return.copy()); winners.append(info["winner"])
                episode_lengths.append(info["episode_step"])
                launches += info["missiles_fired"]; hits += info["missile_hits"]
                structural += info["structural_failures"]; boundary += info["out_of_zone"]
                survivors += sum(info["alive_at_step_end"].values())
                terminal_agents += env.num_agents
                episodes += 1; episode_id += 1; episode_return.fill(0)
                next_episode_seed = episode_seed_base + episodes
                obs, _ = env.reset(seed=next_episode_seed)
                # Continue collecting: do NOT break the rollout loop
        if buffer.pos != horizon:
            raise RuntimeError(
                f"buffer.pos ({buffer.pos}) != planned_horizon ({horizon}) "
                f"at update {trainer.update_count}")
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}
        result = trainer.update(buffer); policy_version += 1
        unchanged = [name for name, value in policy.named_parameters()
                     if torch.equal(before[name], value)]
        actor_changed = {
            aid: any(not torch.equal(
                before[f"actors.{policy.actor_key(aid)}.{name}"], value)
                for name, value in policy.actors[policy.actor_key(aid)].named_parameters())
            for aid in policy.agent_ids}
        critic_changed = any(not torch.equal(before[f"critic.{name}"], value)
                             for name, value in policy.critic.named_parameters())
        critic_head_changed = {
            aid: any(not torch.equal(
                before[f"critic.heads.{aid}.{name}"], value)
                for name, value in policy.critic.heads[aid].named_parameters())
            for aid in policy.agent_ids}
        update_expected, update_contract_valid = {}, {}
        for index, aid in enumerate(policy.agent_ids):
            active_count = result.metrics.get(
                f"active_sample_count/{aid}",
                int(buffer.active_masks[:buffer.pos, index].sum()))
            update_expected[aid], update_contract_valid[aid] = agent_update_contract(
                active_count, actor_changed[aid], critic_head_changed[aid],
                result.metrics.get(f"actor_zero_gradient/{aid}", False),
                result.metrics.get(f"critic_head_zero_gradient/{aid}", False))
        actions_all = buffer.actions[:buffer.pos]
        row = {"environment_steps": steps, "episodes": episodes,
               "mean_episode_return": float(np.mean([x.sum() for x in episode_returns])) if episode_returns else 0.0,
               "mav_return": float(np.mean([x[0] for x in episode_returns])) if episode_returns else 0.0,
               "uav_return": float(np.mean([x[1:].mean() for x in episode_returns])) if episode_returns else 0.0,
               "win_rate": float(np.mean([x == "red" for x in winners])) if winners else 0.0,
               "draw_rate": float(np.mean([x == "draw" for x in winners])) if winners else 0.0,
               "episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
               "launch_rate": launches/max(len(winners),1), "hit_rate": hits/max(launches,1),
               "survival_rate": survivors/max(terminal_agents,1),
               "structural_failure_rate": structural/max(len(winners),1),
               "boundary_rate": boundary/max(len(winners),1),
               "target_consistency_violation": target_violations, "nan_inf_count": nonfinite,
               "unchanged_parameter_count": len(unchanged),
               "actor_changed": json.dumps(actor_changed),
               "all_actors_changed": all(actor_changed.values()),
               "critic_changed": critic_changed,
               "critic_head_changed": json.dumps(critic_head_changed),
               "all_critic_heads_changed": all(critic_head_changed.values()),
               "update_expected": json.dumps(update_expected),
               "update_contract_valid": json.dumps(update_contract_valid),
               "optimization_update_contract_valid": all(update_contract_valid.values()),
               "rollout_planned_horizon": horizon,
               "rollout_collected_steps": buffer.pos,
               "rollout_episode_count": len(winners),
               "rollout_ended_at_episode_boundary": rollout_ended_at_episode_boundary,
               "minimum_active_sample_count": min(
                   result.metrics.get(f"active_sample_count/{aid}", 0) for aid in policy.agent_ids),
               "maximum_approx_kl": max(
                   result.metrics.get(f"approx_kl/{aid}", 0.0) for aid in policy.agent_ids),
               } | result.metrics
        for aid in policy.agent_ids:
            row[f"update_expected/{aid}"] = update_expected[aid]
            row[f"actor_changed/{aid}"] = actor_changed[aid]
            row[f"critic_head_changed/{aid}"] = critic_head_changed[aid]
            row[f"update_contract_valid/{aid}"] = update_contract_valid[aid]
        for key, value in reward_component_sums.items():
            row[f"reward_component/{key}"] = value / max(reward_component_count, 1)
        for head in range(4):
            counts = np.bincount(actions_all[..., head].reshape(-1), minlength=40)
            probs = counts / max(counts.sum(), 1)
            row[f"action_head_{head}_entropy"] = float(-(probs[probs>0]*np.log(probs[probs>0])).sum())
            row[f"action_head_{head}_distribution"] = json.dumps(probs.tolist())
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, steps)
        writer.flush()
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=sorted({k for item in rows for k in item}))
            csv_writer.writeheader(); csv_writer.writerows(rows)
        at_episode_boundary = rollout_ended_at_episode_boundary
        if args.checkpoint_interval and (steps % args.checkpoint_interval == 0
                                         or steps == args.total_environment_steps):
            save_vanilla_happo_checkpoint(output / f"checkpoint_{steps}.pt", policy, trainer,
                                          environment_steps=steps, episodes=episodes,
                                          config=snapshot, numpy_rng=rng,
                                          checkpoint_type=("resumable" if at_episode_boundary
                                                           else "evaluation_weights"),
                                          at_episode_boundary=at_episode_boundary,
                                          policy_version=policy_version,
                                          seed_schedule={"episode_seed_base": episode_seed_base,
                                                         "next_episode": episodes,
                                                         "next_episode_seed": episode_seed_base + episodes})
        if should_run_evaluation(
                args.evaluation_interval, args.evaluation_episodes,
                steps, args.total_environment_steps):
            eval_env = make_paper_env(ROOT, args.scenario,
                                       initial_perturbation=args.evaluation_perturbation)
            latest_eval = deterministic_evaluate(eval_env, policy, args.evaluation_episodes,
                                                 evaluation_seed_base)
            eval_env.close()
            (output / f"evaluation_{steps}.json").write_text(
                json.dumps(latest_eval, indent=2), encoding="utf-8")
    final_checkpoint = output / "checkpoint_final.pt"
    save_vanilla_happo_checkpoint(final_checkpoint, policy, trainer,
                                  environment_steps=steps, episodes=episodes,
                                  config=snapshot, numpy_rng=rng,
                                  checkpoint_type=("resumable" if at_episode_boundary
                                                   else "evaluation_weights"),
                                  at_episode_boundary=at_episode_boundary,
                                  policy_version=policy_version,
                                  seed_schedule={"episode_seed_base": episode_seed_base,
                                                 "next_episode": episodes,
                                                 "next_episode_seed": episode_seed_base + episodes})
    numeric_metrics = [value for row in rows for value in row.values()
                       if isinstance(value, (int, float, bool))]
    optimization_active = bool(rows and np.isfinite(numeric_metrics).all()
                               and all(row["optimization_update_contract_valid"]
                                       and row["nan_inf_count"] == 0 for row in rows))
    runtime_valid = bool(rows and all(
        row.get("agent_order_count") == 1
        and row.get("factor_initialization_count") == 1
        and row.get("runtime_invariants_valid") for row in rows))
    summary = {"environment_steps": steps, "episodes": episodes,
               "latest_metrics": rows[-1], "latest_evaluation": latest_eval,
               "checkpoint": str(final_checkpoint.relative_to(ROOT)),
               "checkpoint_type": ("resumable" if at_episode_boundary
                                   else "evaluation_weights"),
               "resume_semantics": ("episode_boundary" if at_episode_boundary
                                    else "evaluation_only"),
               "resumed_from_semantics": resumed_from_semantics,
               "requested_seed": int(args.seed),
               "restored_seed": episode_seed_base,
               "OPTIMIZATION_PIPELINE_ACTIVE": optimization_active,
               "HAPPO_RUNTIME_INVARIANTS_VALID": runtime_valid,
               "EARLY_PERFORMANCE_SIGNAL_OBSERVED": False,
               "early_performance_signal_reason": "insufficient_evaluation_samples",
               "LEARNING_CONVERGENCE_NOT_VALIDATED": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    writer.close(); env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
