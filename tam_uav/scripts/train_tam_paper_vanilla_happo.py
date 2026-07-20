"""Formal independent-actor vanilla HAPPO trainer for tam_paper_env_v1."""

from __future__ import annotations

import argparse, csv, datetime, importlib.metadata, json, platform, random, sys
import time, traceback, warnings
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
from scripts.vanilla_happo_runtime import (
    active_action_statistics, deterministic_evaluate, flattened_obs,
    infer_policy, make_paper_env, seed_all)
from scripts.tam_output_paths import resolve_tam_output
from scripts.tam_learnability_metrics import (
    RecordWriter, finish_episode, flatten_evaluation, start_episode,
    strictly_better_evaluation, summarize_baseline, update_episode)
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL,
    PAPER_SILENT_ASSUMPTIONS_PRESENT, checkpoint_lineage, protocol_metadata,
    validate_nominal_protocol)


def is_episode_boundary(terminated, truncated):
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated shapes must match")
    return bool(np.logical_or(terminated, truncated).all())


def resume_seed_state(requested_seed, episodes, loaded):
    if loaded.get("resume_semantics") in {
            "episode_boundary", "exact_update_and_episode_boundary"}:
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


def format_console_training_line(steps, total_steps, update_count, episode_count,
                                 episode_records, elapsed_seconds):
    recent = episode_records[-100:]
    if recent:
        reward_last = float(recent[-1]["red_team_episode_return"])
        reward100 = float(np.mean([
            record["red_team_episode_return"] for record in recent]))
        win100 = float(np.mean([record["winner"] == "red" for record in recent]))
        survival100 = float(np.mean([
            record["red_survival_rate"] for record in recent]))
        hit100 = float(np.mean([record["red_hit_rate"] for record in recent]))
        crash100 = float(np.mean([
            record["red_crashes"] / max(record["red_initial_count"], 1)
            for record in recent]))
        values = {
            "reward_last": f"{reward_last:.1f}",
            "reward100": f"{reward100:.1f}",
            "win100": f"{win100:.2f}",
            "survival100": f"{survival100:.2f}",
            "hit100": f"{hit100:.2f}",
            "crash100": f"{crash100:.2f}",
        }
    else:
        values = {key: "N/A" for key in (
            "reward_last", "reward100", "win100", "survival100", "hit100",
            "crash100")}
    speed = steps / max(elapsed_seconds, 1e-12)
    progress = 100.0 * steps / max(total_steps, 1)
    return (
        f"[TRAIN] step={steps}/{total_steps} progress={progress:.2f}% "
        f"update={update_count} episodes={episode_count} "
        f"reward_last={values['reward_last']} reward100={values['reward100']} "
        f"win100={values['win100']} survival100={values['survival100']} "
        f"hit100={values['hit100']} crash100={values['crash100']} "
        f"speed={speed:.2f}step/s")


def select_run_directory(root, requested):
    output = resolve_tam_output(root, requested)
    if output.exists() and any(output.iterdir()):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = output.with_name(f"{output.name}_{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def process_rss_memory_mb():
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024.0 ** 2))
    except Exception:
        return None


def aggregate_episode_rows(records, agent_ids, roles):
    if not records:
        return {
            "red_team_episode_return": None,
            "mav_return": None,
            "uav_return": None,
            "win_rate": None,
            "draw_rate": None,
            "episode_length": None,
            "red_survival_rate": None,
            "blue_survival_rate": None,
            "survival_rate": None,
        }
    result = {
        "red_team_episode_return": float(np.mean(
            [record["red_team_episode_return"] for record in records])),
        "mean_episode_return": float(np.mean(
            [record["red_team_episode_return"] for record in records])),
        "win_rate": float(np.mean([record["winner"] == "red" for record in records])),
        "draw_rate": float(np.mean([record["winner"] == "draw" for record in records])),
        "episode_length": float(np.mean([record["episode_length"] for record in records])),
        "red_survival_rate": float(np.mean(
            [record["red_survival_rate"] for record in records])),
        "blue_survival_rate": float(np.mean(
            [record["blue_survival_rate"] for record in records])),
        "red_combat_survival_rate": float(np.mean(
            [record["red_combat_survival_rate"] for record in records])),
        "blue_combat_survival_rate": float(np.mean(
            [record["blue_combat_survival_rate"] for record in records])),
        "red_survivor_count": float(np.mean(
            [record["red_survivor_count"] for record in records])),
        "blue_survivor_count": float(np.mean(
            [record["blue_survivor_count"] for record in records])),
    }
    result["survival_rate"] = result["red_survival_rate"]
    for aid in agent_ids:
        result[f"agent_return/{aid}"] = float(np.mean(
            [record[f"agent_return/{aid}"] for record in records]))
    for role in ("mav", "attack_uav"):
        role_agents = [aid for aid in agent_ids if roles.get(aid) == role]
        result[f"role_mean_return/{role}"] = (
            float(np.mean([record[f"role_mean_return/{role}"]
                           for record in records])) if role_agents else None)
        result[f"role_total_return/{role}"] = (
            float(np.mean([record[f"role_total_return/{role}"]
                           for record in records])) if role_agents else None)
    result["mav_return"] = result["role_mean_return/mav"]
    result["uav_return"] = result["role_mean_return/attack_uav"]
    for side in ("red", "blue"):
        for key in ("missiles_fired", "hits", "hit_rate", "kills",
                    "boundary_deaths", "crashes", "structural_failures",
                    "maximum_speed_mps", "maximum_load_g"):
            result[f"{side}_{key}"] = float(np.mean(
                [record[f"{side}_{key}"] for record in records]))
    result["launch_rate"] = result["red_missiles_fired"]
    result["hit_rate"] = result["red_hit_rate"]
    result["boundary_rate"] = result["red_boundary_deaths"]
    result["structural_failure_rate"] = result["red_structural_failures"]
    return result


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
    p.add_argument("--disable-evaluation", action="store_true")
    p.add_argument("--console-log-interval", type=int, default=10240)
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
    p.add_argument("--actor-hidden-sizes", type=int, nargs=2, default=[256, 128])
    p.add_argument("--critic-hidden-sizes", type=int, nargs=2, default=[256, 128])
    p.add_argument("--value-loss-type", choices=(
        "clipped_huber", "legacy_clipped_mse"), default="clipped_huber")
    p.add_argument("--huber-delta", type=float, default=10.0)
    p.add_argument("--evaluation-seed-base", type=int, default=None)
    p.add_argument("--evaluation-perturbation", choices=(NOMINAL_PERTURBATION,),
                   default=NOMINAL_PERTURBATION,
                   help="paper_nominal periodic evaluation is fixed to perturbation none")
    return p.parse_args(argv)


def main():
    args = parse_args()
    validate_nominal_protocol(args.scenario, args.evaluation_perturbation)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    output = select_run_directory(ROOT, args.output_directory)
    env = make_paper_env(ROOT, args.scenario)
    policy, obs_dim, state_dim = infer_policy(
        env, args.actor_sharing, hidden_dim=args.hidden_dim, device=device,
        actor_hidden_sizes=args.actor_hidden_sizes,
        critic_hidden_sizes=args.critic_hidden_sizes)
    if args.actor_sharing == "role_shared_ablation":
        warnings.warn("role_shared_ablation is a legacy alias for "
                      "parameter_sharing_ppo_ablation", FutureWarning)
    if policy.actor_sharing == "independent":
        trainer = VanillaHAPPOTrainer(
            policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            clip_param=args.clip_param, value_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_gradient_norm,
            ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
            gamma=args.gamma, gae_lambda=args.gae_lambda, seed=args.seed,
            value_loss_type=args.value_loss_type, huber_delta=args.huber_delta)
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
            expected_scenario=args.scenario,
            expected_environment_fidelity_revision=ENVIRONMENT_FIDELITY_REVISION,
            expected_experiment_protocol=PAPER_NOMINAL_PROTOCOL,
            expected_initial_perturbation=NOMINAL_PERTURBATION,
            expected_dynamics_backend="jsbsim",
            expected_paper_silent_assumptions_present=(
                PAPER_SILENT_ASSUMPTIONS_PRESENT))
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
        "tam": False, "recurrent": False, "attention": False,
        "pure_happo": True,
        "actor_hidden_sizes": list(policy.actor_hidden_sizes),
        "critic_hidden_sizes": list(policy.critic_hidden_sizes),
        "value_loss_type": getattr(
            trainer, "value_loss_type", args.value_loss_type),
        "huber_delta": float(getattr(trainer, "huber_delta", args.huber_delta)),
        "paper_table_4_feedforward_alignment": bool(
            tuple(policy.actor_hidden_sizes) == (256, 128)
            and tuple(policy.critic_hidden_sizes) == (256, 128)
            and getattr(trainer, "value_loss_type", None) == "clipped_huber"
            and float(getattr(trainer, "huber_delta", 0.0)) == 10.0),
        "algorithm_mode": trainer.algorithm_mode,
        "algorithm_label": ("vanilla_happo" if args.actor_sharing == "independent"
                            else "parameter_sharing_ppo_ablation"),
        "reward_semantics": "heterogeneous_per_agent",
        "theoretical_team_reward_monotonic_guarantee_claimed": False,
        "resumed_from_semantics": resumed_from_semantics,
        "requested_seed": int(args.seed),
        "restored_seed": episode_seed_base,
        "next_episode_reset_seed": next_episode_seed,
        "evaluation_seed_base": int(evaluation_seed_base),
        "evaluation_perturbation": args.evaluation_perturbation,
        "evaluation_protocol": "fixed_seed_nominal",
        "requested_output_directory": args.output_directory,
        "actual_output_directory": str(output.relative_to(ROOT)),
        "training_code_generation": "telemetry_v2_active_action_metrics",
        "actor_sharing_label": ("formal_independent" if args.actor_sharing == "independent"
                                else "parameter_sharing_ablation"),
    } | protocol_metadata(
        args.scenario, NOMINAL_PERTURBATION, "jsbsim", PAPER_NOMINAL_PROTOCOL)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output / "tensorboard")
        snapshot["tensorboard_writer"] = "torch.utils.tensorboard"
    except Exception:
        from scripts.tensorboard_fallback import FallbackSummaryWriter
        writer = FallbackSummaryWriter(output / "tensorboard")
        snapshot["tensorboard_writer"] = "dependency_free_tfrecord_fallback"
    (output / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8")
    print(
        f"[CONFIG] scenario={args.scenario} seed={args.seed} "
        f"total_steps={args.total_environment_steps}")
    csv_path = output / "training.csv"
    episode_writer = RecordWriter(output / "episodes.csv", output / "episodes.jsonl")
    evaluation_writer = None
    if not args.disable_evaluation:
        evaluation_writer = RecordWriter(
            output / "evaluation_history.csv", output / "evaluation_history.jsonl")
    checkpoint_manifest_path = output / "checkpoints.jsonl"
    rows, episode_records, evaluation_records, checkpoint_records = [], [], [], []
    latest_eval = None
    current_checkpoint = None
    run_start = time.perf_counter()
    best_evaluation_return = None
    best_checkpoint = None
    at_episode_boundary = False
    last_console_log_step = 0

    def append_checkpoint_record(path, checkpoint_type, boundary, extra=None,
                                 update_boundary=False, discarded_steps=0):
        if checkpoint_type == "exact_update_and_episode_boundary":
            resume_semantics = checkpoint_type
        elif checkpoint_type == "episode_boundary_restart":
            resume_semantics = checkpoint_type
        else:
            resume_semantics = "evaluation_only"
        record = {
            "path": str(Path(path).relative_to(ROOT)),
            "checkpoint_type": checkpoint_type,
            "resume_semantics": resume_semantics,
            "at_episode_boundary": bool(boundary),
            "saved_at_episode_boundary": bool(boundary),
            "saved_at_update_boundary": bool(update_boundary),
            "discarded_partial_rollout_steps": int(discarded_steps),
            "exact_training_continuation": bool(
                checkpoint_type == "exact_update_and_episode_boundary"),
            "environment_steps": int(steps), "episodes": int(episodes),
            "policy_version": int(policy_version),
            "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        } | dict(extra or {})
        checkpoint_records.append(record)
        with checkpoint_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def save_checkpoint(path, checkpoint_type, boundary, extra_metadata=None,
                        update_boundary=False, discarded_steps=0):
        nonlocal current_checkpoint
        save_vanilla_happo_checkpoint(
            path, policy, trainer, environment_steps=steps, episodes=episodes,
            config=snapshot, numpy_rng=rng, checkpoint_type=checkpoint_type,
            at_episode_boundary=boundary, policy_version=policy_version,
            seed_schedule={"episode_seed_base": episode_seed_base,
                           "next_episode": episodes,
                           "next_episode_seed": episode_seed_base + episodes},
            extra_metadata=extra_metadata,
            saved_at_update_boundary=update_boundary,
            discarded_partial_rollout_steps=discarded_steps)
        current_checkpoint = str(Path(path).relative_to(ROOT))
        append_checkpoint_record(
            path, checkpoint_type, boundary, extra_metadata,
            update_boundary, discarded_steps)

    try:
        if not args.disable_evaluation:
            # Step-0 evaluation and baselines use this exact policy object.
            def run_evaluation(baseline, count, seed, explicit_seeds):
                eval_env = make_paper_env(ROOT, args.scenario)
                try:
                    return deterministic_evaluate(
                        eval_env, policy, count, seed, baseline=baseline,
                        episode_seeds=explicit_seeds)
                finally:
                    eval_env.close()

            evaluation_0 = run_evaluation(
                "trained_happo", 1, evaluation_seed_base, [evaluation_seed_base])
            evaluation_0.update({
                "policy_update_count": 0, "environment_steps": 0,
                "checkpoint": None, "evaluation_stage": "pre_training",
                "evaluated_environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
                "evaluated_experiment_protocol": PAPER_NOMINAL_PROTOCOL,
                "algorithm_label": snapshot["algorithm_label"],
                "step_0_policy_semantics": "untrained_initial_training_policy",
            })
            (output / "evaluation_0.json").write_text(
                json.dumps(evaluation_0, indent=2), encoding="utf-8")
            evaluation_0_row = flatten_evaluation(
                evaluation_0, environment_steps=0, trainer_update_count=0,
                policy_version=0, actor_sharing=policy.actor_sharing,
                algorithm_label=snapshot["algorithm_label"],
                evaluation_stage="pre_training", checkpoint=None)
            evaluation_writer.append(evaluation_0_row)
            evaluation_records.append(evaluation_0_row)

            rule_seed = evaluation_seed_base + 1
            neutral_seed = evaluation_seed_base + 2
            random_seeds = [evaluation_seed_base + 100 + index for index in range(10)]
            baseline_reference = {
                "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
                "experiment_protocol": PAPER_NOMINAL_PROTOCOL,
                "scenario": args.scenario,
                "algorithm_label": snapshot["algorithm_label"],
                "untrained_policy_is_training_policy_object": True,
                "rule": summarize_baseline(run_evaluation(
                    "rule", 1, rule_seed, [rule_seed])),
                "neutral": summarize_baseline(run_evaluation(
                    "neutral", 1, neutral_seed, [neutral_seed])),
                "untrained_happo": summarize_baseline(evaluation_0),
                "random": summarize_baseline(run_evaluation(
                    "random", 10, random_seeds[0], random_seeds)),
            }
            (output / "baseline_reference.json").write_text(
                json.dumps(baseline_reference, indent=2), encoding="utf-8")

        obs, _ = env.reset(seed=next_episode_seed)
        episode_id = episodes
        episode_accumulator = start_episode(
            env, episodes, next_episode_seed, steps, policy_version)
        while (steps < args.total_environment_steps
               and (args.max_updates is None or trainer.update_count < args.max_updates)):
            update_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            horizon = min(args.rollout_length, args.total_environment_steps - steps)
            intervals = [args.checkpoint_interval]
            if not args.disable_evaluation:
                intervals.insert(0, args.evaluation_interval)
            for interval in intervals:
                if interval:
                    horizon = min(horizon, interval - steps % interval)
            buffer = VanillaHAPPORolloutBuffer(horizon, env.num_agents, obs_dim, state_dim)
            completed_records = []
            target_violations = nonfinite = 0
            rollout_ended_at_episode_boundary = False
            component_sums = {aid: {} for aid in env.agent_ids}
            component_counts = {aid: 0 for aid in env.agent_ids}
            collection_start = time.perf_counter()
            for _local_step in range(horizon):
                rollout_ended_at_episode_boundary = False
                actor_obs, state = flattened_obs(env, obs), env.get_state()
                available_dict = env.get_avail_actions()
                available = np.stack([available_dict[aid] for aid in env.agent_ids])
                alive_start = np.array([
                    float(next(agent for agent in env.task.agents
                               if agent.agent_id == aid).alive)
                    for aid in env.agent_ids], np.float32)
                with torch.no_grad():
                    action_out = policy.act(actor_obs, state, available)
                actions = action_out["actions"].cpu().numpy()
                next_obs, rewards, term, trunc, info = env.step(
                    {aid: actions[index] for index, aid in enumerate(env.agent_ids)})
                reward_array = np.array(
                    [rewards[aid] for aid in env.agent_ids], np.float32)
                alive_end = info["alive_at_step_end"]
                global_truncated = all(trunc.values())
                global_terminated = all(term.values()) and not global_truncated
                terminated = np.array([
                    float(not alive_end[aid] or global_terminated)
                    for aid in env.agent_ids])
                truncated = np.array([
                    float(global_truncated and alive_end[aid]) for aid in env.agent_ids])
                with torch.no_grad():
                    next_value = policy.value(torch.as_tensor(
                        env.get_state(), dtype=torch.float32,
                        device=device)).cpu().numpy()
                buffer.add(
                    obs=actor_obs, state=state, actions=actions,
                    log_probs=action_out["log_probs"].cpu().numpy(),
                    rewards=reward_array,
                    value=action_out["value"].cpu().numpy(), next_value=next_value,
                    terminated=terminated, truncated=truncated,
                    active_masks=alive_start, available_actions=available,
                    agent_alive=alive_start, episode_id=episode_id,
                    decision_step=info["episode_step"], policy_version=policy_version)
                steps += 1; obs = next_obs
                update_episode(episode_accumulator, env, rewards, info)
                target_violations += len(info["target_consistency_violation"])
                nonfinite += int(not np.isfinite(reward_array).all()
                                 or not np.isfinite(env.get_state()).all())
                for aid in env.agent_ids:
                    for key, value in info["reward_components"][aid].items():
                        component_sums[aid][key] = (
                            component_sums[aid].get(key, 0.0) + float(value))
                    component_counts[aid] += 1
                episode_done = all(bool(term[aid] or trunc[aid])
                                   for aid in env.agent_ids)
                if episode_done:
                    rollout_ended_at_episode_boundary = True
                    episode_record = finish_episode(
                        episode_accumulator, env, info, steps,
                        snapshot["algorithm_label"], ENVIRONMENT_FIDELITY_REVISION,
                        PAPER_NOMINAL_PROTOCOL)
                    episode_writer.append(episode_record)
                    episode_records.append(episode_record)
                    completed_records.append(episode_record)
                    episodes += 1; episode_id += 1
                    next_episode_seed = episode_seed_base + episodes
                    save_checkpoint(
                        output / "latest_episode_boundary_restart.pt",
                        "episode_boundary_restart", True,
                        update_boundary=False,
                        discarded_steps=_local_step + 1)
                    obs, _ = env.reset(seed=next_episode_seed)
                    episode_accumulator = start_episode(
                        env, episodes, next_episode_seed, steps, policy_version)
            collection_wall_time = time.perf_counter() - collection_start
            if buffer.pos != horizon:
                raise RuntimeError(
                    f"buffer.pos ({buffer.pos}) != planned_horizon ({horizon}) "
                    f"at update {trainer.update_count}")
            optimization_start = time.perf_counter()
            before = {name: value.detach().clone()
                      for name, value in policy.named_parameters()}
            result = trainer.update(buffer); policy_version += 1
            optimization_wall_time = time.perf_counter() - optimization_start
            unchanged = [name for name, value in policy.named_parameters()
                         if torch.equal(before[name], value)]
            actor_changed = {
                aid: any(not torch.equal(
                    before[f"actors.{policy.actor_key(aid)}.{name}"], value)
                    for name, value in policy.actors[
                        policy.actor_key(aid)].named_parameters())
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
            row = {
                "update_index": int(trainer.update_count),
                "environment_steps": steps, "episodes": episodes,
                "completed_episode_count_in_rollout": len(completed_records),
                "target_consistency_violation": target_violations,
                "nan_inf_count": nonfinite,
                "unchanged_parameter_count": len(unchanged),
                "actor_changed": json.dumps(actor_changed),
                "all_actors_changed": all(actor_changed.values()),
                "critic_changed": critic_changed,
                "critic_head_changed": json.dumps(critic_head_changed),
                "all_critic_heads_changed": all(critic_head_changed.values()),
                "update_expected": json.dumps(update_expected),
                "update_contract_valid": json.dumps(update_contract_valid),
                "optimization_update_contract_valid": all(
                    update_contract_valid.values()),
                "rollout_planned_horizon": horizon,
                "rollout_collected_steps": buffer.pos,
                "rollout_episode_count": len(completed_records),
                "rollout_ended_at_episode_boundary": rollout_ended_at_episode_boundary,
                "minimum_active_sample_count": min(
                    result.metrics.get(f"active_sample_count/{aid}", 0)
                    for aid in policy.agent_ids),
                "maximum_approx_kl": max(
                    result.metrics.get(f"approx_kl/{aid}", 0.0)
                    for aid in policy.agent_ids),
                "collection_wall_time_s": collection_wall_time,
                "optimization_wall_time_s": optimization_wall_time,
                "evaluation_wall_time_s": 0.0,
                "environment_steps_per_second": (
                    horizon / max(collection_wall_time, 1e-12)),
                "cuda_peak_allocated_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                    if device.type == "cuda" else None),
                "cuda_peak_reserved_memory_mb": (
                    float(torch.cuda.max_memory_reserved(device) / 1024 ** 2)
                    if device.type == "cuda" else None),
                "process_rss_memory_mb": process_rss_memory_mb(),
            } | aggregate_episode_rows(
                completed_records, env.agent_ids, env.agent_roles) | result.metrics
            for aid in policy.agent_ids:
                row[f"update_expected/{aid}"] = update_expected[aid]
                row[f"actor_changed/{aid}"] = actor_changed[aid]
                row[f"critic_head_changed/{aid}"] = critic_head_changed[aid]
                row[f"update_contract_valid/{aid}"] = update_contract_valid[aid]
            controlled_component_totals, controlled_component_count = {}, {}
            for aid in env.agent_ids:
                for key, value in component_sums[aid].items():
                    mean = value / max(component_counts[aid], 1)
                    row[f"reward_component/agent/{aid}/{key}"] = mean
                    controlled_component_totals[key] = (
                        controlled_component_totals.get(key, 0.0) + value)
                    controlled_component_count[key] = (
                        controlled_component_count.get(key, 0) + component_counts[aid])
            for key, value in controlled_component_totals.items():
                mean = value / max(controlled_component_count[key], 1)
                row[f"reward_component/controlled_mean/{key}"] = mean
                row[f"reward_component/{key}"] = mean
            for role in ("mav", "attack_uav"):
                role_ids = [aid for aid in env.agent_ids
                            if env.agent_roles.get(aid) == role]
                if role_ids:
                    component_keys = sorted({
                        key for aid in role_ids for key in component_sums[aid]})
                    for key in component_keys:
                        numerator = sum(component_sums[aid].get(key, 0.0)
                                        for aid in role_ids)
                        denominator = sum(component_counts[aid] for aid in role_ids)
                        row[f"reward_component/role/{role}/{key}"] = (
                            numerator / max(denominator, 1))
            active_stats = active_action_statistics(
                actions_all, buffer.active_masks[:buffer.pos], action_levels=40)
            row["active_action_sample_count"] = active_stats[
                "active_action_sample_count"]
            for head in range(4):
                entropy = active_stats[f"active_action_head_{head}_entropy"]
                distribution = active_stats[
                    f"active_action_head_{head}_distribution"]
                encoded = json.dumps(distribution) if distribution is not None else None
                row[f"active_action_head_{head}_entropy"] = entropy
                row[f"active_action_head_{head}_distribution"] = encoded
                row[f"action_head_{head}_entropy"] = entropy
                row[f"action_head_{head}_distribution"] = encoded

            invalid_reason = None
            numeric_result = [value for value in result.metrics.values()
                              if isinstance(value, (int, float, bool))]
            if nonfinite or not np.isfinite(numeric_result).all():
                invalid_reason = "nonfinite training signal"
            elif target_violations:
                invalid_reason = "target consistency violation"
            elif not all(update_contract_valid.values()):
                invalid_reason = "optimization update contract failure"
            elif buffer.pos != horizon:
                invalid_reason = "rollout horizon mismatch"

            if (not args.disable_evaluation and invalid_reason is None
                    and should_run_evaluation(
                    args.evaluation_interval, args.evaluation_episodes,
                    steps, args.total_environment_steps)):
                evaluation_start = time.perf_counter()
                latest_eval = run_evaluation(
                    "trained_happo", args.evaluation_episodes,
                    evaluation_seed_base, [evaluation_seed_base])
                latest_eval.update({
                    "policy_update_count": int(trainer.update_count),
                    "environment_steps": int(steps),
                    "checkpoint": None, "evaluation_stage": "periodic",
                    "evaluated_environment_fidelity_revision": (
                        ENVIRONMENT_FIDELITY_REVISION),
                    "evaluated_experiment_protocol": PAPER_NOMINAL_PROTOCOL,
                    "algorithm_label": snapshot["algorithm_label"],
                })
                evaluation_row = flatten_evaluation(
                    latest_eval, environment_steps=steps,
                    trainer_update_count=trainer.update_count,
                    policy_version=policy_version,
                    actor_sharing=policy.actor_sharing,
                    algorithm_label=snapshot["algorithm_label"],
                    evaluation_stage="periodic", checkpoint=None)
                evaluation_writer.append(evaluation_row)
                evaluation_records.append(evaluation_row)
                (output / f"evaluation_{steps}.json").write_text(
                    json.dumps(latest_eval, indent=2), encoding="utf-8")
                row["evaluation_wall_time_s"] = (
                    time.perf_counter() - evaluation_start)
                candidate_return = evaluation_row["red_team_episode_return"]
                if strictly_better_evaluation(best_evaluation_return, candidate_return):
                    best_evaluation_return = candidate_return
                    best_checkpoint = output / "best_evaluation_checkpoint.pt"
                    best_extra = {
                        "selected_by": "best_deterministic_red_team_return",
                        "selected_at_environment_steps": int(steps),
                        "evaluation_red_team_return": float(candidate_return),
                        "evaluation_winner": evaluation_row["winner"],
                    }
                    save_checkpoint(
                        best_checkpoint, "evaluation_weights", False, best_extra)

            row["update_wall_time_s"] = time.perf_counter() - update_start
            row["elapsed_wall_time_s"] = time.perf_counter() - run_start
            rows.append(row)
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(key, value, steps)
            writer.flush()
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                fields = sorted({key for item in rows for key in item})
                csv_writer = csv.DictWriter(handle, fieldnames=fields)
                csv_writer.writeheader(); csv_writer.writerows(rows)
            if invalid_reason is not None:
                raise RuntimeError(invalid_reason)

            at_episode_boundary = rollout_ended_at_episode_boundary
            if at_episode_boundary:
                save_checkpoint(
                    output / "latest_exact_resumable.pt",
                    "exact_update_and_episode_boundary", True,
                    update_boundary=True, discarded_steps=0)
            checkpoint_due = bool(args.checkpoint_interval and (
                    steps % args.checkpoint_interval == 0
                    or steps == args.total_environment_steps))
            if checkpoint_due:
                save_checkpoint(
                    output / f"checkpoint_{steps}.pt", "evaluation_weights", False,
                    update_boundary=True)
            interval_due = bool(
                args.console_log_interval > 0
                and steps - last_console_log_step >= args.console_log_interval)
            if interval_due or checkpoint_due or steps == args.total_environment_steps:
                print(format_console_training_line(
                    steps, args.total_environment_steps, trainer.update_count,
                    episodes, episode_records, time.perf_counter() - run_start))
                last_console_log_step = steps

        final_checkpoint = output / "checkpoint_final.pt"
        final_type = "evaluation_weights"
        save_checkpoint(final_checkpoint, final_type, False, update_boundary=True)
        numeric_metrics = [value for row in rows for value in row.values()
                           if isinstance(value, (int, float, bool))]
        optimization_active = bool(
            rows and np.isfinite(numeric_metrics).all()
            and all(row["optimization_update_contract_valid"]
                    and row["nan_inf_count"] == 0 for row in rows))
        runtime_valid = bool(rows and all(
            row.get("agent_order_count") == 1
            and row.get("factor_initialization_count") == 1
            and row.get("runtime_invariants_valid")
            and row["rollout_planned_horizon"] == row["rollout_collected_steps"]
            for row in rows))
        approx_values = [abs(float(row.get(f"approx_kl/{aid}", 0.0)))
                         for row in rows for aid in policy.agent_ids]
        clip_values = [float(row.get(f"clip_fraction/{aid}", 0.0))
                       for row in rows for aid in policy.agent_ids]
        entropy_values = [float(row.get(f"entropy/{aid}", 0.0))
                          for row in rows for aid in policy.agent_ids]
        actor_gradients = [float(row.get(f"gradient_norm/{aid}", 0.0))
                           for row in rows for aid in policy.agent_ids]
        critic_gradients = [float(row.get("critic_gradient_norm", 0.0))
                            for row in rows]
        total_wall_time = time.perf_counter() - run_start
        health = {
            "max_abs_approx_kl": max(approx_values, default=0.0),
            "max_clip_fraction": max(clip_values, default=0.0),
            "min_actor_entropy": min(entropy_values, default=0.0),
            "max_actor_gradient_norm": max(actor_gradients, default=0.0),
            "max_critic_gradient_norm": max(critic_gradients, default=0.0),
            "updates_with_failed_contract": sum(
                not row["optimization_update_contract_valid"] for row in rows),
            "updates_with_nonfinite": sum(row["nan_inf_count"] > 0 for row in rows),
            "updates_with_zero_active_controlled_agent": sum(
                row["minimum_active_sample_count"] == 0 for row in rows),
            "updates_with_any_actor_unchanged_when_expected": sum(any(
                row[f"update_expected/{aid}"] and not row[f"actor_changed/{aid}"]
                for aid in policy.agent_ids) for row in rows),
            "updates_with_any_critic_head_unchanged_when_expected": sum(any(
                row[f"update_expected/{aid}"]
                and not row[f"critic_head_changed/{aid}"]
                for aid in policy.agent_ids) for row in rows),
        }
        summary = {
            "environment_steps": steps, "episodes": episodes,
            "latest_metrics": rows[-1] if rows else None,
            "latest_evaluation": latest_eval,
            "checkpoint": str(final_checkpoint.relative_to(ROOT)),
            "checkpoint_type": final_type,
            "resume_semantics": (
                "episode_boundary" if at_episode_boundary else "evaluation_only"),
            "resumed_from_semantics": resumed_from_semantics,
            "requested_seed": int(args.seed), "restored_seed": episode_seed_base,
            "OPTIMIZATION_PIPELINE_ACTIVE": optimization_active,
            "HAPPO_RUNTIME_INVARIANTS_VALID": runtime_valid,
            "EARLY_PERFORMANCE_SIGNAL_OBSERVED": False,
            "early_performance_signal_reason": "requires learnability analysis",
            "LEARNING_CONVERGENCE_NOT_VALIDATED": True,
            "evaluated_environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
            "evaluated_experiment_protocol": PAPER_NOMINAL_PROTOCOL,
            "algorithm_label": snapshot["algorithm_label"],
            "total_wall_time_s": total_wall_time,
            "mean_environment_steps_per_second": float(np.mean(
                [row["environment_steps_per_second"] for row in rows])),
            "total_updates": len(rows),
            "total_completed_episodes": len(episode_records),
            "actual_final_environment_steps": steps,
            "expected_target_environment_steps": args.total_environment_steps,
            "optimization_health_summary": health,
            "checkpoint_records": checkpoint_records,
            "best_evaluation_checkpoint": (
                str(best_checkpoint.relative_to(ROOT)) if best_checkpoint else None),
            "best_evaluation_red_team_return": best_evaluation_return,
        } | checkpoint_lineage(
            snapshot | {"environment_steps": steps, "episodes": episodes})
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[DONE] steps={steps} updates={trainer.update_count} "
            f"episodes={episodes} wall_time={total_wall_time:.1f}s output={output}")
    except Exception:
        failure = {
            "environment_steps": int(steps),
            "update_index": int(getattr(trainer, "update_count", 0)),
            "episode_index": int(episodes),
            "traceback": traceback.format_exc(),
            "recent_training_metrics": rows[-5:],
            "recent_episodes": episode_records[-5:],
            "current_checkpoint": current_checkpoint,
            "seed": int(args.seed), "episode_seed_base": int(episode_seed_base),
            "next_episode_seed": int(next_episode_seed),
            "cuda_peak_allocated_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                if device.type == "cuda" else None),
            "cuda_peak_reserved_memory_mb": (
                float(torch.cuda.max_memory_reserved(device) / 1024 ** 2)
                if device.type == "cuda" else None),
        }
        (output / "failure_report.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        writer.close(); env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
