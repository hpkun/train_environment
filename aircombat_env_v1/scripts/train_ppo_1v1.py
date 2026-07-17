"""Train feed-forward PPO with a performance-driven 1v1 curriculum."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

for _name, _value in (
    ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE"),
):
    os.environ.setdefault(_name, _value)

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aircombat_env_v1.evaluation import evaluate_policy
from aircombat_env_v1.ppo import Actor, Critic, RolloutBuffer, ppo_update
from aircombat_env_v1.training import (
    PerformanceCurriculum, TrainingConfig, append_csv, best_fixed_key,
    best_joint_key, best_randomized_key, load_training_state, resolve_device,
    save_model, save_training_state, set_random_seeds, write_json)
from aircombat_env_v1.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=150000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def _episode_counts(infos, lengths):
    counts = {
        "completed_episodes": 0, "valid_completed_episodes": 0,
        "red_hits": 0, "blue_hits": 0, "red_crashes": 0,
        "blue_crashes": 0, "opponent_failures": 0,
        "numerical_invalid_episodes": 0, "timeouts": 0,
    }
    completed_lengths = []
    for rank, info in enumerate(infos):
        terminal = info.get("terminal_info")
        if terminal is None:
            continue
        counts["completed_episodes"] += 1
        completed_lengths.append(int(lengths[rank]))
        lengths[rank] = 0
        event = terminal.get("event")
        if terminal.get("valid_combat_outcome", False):
            counts["valid_completed_episodes"] += 1
        mapping = {
            "red_hit": "red_hits", "blue_hit": "blue_hits",
            "red_crash": "red_crashes", "blue_crash": "blue_crashes",
            "timeout": "timeouts",
        }
        if event in mapping:
            counts[mapping[event]] += 1
        if terminal.get("opponent_failure", False):
            counts["opponent_failures"] += 1
        if terminal.get("invalid_episode", False):
            counts["numerical_invalid_episodes"] += 1
    return counts, completed_lengths


def _maybe_save_best(path, key, current_record, actor, critic, config,
                     global_step, result):
    if key is None:
        return current_record, False
    if current_record is None or tuple(key) > tuple(current_record["key"]):
        save_model(path, actor, critic, config, global_step)
        return {
            "global_step": int(global_step), "key": list(key),
            "red_hit_rate": result.get("red_hit_rate"),
            "mean_return": result.get("mean_return"),
        }, True
    return current_record, False


def main():
    args = parse_args()
    config = TrainingConfig(
        total_steps=args.total_steps, num_envs=args.num_envs,
        rollout_steps=args.rollout_steps, seed=args.seed, device=args.device,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "aircombat_env_v1/outputs"
    ) / f"ppo_1v1_learnability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {
        **asdict(config),
        "algorithm_claim": "single_agent_PPO_learnability_baseline",
        "paper_table4_matches": {
            "gamma": 0.99, "gae_lambda": 0.95,
            "clip_epsilon": 0.2, "entropy_coef": 0.01,
        },
        "engineering_parameters": {
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
        },
    })
    set_random_seeds(config.seed)
    device = resolve_device(config.device)
    actor, critic = Actor().to(device), Critic().to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate)
    global_step = update_index = 0
    curriculum = PerformanceCurriculum()
    best_records = {"fixed": None, "randomized": None, "joint": None}
    if args.resume:
        state = load_training_state(
            args.resume, actor, critic, optimizer, device)
        global_step = int(state["global_step"])
        update_index = int(state["update_index"])
        curriculum = PerformanceCurriculum(**(
            state.get("curriculum") or {}))
        best_records.update(state.get("best_records") or {})

    next_eval = (
        (global_step // config.eval_interval) + 1) * config.eval_interval
    start_time = time.perf_counter()
    with SubprocVecEnv(
            config.num_envs, config.seed,
            {"scenario_mode": "fixed_tail_chase",
             "opponent_policy": "paper_greedy", "max_steps": 1000}) as vec_env:
        vec_env.set_curriculum_stage(curriculum.stage)
        observations, _ = vec_env.reset()
        episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        print(
            f"curriculum stage {curriculum.stage} at step {global_step}",
            flush=True)
        while global_step < config.total_steps:
            remaining = config.total_steps - global_step
            rollout_steps = min(
                config.rollout_steps,
                max(1, int(np.ceil(remaining / config.num_envs))))
            buffer = RolloutBuffer(rollout_steps, config.num_envs)
            rollout_rewards = []
            aggregate = {
                "completed_episodes": 0, "valid_completed_episodes": 0,
                "red_hits": 0, "blue_hits": 0, "red_crashes": 0,
                "blue_crashes": 0, "opponent_failures": 0,
                "numerical_invalid_episodes": 0, "timeouts": 0,
            }
            update_lengths = []
            for _ in range(rollout_steps):
                observation_tensor = torch.as_tensor(
                    observations, dtype=torch.float32, device=device)
                with torch.no_grad():
                    distribution = actor.distribution(observation_tensor)
                    actions_tensor = distribution.sample()
                    log_probs_tensor = distribution.log_prob(actions_tensor)
                    values_tensor = critic(observation_tensor)
                actions = actions_tensor.cpu().numpy()
                next_observations, rewards, terminated, truncated, infos = (
                    vec_env.step(actions))
                episode_lengths += 1
                bootstrap_observations = next_observations.copy()
                for rank, info in enumerate(infos):
                    if truncated[rank]:
                        bootstrap_observations[rank] = info[
                            "terminal_observation"]
                with torch.no_grad():
                    next_values = critic(torch.as_tensor(
                        bootstrap_observations, dtype=torch.float32,
                        device=device)).cpu().numpy()
                buffer.add(
                    observations, actions, rewards,
                    values_tensor.cpu().numpy(), next_values,
                    log_probs_tensor.cpu().numpy(), terminated, truncated)
                observations = next_observations
                rollout_rewards.extend(rewards.tolist())
                counts, completed_lengths = _episode_counts(
                    infos, episode_lengths)
                for key, value in counts.items():
                    aggregate[key] += value
                update_lengths.extend(completed_lengths)
            global_step += rollout_steps * config.num_envs
            update_index += 1
            batch = buffer.flatten(config.gamma, config.gae_lambda)
            try:
                metrics = ppo_update(
                    actor, critic, optimizer, batch, device,
                    config.clip_epsilon, config.update_epochs,
                    config.minibatch_size, config.entropy_coef,
                    config.value_coef, config.max_grad_norm, config.target_kl)
            except FloatingPointError:
                torch.save({
                    "global_step": global_step, "batch": batch,
                    "actor": actor.state_dict(), "critic": critic.state_dict(),
                }, output_dir / "diagnostic_nonfinite.pt")
                raise
            elapsed = time.perf_counter() - start_time
            row = {
                "global_step": global_step, "update": update_index,
                "curriculum_stage": curriculum.stage,
                "mean_rollout_reward": float(np.mean(rollout_rewards)),
                **aggregate,
                "mean_episode_length": (
                    float(np.mean(update_lengths)) if update_lengths else 0.0),
                **metrics,
                "environment_steps_per_second": global_step / max(elapsed, 1e-9),
            }
            append_csv(output_dir / "training_log.csv", row)
            save_model(
                output_dir / "latest.pt", actor, critic, config, global_step)

            if global_step >= next_eval or global_step >= config.total_steps:
                fixed = evaluate_policy(
                    "ppo", config.eval_episodes, "fixed_tail_chase",
                    "paper_greedy", config.seed + global_step,
                    actor=actor, device=device)
                randomized = None
                fixed_updated = randomized_updated = joint_updated = False
                best_records["fixed"], fixed_updated = _maybe_save_best(
                    output_dir / "best_fixed.pt", best_fixed_key(fixed),
                    best_records["fixed"], actor, critic, config, global_step,
                    fixed)
                if curriculum.stage == 2:
                    randomized = evaluate_policy(
                        "ppo", config.eval_episodes,
                        "randomized_tail_chase", "paper_greedy",
                        config.seed + global_step, actor=actor, device=device)
                    best_records["randomized"], randomized_updated = (
                        _maybe_save_best(
                            output_dir / "best_randomized.pt",
                            best_randomized_key(randomized),
                            best_records["randomized"], actor, critic, config,
                            global_step, randomized))
                    joint_key = best_joint_key(fixed, randomized)
                    joint_result = {
                        "red_hit_rate": min(
                            fixed["red_hit_rate"], randomized["red_hit_rate"]),
                        "mean_return": 0.5 * (
                            fixed["mean_return"] + randomized["mean_return"]),
                    }
                    best_records["joint"], joint_updated = _maybe_save_best(
                        output_dir / "best_joint.pt", joint_key,
                        best_records["joint"], actor, critic, config,
                        global_step, joint_result)
                common = {
                    "global_step": global_step,
                    "curriculum_stage": curriculum.stage,
                    "best_fixed_updated": fixed_updated,
                    "best_randomized_updated": randomized_updated,
                    "best_joint_updated": joint_updated,
                }
                append_csv(
                    output_dir / "evaluation_log.csv", {**common, **fixed})
                if randomized is not None:
                    append_csv(
                        output_dir / "evaluation_log.csv",
                        {**common, **randomized})
                transitioned = curriculum.update(
                    fixed["red_hit_rate"],
                    None if randomized is None else randomized["red_hit_rate"])
                if transitioned:
                    vec_env.set_curriculum_stage(curriculum.stage)
                    print(
                        f"curriculum stage {curriculum.stage} at step "
                        f"{global_step}", flush=True)
                while next_eval <= global_step:
                    next_eval += config.eval_interval

            legacy_fixed = best_records["fixed"] or {}
            save_training_state(
                output_dir / "training_state.pt", actor, critic, optimizer,
                global_step, update_index,
                legacy_fixed.get("red_hit_rate", -1.0), config,
                legacy_fixed.get("mean_return", float("-inf")), 0,
                curriculum.state_dict(), best_records)
            print(
                f"step={global_step} update={update_index} "
                f"stage={curriculum.stage} "
                f"reward={row['mean_rollout_reward']:.4f} "
                f"kl={metrics['approximate_kl']:.5f}", flush=True)
            if curriculum.learnability_passed:
                print(
                    f"learnability gate passed at step {global_step}",
                    flush=True)
                break

    save_model(output_dir / "final.pt", actor, critic, config, global_step)
    summary = {
        "output_dir": str(output_dir.resolve()),
        "global_step": global_step, "updates": update_index,
        "elapsed_seconds": time.perf_counter() - start_time,
        "curriculum": curriculum.state_dict(),
        "best_records": best_records,
        "learnability_passed": curriculum.learnability_passed,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
