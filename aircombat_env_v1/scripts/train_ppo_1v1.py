"""Train feed-forward PPO on the single-agent JSBSim 1v1 environment."""

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
    TrainingConfig, append_csv, curriculum_stage, load_training_state,
    resolve_device, save_model, save_training_state, set_random_seeds,
    write_json)
from aircombat_env_v1.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=100000)
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
        "completed_episodes": 0, "red_hits": 0, "blue_hits": 0,
        "red_crashes": 0, "blue_crashes": 0,
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
        mapping = {
            "red_hit": "red_hits", "blue_hit": "blue_hits",
            "red_crash": "red_crashes", "blue_crash": "blue_crashes",
            "timeout": "timeouts",
        }
        if event in mapping:
            counts[mapping[event]] += 1
        if "numerical_invalid" in str(event):
            counts["numerical_invalid_episodes"] += 1
    return counts, completed_lengths


def _is_better(result, best_hit_rate, best_return, best_invalid):
    key = (
        result["red_hit_rate"],
        result["mean_return"],
        -result["numerical_invalid"],
    )
    return key > (best_hit_rate, best_return, -best_invalid)


def main():
    args = parse_args()
    config = TrainingConfig(
        total_steps=args.total_steps, num_envs=args.num_envs,
        rollout_steps=args.rollout_steps, seed=args.seed, device=args.device,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "aircombat_env_v1/outputs"
    ) / f"ppo_1v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))
    set_random_seeds(config.seed)
    device = resolve_device(config.device)
    actor, critic = Actor().to(device), Critic().to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate)
    global_step = update_index = 0
    best_hit_rate, best_return, best_invalid = -1.0, float("-inf"), 10**9
    if args.resume:
        state = load_training_state(
            args.resume, actor, critic, optimizer, device)
        global_step = int(state["global_step"])
        update_index = int(state["update_index"])
        best_hit_rate = float(state["best_eval_hit_rate"])
        best_return = float(state.get("best_eval_return", float("-inf")))
        best_invalid = int(state.get(
            "best_eval_numerical_invalid", 10**9))

    baseline_episodes = 20
    zero_result = evaluate_policy(
        "zero", baseline_episodes, seed=config.seed)
    random_result = evaluate_policy(
        "random", baseline_episodes, seed=config.seed)
    rule_result = evaluate_policy(
        "pursuit_rule", baseline_episodes, seed=config.seed)
    initial_ppo = evaluate_policy(
        "ppo", baseline_episodes, seed=config.seed, actor=actor, device=device)

    next_eval = (
        (global_step // config.eval_interval) + 1) * config.eval_interval
    stage = curriculum_stage(global_step, config.total_steps)
    start_time = time.perf_counter()
    all_episode_lengths = []
    observations = None
    with SubprocVecEnv(
            config.num_envs, config.seed,
            {"scenario_mode": "fixed_tail_chase",
             "opponent_policy": "straight", "max_steps": 1000}) as vec_env:
        vec_env.set_curriculum_stage(stage)
        observations, _ = vec_env.reset()
        episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        print(f"curriculum stage {stage} at step {global_step}", flush=True)
        while global_step < config.total_steps:
            desired_stage = curriculum_stage(global_step, config.total_steps)
            if desired_stage != stage:
                stage = desired_stage
                vec_env.set_curriculum_stage(stage)
                print(
                    f"curriculum stage {stage} at step {global_step}",
                    flush=True)
            remaining = config.total_steps - global_step
            rollout_steps = min(
                config.rollout_steps,
                max(1, int(np.ceil(remaining / config.num_envs))))
            buffer = RolloutBuffer(
                rollout_steps, config.num_envs)
            rollout_rewards = []
            aggregate = {
                "completed_episodes": 0, "red_hits": 0, "blue_hits": 0,
                "red_crashes": 0, "blue_crashes": 0,
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
            all_episode_lengths.extend(update_lengths)
            elapsed = time.perf_counter() - start_time
            row = {
                "global_step": global_step, "update": update_index,
                "curriculum_stage": stage,
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
            save_training_state(
                output_dir / "training_state.pt", actor, critic, optimizer,
                global_step, update_index, best_hit_rate, config,
                best_return, best_invalid)
            if global_step >= next_eval or global_step >= config.total_steps:
                fixed = evaluate_policy(
                    "ppo", config.eval_episodes, "fixed_tail_chase",
                    "straight", config.seed + global_step, actor=actor,
                    device=device)
                randomized = evaluate_policy(
                    "ppo", config.eval_episodes, "randomized_tail_chase",
                    "straight", config.seed + global_step, actor=actor,
                    device=device)
                for result in (fixed, randomized):
                    append_csv(
                        output_dir / "evaluation_log.csv",
                        {"global_step": global_step, **result})
                if _is_better(
                        randomized, best_hit_rate, best_return, best_invalid):
                    best_hit_rate = randomized["red_hit_rate"]
                    best_return = randomized["mean_return"]
                    best_invalid = randomized["numerical_invalid"]
                    save_model(
                        output_dir / "best.pt", actor, critic, config,
                        global_step)
                save_training_state(
                    output_dir / "training_state.pt", actor, critic, optimizer,
                    global_step, update_index, best_hit_rate, config,
                    best_return, best_invalid)
                while next_eval <= global_step:
                    next_eval += config.eval_interval
            print(
                f"step={global_step} update={update_index} "
                f"reward={row['mean_rollout_reward']:.4f} "
                f"kl={metrics['approximate_kl']:.5f}", flush=True)

    save_model(output_dir / "final.pt", actor, critic, config, global_step)
    if not (output_dir / "best.pt").exists():
        save_model(output_dir / "best.pt", actor, critic, config, global_step)
    final_fixed = evaluate_policy(
        "ppo", 20, "fixed_tail_chase", "straight", config.seed + 1000000,
        actor=actor, device=device)
    final_randomized = evaluate_policy(
        "ppo", 20, "randomized_tail_chase", "straight",
        config.seed + 1000000, actor=actor, device=device)
    final_offset = evaluate_policy(
        "ppo", 20, "offset_tail_chase", "straight", config.seed + 1000000,
        actor=actor, device=device)
    final_pursuit = evaluate_policy(
        "ppo", 20, "randomized_tail_chase", "pursuit",
        config.seed + 1000000, actor=actor, device=device)
    summary = {
        "output_dir": str(output_dir.resolve()),
        "global_step": global_step,
        "updates": update_index,
        "elapsed_seconds": time.perf_counter() - start_time,
        "zero_hit_rate": zero_result["red_hit_rate"],
        "random_hit_rate": random_result["red_hit_rate"],
        "pursuit_rule_hit_rate": rule_result["red_hit_rate"],
        "initial_ppo_hit_rate": initial_ppo["red_hit_rate"],
        "final_ppo_hit_rate": final_fixed["red_hit_rate"],
        "randomized_final_hit_rate": final_randomized["red_hit_rate"],
        "offset_final_hit_rate": final_offset["red_hit_rate"],
        "pursuit_opponent_final_hit_rate": final_pursuit["red_hit_rate"],
        "baseline_results": {
            "zero": zero_result, "random": random_result,
            "pursuit_rule": rule_result, "initial_ppo": initial_ppo,
        },
        "final_results": {
            "fixed_straight": final_fixed,
            "randomized_straight": final_randomized,
            "offset_straight": final_offset,
            "randomized_pursuit": final_pursuit,
        },
    }
    summary["learnability_passed"] = bool(
        sum(result["numerical_invalid"] for result in summary[
            "final_results"].values()) == 0
        and final_fixed["red_hit_rate"] >= 0.8
        and final_randomized["red_hit_rate"] >= 0.6
        and final_fixed["red_hit_rate"] - random_result["red_hit_rate"] >= 0.4)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
