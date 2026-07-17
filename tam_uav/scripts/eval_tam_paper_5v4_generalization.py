"""Evaluate a vanilla-HAPPO baseline checkpoint; this is not TAM-HAPPO."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, read_vanilla_happo_checkpoint_metadata)
from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import deterministic_evaluate, infer_policy, make_paper_env
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, GENERALIZATION_EPISODES_PER_LEVEL,
    GENERALIZATION_PERTURBATION_LEVELS, PAPER_5V4_GENERALIZATION_PROTOCOL,
    PAPER_NOMINAL_PROTOCOL, PAPER_SILENT_ASSUMPTIONS_PRESENT, checkpoint_lineage,
    validate_generalization_protocol)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-per-level", type=int,
                        default=GENERALIZATION_EPISODES_PER_LEVEL)
    parser.add_argument("--seed", type=int, default=42026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/tam_paper_5v4_generalization.json")
    return parser.parse_args(argv)


def summarize_level(result):
    episodes = result["episodes_detail"]
    missiles = int(sum(item["red_missiles_fired"] for item in episodes))
    hits = int(sum(item["red_hits"] for item in episodes))
    rewards = np.asarray([item["red_episode_return"] for item in episodes], dtype=float)
    return {
        "episodes": len(episodes),
        "episode_seeds": result["episode_seeds"],
        "win_rate": float(np.mean([item["winner"] == "red" for item in episodes])),
        "average_cumulative_team_reward": float(rewards.mean()),
        "cumulative_team_reward_std": float(rewards.std()),
        "red_survival_rate": float(np.mean(
            [item["red_survival_rate"] for item in episodes])),
        "red_hit_rate": float(hits / missiles) if missiles else 0.0,
        "red_missiles_fired_total": missiles,
        "red_hits_total": hits,
        "average_episode_steps": float(np.mean(
            [item["episode_steps"] for item in episodes])),
        "winner_counts": dict(Counter(item["winner"] for item in episodes)),
        "termination_reason_counts": dict(Counter(
            item["termination_reason"] for item in episodes)),
        "target_consistency_violations": int(sum(
            item["target_consistency_violations"] for item in episodes)),
        "structural_failures": int(sum(
            item["red_structural_failures"] + item["blue_structural_failures"]
            for item in episodes)),
        "all_finite": bool(all(item["finite"] for item in episodes)),
    }


def main():
    args = parse_args()
    if args.episodes_per_level <= 0:
        raise ValueError("--episodes-per-level must be positive")
    checkpoint = resolve_tam_output(ROOT, args.checkpoint)
    metadata = read_vanilla_happo_checkpoint_metadata(checkpoint)
    if metadata.get("scenario") != "5v4":
        raise ValueError("paper_5v4_generalization requires a 5v4 checkpoint")
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    level_results = {}
    for level_index, level in enumerate(GENERALIZATION_PERTURBATION_LEVELS):
        validate_generalization_protocol("5v4", level)
        env = make_paper_env(
            ROOT, "5v4", initial_perturbation=level,
            experiment_protocol=PAPER_5V4_GENERALIZATION_PROTOCOL)
        try:
            policy, obs_dim, state_dim = infer_policy(
                env, metadata["actor_sharing"], metadata["hidden_dim"], device)
            if obs_dim != metadata["actor_obs_dim"] or state_dim != metadata["critic_state_dim"]:
                raise ValueError("checkpoint dimensions do not match 5v4 environment")
            load_vanilla_happo_checkpoint(
                checkpoint, policy, restore_rng=False, expected_scenario="5v4",
                expected_environment_fidelity_revision=ENVIRONMENT_FIDELITY_REVISION,
                expected_experiment_protocol=PAPER_NOMINAL_PROTOCOL,
                expected_initial_perturbation="none",
                expected_dynamics_backend="jsbsim")
            episode_seeds = [
                args.seed + level_index * 10000 + episode
                for episode in range(args.episodes_per_level)]
            result = deterministic_evaluate(
                env, policy, args.episodes_per_level, args.seed,
                baseline="trained_happo", episode_seeds=episode_seeds)
            level_results[level] = summarize_level(result)
        finally:
            env.close()
    output_data = {
        "experiment_protocol": PAPER_5V4_GENERALIZATION_PROTOCOL,
        "scenario": "5v4",
        "perturbation_levels": list(GENERALIZATION_PERTURBATION_LEVELS),
        "episodes_per_level": args.episodes_per_level,
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "initial_perturbation": list(GENERALIZATION_PERTURBATION_LEVELS),
        "dynamics_backend": "jsbsim",
        "paper_nominal_experiment": False,
        "paper_generalization_experiment": True,
        "paper_silent_assumptions_present": PAPER_SILENT_ASSUMPTIONS_PRESENT,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "actor_sharing": metadata["actor_sharing"],
        "algorithm_label": ("vanilla_happo" if metadata["actor_sharing"] == "independent"
                            else "parameter_sharing_ppo_ablation"),
        "evaluated_environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "evaluated_experiment_protocol": PAPER_5V4_GENERALIZATION_PROTOCOL,
        "levels": level_results,
    } | checkpoint_lineage(metadata)
    output = resolve_tam_output(ROOT, args.output, create_parent=True)
    output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
