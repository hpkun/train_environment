"""Deterministic evaluation and baseline comparison for vanilla HAPPO."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, read_vanilla_happo_checkpoint_metadata)
from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import (deterministic_evaluate, infer_policy,
                                           make_paper_env, seed_all)
from uav_env.JSBSim.paper.protocol import (
    NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL, protocol_metadata,
    validate_nominal_protocol)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("2v2", "3v2", "5v4"), default="2v2")
    parser.add_argument("--baseline", choices=("neutral", "random", "rule",
                                               "untrained_happo", "trained_happo"),
                        default="trained_happo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/tam_paper_vanilla_happo/evaluation.json")
    parser.add_argument("--perturbation", choices=(NOMINAL_PERTURBATION,),
                        default=NOMINAL_PERTURBATION,
                        help="paper_nominal is fixed to the unperturbed Table 5-7 state")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    validate_nominal_protocol(args.scenario, args.perturbation)
    import torch
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    env = make_paper_env(
        ROOT, args.scenario, initial_perturbation=args.perturbation)
    metadata = None
    if args.baseline == "trained_happo":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for trained_happo")
        checkpoint = resolve_tam_output(ROOT, args.checkpoint)
        metadata = read_vanilla_happo_checkpoint_metadata(checkpoint)
        if metadata["scenario"] != args.scenario:
            raise ValueError("checkpoint scenario does not match evaluation scenario")
        policy, obs_dim, state_dim = infer_policy(
            env, metadata["actor_sharing"], metadata["hidden_dim"], device)
        if obs_dim != metadata["actor_obs_dim"] or state_dim != metadata["critic_state_dim"]:
            raise ValueError("checkpoint dimensions do not match evaluation environment")
        load_vanilla_happo_checkpoint(
            checkpoint, policy, restore_rng=False, expected_scenario=args.scenario)
    else:
        seed_all(args.seed)
        policy, _, _ = infer_policy(env, "independent", 128, device)
    result = deterministic_evaluate(env, policy, args.episodes, args.seed, args.baseline)
    result["algorithm_label"] = (metadata["algorithm_mode"] if metadata
                                 else "untrained_or_nonlearning_baseline")
    result["formal_happo"] = bool(metadata and metadata["actor_sharing"] == "independent")
    result.update(protocol_metadata(
        args.scenario, args.perturbation, "jsbsim", PAPER_NOMINAL_PROTOCOL))
    env.close()
    output = resolve_tam_output(ROOT, args.output, create_parent=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
