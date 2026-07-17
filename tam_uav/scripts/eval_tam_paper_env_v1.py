"""Evaluate one nominal TAM paper scenario with the rule baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import deterministic_evaluate, make_paper_env
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL, protocol_metadata,
    validate_nominal_protocol)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("2v2", "3v2", "5v4"), default="3v2")
    parser.add_argument(
        "--episodes", type=int, default=1,
        help=("default 1 because nominal rule evaluation repeats deterministically; "
              "set explicitly for additional episodes"))
    parser.add_argument("--backend", choices=("simple", "jsbsim"), default="jsbsim")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="outputs/tam_paper_env_v1_evaluation.json")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    validate_nominal_protocol(args.scenario, NOMINAL_PERTURBATION)
    env = make_paper_env(
        ROOT, args.scenario, initial_perturbation=NOMINAL_PERTURBATION,
        dynamics_backend=args.backend, experiment_protocol=PAPER_NOMINAL_PROTOCOL)
    try:
        result = deterministic_evaluate(
            env, None, args.episodes, args.seed, baseline="rule")
    finally:
        env.close()
    result.update(protocol_metadata(
        args.scenario, NOMINAL_PERTURBATION, args.backend, PAPER_NOMINAL_PROTOCOL))
    result["evaluated_environment_fidelity_revision"] = ENVIRONMENT_FIDELITY_REVISION
    result["evaluated_experiment_protocol"] = PAPER_NOMINAL_PROTOCOL
    result["algorithm_label"] = "rule_baseline"
    output = resolve_tam_output(ROOT, args.output, create_parent=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
