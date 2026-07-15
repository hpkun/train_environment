"""Deterministic evaluation and baseline comparison for vanilla HAPPO."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo_checkpoint import load_vanilla_happo_checkpoint
from scripts.vanilla_happo_runtime import deterministic_evaluate, infer_policy, make_paper_env


def main():
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
    args = parser.parse_args()
    import torch
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    env = make_paper_env(ROOT, args.scenario)
    policy, _, _ = infer_policy(env, "independent", 128, device)
    if args.baseline == "trained_happo":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for trained_happo")
        load_vanilla_happo_checkpoint(ROOT / args.checkpoint, policy, restore_rng=False)
    result = deterministic_evaluate(env, policy, args.episodes, args.seed, args.baseline)
    env.close()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
