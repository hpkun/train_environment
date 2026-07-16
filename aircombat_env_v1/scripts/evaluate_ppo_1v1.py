"""Evaluate a PPO checkpoint in one 1v1 scenario."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

for _name, _value in (
    ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE"),
):
    os.environ.setdefault(_name, _value)

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aircombat_env_v1.evaluation import evaluate_policy
from aircombat_env_v1.ppo import Actor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scenario", default="fixed_tail_chase")
    parser.add_argument("--opponent", default="straight")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    result = evaluate_policy(
        "ppo", args.episodes, args.scenario, args.opponent, args.seed,
        args.stochastic, actor, "cpu")
    output = Path(args.output) if args.output else (
        Path(args.checkpoint).with_suffix("").parent
        / f"evaluation_{args.scenario}_{args.opponent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output if output.suffix == ".json" else output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
