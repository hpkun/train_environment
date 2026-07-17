"""Evaluate one checkpoint on all four required 1v1 scenarios."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for scenario, opponent in (
        ("fixed_tail_chase", "paper_greedy"),
        ("randomized_tail_chase", "paper_greedy"),
        ("offset_tail_chase", "paper_greedy"),
        ("randomized_tail_chase", "pursuit"),
    ):
        rows.append(evaluate_policy(
            "ppo", args.episodes, scenario, opponent, args.seed,
            actor=actor, device="cpu"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": checkpoint.get("global_step"),
            "results": rows,
        }, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("global_step"),
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
