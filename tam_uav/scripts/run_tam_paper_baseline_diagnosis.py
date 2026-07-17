"""Run fixed-seed TAM paper-environment baseline evaluations sequentially."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import (
    deterministic_evaluate, infer_policy, make_paper_env, seed_all)


BASELINES = ("neutral", "random", "rule", "untrained_happo")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("2v2", "3v2", "5v4"), default="2v2")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--perturbation", choices=("none", "low", "medium", "large"),
                        default="none",
                        help=("none: nominal Table 5-7 experiment; "
                              "low/medium/large: 5v4 generalization experiment"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-directory",
        default="outputs/environment_diagnosis_2v2_seed120000_50ep_v2")
    return parser.parse_args(argv)


def summarize(result):
    episodes = result["episodes_detail"]
    winners = Counter(item["winner"] for item in episodes)
    termination_reasons = Counter(item["termination_reason"] for item in episodes)
    returns = np.asarray([item["red_episode_return"] for item in episodes], dtype=float)
    survival = np.asarray([item["red_survival_rate"] for item in episodes], dtype=float)
    structural = np.asarray(
        [item["red_structural_failures"] for item in episodes], dtype=float)
    crashes = np.asarray([item["red_crashes"] for item in episodes], dtype=float)
    boundary = np.asarray([item["red_boundary"] for item in episodes], dtype=float)
    steps = np.asarray([item["episode_steps"] for item in episodes], dtype=float)
    missiles = int(sum(item["red_missiles_fired"] for item in episodes))
    hits = int(sum(item["red_hits"] for item in episodes))
    return {
        "episodes": len(episodes),
        "win_rate": float(winners["red"] / len(episodes)),
        "winner_counts": {key: int(winners[key]) for key in ("red", "blue", "draw")},
        "red_episode_return_mean": float(returns.mean()),
        "red_episode_return_std": float(returns.std()),
        "red_survival_rate_mean": float(survival.mean()),
        "red_missiles_fired_total": missiles,
        "red_hits_total": hits,
        "red_hit_rate": float(hits / missiles) if missiles else 0.0,
        "red_structural_failures_mean": float(structural.mean()),
        "red_crashes_mean": float(crashes.mean()),
        "red_boundary_mean": float(boundary.mean()),
        "episode_steps_mean": float(steps.mean()),
        "termination_reason_counts": dict(sorted(termination_reasons.items())),
        "target_consistency_violations_total": int(sum(
            item["target_consistency_violations"] for item in episodes)),
        "all_finite": bool(all(item["finite"] for item in episodes)),
    }


def summary_text(summary):
    lines = []
    for baseline in BASELINES:
        item = summary[baseline]
        lines.extend([
            f"[{baseline}]",
            f"episodes: {item['episodes']}",
            f"win_rate: {item['win_rate']:.6f}",
            f"winner_counts: {json.dumps(item['winner_counts'], sort_keys=True)}",
            f"red_episode_return: {item['red_episode_return_mean']:.6f} "
            f"+/- {item['red_episode_return_std']:.6f}",
            f"red_survival_rate: {item['red_survival_rate_mean']:.6f}",
            f"red_missiles_fired: {item['red_missiles_fired_total']}",
            f"red_hits: {item['red_hits_total']}",
            f"red_hit_rate: {item['red_hit_rate']:.6f}",
            f"red_structural_failures: {item['red_structural_failures_mean']:.6f}",
            f"red_crashes: {item['red_crashes_mean']:.6f}",
            f"red_boundary: {item['red_boundary_mean']:.6f}",
            f"episode_steps: {item['episode_steps_mean']:.6f}",
            f"termination_reasons: "
            f"{json.dumps(item['termination_reason_counts'], sort_keys=True)}",
            f"target_consistency_violations: "
            f"{item['target_consistency_violations_total']}",
            f"all_finite: {item['all_finite']}",
            "",
        ])
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    output = resolve_tam_output(ROOT, args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for baseline in BASELINES:
        env = None
        try:
            seed_all(args.seed)
            env = make_paper_env(
                ROOT, args.scenario, initial_perturbation=args.perturbation)
            policy = None
            if baseline == "untrained_happo":
                policy, _, _ = infer_policy(env, "independent", 128, device)
            result = deterministic_evaluate(
                env, policy, args.episodes, args.seed, baseline)
            result["scenario"] = args.scenario
            result["seed"] = args.seed
            result["perturbation"] = args.perturbation
            (output / f"{baseline}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
            results[baseline] = result
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            if env is not None:
                env.close()
    summary = {baseline: summarize(results[baseline]) for baseline in BASELINES}
    (output / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (output / "baseline_summary.txt").write_text(
        summary_text(summary), encoding="utf-8")
    (output / "DONE").write_text("", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
