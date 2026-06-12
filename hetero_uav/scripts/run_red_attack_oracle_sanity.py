"""Run scripted red attack oracle sanity checks without training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from red_attack_audit_utils import (
    DEFAULT_CONFIG,
    alive_counts,
    blue_actions,
    collect_step_counts,
    make_env,
    red_oracle_actions,
    team_done,
    write_json,
    write_md,
)


CASES = [
    ("red_brma_rule_vs_blue_zero", "brma_rule", "zero"),
    ("red_brma_rule_vs_blue_brma_rule", "brma_rule", "brma_rule"),
    ("red_direct_chase_vs_blue_zero", "direct_chase", "zero"),
    ("red_direct_chase_vs_blue_brma_rule", "direct_chase", "brma_rule"),
]


def _winner(env, terminated: dict, truncated: dict) -> str:
    counts = alive_counts(env)
    if counts["blue_alive"] == 0 and counts["red_alive"] > 0:
        return "red"
    if counts["red_alive"] == 0 and counts["blue_alive"] > 0:
        return "blue"
    if all(truncated.values()):
        if counts["red_alive"] > counts["blue_alive"]:
            return "red_timeout_alive_advantage"
        if counts["blue_alive"] > counts["red_alive"]:
            return "blue_timeout_alive_advantage"
        return "draw"
    return "unknown"


def run_case(config: str, name: str, red_mode: str, blue_mode: str, episodes: int, max_steps: int) -> dict:
    episode_records = []
    for ep in range(episodes):
        env = make_env(config)
        try:
            obs, _info = env.reset(seed=ep)
            red_fired = blue_fired = 0
            first_red = first_blue = None
            prev_red_hits = prev_blue_hits = 0
            red_hits = blue_hits = 0
            terminated = truncated = {}
            for step in range(1, max_steps + 1):
                actions = {}
                actions.update(red_oracle_actions(env, obs, red_mode))
                actions.update(blue_actions(env, obs, blue_mode))
                obs, _rewards, terminated, truncated, info = env.step(actions)
                counts = collect_step_counts(info)
                if counts["red_fired"] and first_red is None:
                    first_red = step
                if counts["blue_fired"] and first_blue is None:
                    first_blue = step
                red_fired += counts["red_fired"]
                blue_fired += counts["blue_fired"]
                red_hits = max(red_hits, counts["red_hits_total"])
                blue_hits = max(blue_hits, counts["blue_hits_total"])
                if team_done(terminated, truncated):
                    break
            counts_alive = alive_counts(env)
            episode_records.append({
                "steps": step,
                "red_fired": red_fired,
                "blue_fired": blue_fired,
                "red_hits": red_hits,
                "blue_hits": blue_hits,
                "first_red_fire_step": first_red,
                "first_blue_fire_step": first_blue,
                "winner": _winner(env, terminated, truncated),
                **counts_alive,
            })
        finally:
            env.close()

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in episode_records]
        return float(np.mean(vals)) if vals else 0.0

    def rate(predicate) -> float:
        return float(np.mean([1.0 if predicate(r) else 0.0 for r in episode_records])) if episode_records else 0.0

    first_red_vals = [r["first_red_fire_step"] for r in episode_records if r["first_red_fire_step"] is not None]
    first_blue_vals = [r["first_blue_fire_step"] for r in episode_records if r["first_blue_fire_step"] is not None]
    red_fire_possible = rate(lambda r: r["red_fired"] > 0)
    red_hit_rate = rate(lambda r: r["red_hits"] > 0)
    if red_fire_possible == 0.0:
        conclusion = "red_oracle_cannot_fire"
    elif red_hit_rate == 0.0:
        conclusion = "red_oracle_can_fire_but_not_hit"
    else:
        conclusion = "red_oracle_can_fire_and_hit"

    return {
        "case": name,
        "red_policy": red_mode,
        "blue_policy": blue_mode,
        "episodes": episodes,
        "max_steps": max_steps,
        "red_missiles_fired_mean": mean("red_fired"),
        "blue_missiles_fired_mean": mean("blue_fired"),
        "red_missile_hits_mean": mean("red_hits"),
        "blue_missile_hits_mean": mean("blue_hits"),
        "blue_dead_mean": mean("blue_dead"),
        "red_dead_mean": mean("red_dead"),
        "red_win_rate": rate(lambda r: r["winner"].startswith("red")),
        "blue_win_rate": rate(lambda r: r["winner"].startswith("blue")),
        "timeout_rate": rate(lambda r: r["steps"] >= max_steps),
        "red_alive_final_mean": mean("red_alive"),
        "blue_alive_final_mean": mean("blue_alive"),
        "first_red_fire_step_mean": float(np.mean(first_red_vals)) if first_red_vals else None,
        "first_blue_fire_step_mean": float(np.mean(first_blue_vals)) if first_blue_vals else None,
        "red_fire_possible_rate": red_fire_possible,
        "conclusion": conclusion,
        "episodes_detail": episode_records,
    }


def build_report(config: str, episodes: int, max_steps: int) -> dict:
    records = [run_case(config, *case, episodes=episodes, max_steps=max_steps) for case in CASES]
    if all(r["red_fire_possible_rate"] == 0.0 for r in records):
        overall = "scripted_red_oracle_cannot_fire_environment_chain_suspect"
    elif any(r["red_missile_hits_mean"] > 0.0 for r in records):
        overall = "scripted_red_oracle_can_fire_and_hit"
    else:
        overall = "scripted_red_oracle_can_fire_but_hit_not_observed"
    return {
        "config": config,
        "episodes": episodes,
        "max_steps": max_steps,
        "cases": records,
        "overall_conclusion": overall,
    }


def write_report_md(data: dict, output_md: str) -> None:
    lines = ["# Red Attack Oracle Sanity", "", f"- config: `{data['config']}`", ""]
    for rec in data["cases"]:
        lines.extend([
            f"## {rec['case']}",
            f"- red_missiles_fired_mean: {rec['red_missiles_fired_mean']}",
            f"- red_missile_hits_mean: {rec['red_missile_hits_mean']}",
            f"- blue_dead_mean: {rec['blue_dead_mean']}",
            f"- first_red_fire_step_mean: {rec['first_red_fire_step_mean']}",
            f"- conclusion: {rec['conclusion']}",
            "",
        ])
    lines.append(f"overall_conclusion: {data['overall_conclusion']}")
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scripted red attack oracle sanity checks")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/red_attack_oracle/red_attack_oracle_sanity.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/red_attack_oracle/red_attack_oracle_sanity.md",
    )
    args = parser.parse_args()
    data = build_report(args.config, args.episodes, args.max_steps)
    out_json = write_json(args.output_json, data)
    write_report_md(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"overall_conclusion: {data['overall_conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

