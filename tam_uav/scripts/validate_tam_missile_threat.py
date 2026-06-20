"""Collect missile-threat statistics without modifying combat behavior."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from scripts.diagnose_tam_mav_policy_drift import _load_checkpoint_policy, _run_episode
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


def _bin_label(value, bounds, labels):
    for upper, label in zip(bounds, labels):
        if value < upper:
            return label
    return labels[-1]


def _geometry_summary(launches, successful_ids):
    specs = {
        "range_m": ([3000.0, 6000.0], ["<3000", "3000-6000", ">=6000"]),
        "AO_deg": ([30.0, 60.0], ["<30", "30-60", ">=60"]),
        "TA_deg": ([60.0, 120.0], ["<60", "60-120", ">=120"]),
        "shooter_speed_mps": ([200.0, 300.0], ["<200", "200-300", ">=300"]),
    }
    result = {}
    for field, (bounds, labels) in specs.items():
        counts = {label: {"launches": 0, "hits": 0, "hit_rate": 0.0} for label in labels}
        for launch in launches:
            value = launch.get(field)
            if value in (None, ""):
                continue
            label = _bin_label(float(value), bounds, labels)
            counts[label]["launches"] += 1
            counts[label]["hits"] += int(str(launch.get("missile_id")) in successful_ids)
        for values in counts.values():
            values["hit_rate"] = (
                values["hits"] / values["launches"] if values["launches"] else 0.0
            )
        result[field] = counts
    return result


def summarize_missile_threat(episodes):
    records = [record for episode in episodes for record in episode.get("launch_quality", [])]
    launches = [record for record in records if not record.get("termination_reason")]
    completed = [record for record in records if record.get("termination_reason")]
    successful_ids = {
        str(record.get("missile_id")) for record in completed if record.get("is_success")
    }
    red_launches = [record for record in launches if record.get("shooter_team", record.get("team")) == "red"]
    blue_launches = [record for record in launches if record.get("shooter_team", record.get("team")) == "blue"]
    red_completed = [record for record in completed if record.get("shooter_team", record.get("team")) == "red"]
    blue_completed = [record for record in completed if record.get("shooter_team", record.get("team")) == "blue"]
    warning_count = sum(
        bool(row.get("missile_warning"))
        for episode in episodes for row in episode.get("trace", [])
    )
    warning_rates = {}
    for seconds in (5, 10, 20):
        warned = 0
        hit_after_warning = 0
        horizon_steps = seconds * 5
        for episode in episodes:
            warning_steps = [
                int(row["step"]) for row in episode.get("trace", [])
                if row.get("missile_warning")
            ]
            if not warning_steps:
                continue
            warned += 1
            death_step = int(episode.get("death_step", -1))
            if (
                episode.get("death_reason") == "Missile_Kill"
                and death_step > 0
                and any(0 <= death_step - step <= horizon_steps for step in warning_steps)
            ):
                hit_after_warning += 1
        warning_rates[f"{seconds}s"] = hit_after_warning / warned if warned else 0.0
    post_launch_survival = []
    for episode in episodes:
        death_step = int(episode.get("death_step", -1))
        if death_step <= 0:
            continue
        for launch in episode.get("launch_quality", []):
            if (
                not launch.get("termination_reason")
                and launch.get("shooter_team", launch.get("team")) == "blue"
                and launch.get("launch_step") not in (None, "")
                and death_step >= int(launch["launch_step"])
            ):
                post_launch_survival.append((death_step - int(launch["launch_step"])) / 5.0)
    no_launch_opportunities = sum(
        any(float(row.get("nearest_blue_range_m", float("inf"))) <= 10000.0 for row in episode.get("trace", []))
        and not any(
            not record.get("termination_reason")
            and record.get("shooter_team", record.get("team")) == "red"
            for record in episode.get("launch_quality", [])
        )
        for episode in episodes
    )
    red_hits = sum(bool(record.get("is_success")) for record in red_completed)
    blue_hits = sum(bool(record.get("is_success")) for record in blue_completed)
    return {
        "episodes": len(episodes),
        "red_launch_count": len(red_launches),
        "red_hit_count": red_hits,
        "red_hit_rate": red_hits / len(red_launches) if red_launches else 0.0,
        "blue_launch_count": len(blue_launches),
        "blue_hit_count": blue_hits,
        "blue_hit_rate": blue_hits / len(blue_launches) if blue_launches else 0.0,
        "missile_warning_count": int(warning_count),
        "warning_to_hit_rate": warning_rates,
        "blue_launch_to_red_death_time_sec_mean": (
            float(np.mean(post_launch_survival)) if post_launch_survival else None
        ),
        "mav_death_reasons": dict(Counter(
            str(episode.get("death_reason", "missing")) for episode in episodes
        )),
        "mav_death_time_sec_mean": float(np.mean([
            episode["death_step"] / 5.0 for episode in episodes
            if int(episode.get("death_step", -1)) > 0
        ])) if any(int(episode.get("death_step", -1)) > 0 for episode in episodes) else None,
        "red_launch_opportunity_without_launch_episodes": int(no_launch_opportunities),
        "red_zero_hit_termination_reasons": dict(Counter(
            str(record.get("termination_reason")) for record in red_completed
            if not record.get("is_success")
        )) if red_hits == 0 else {},
        "geometry_hit_rate": _geometry_summary(launches, successful_ids),
    }


def run_audit(config, output_dir, episodes=10, max_steps=1000,
              checkpoint=None, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = (
        _load_checkpoint_policy(checkpoint, device)
        if checkpoint else TAMCategoricalRecurrentHAPPOPolicy().to(device)
    )
    policy.eval()
    adapter = HeteroObsAdapterV2()
    specs = {
        "deterministic": (True, False),
        "stochastic": (False, False),
        "no_blue_missile_debug": (True, True),
        "formal_blue_missile_enabled": (True, False),
    }
    scenarios = {}
    for name, (deterministic, no_blue_missile) in specs.items():
        records = [
            _run_episode(
                config, policy, adapter, deterministic=deterministic,
                no_blue_missile=no_blue_missile, max_steps=max_steps,
                seed=seed + episode, device=device,
            )
            for episode in range(episodes)
        ]
        scenarios[name] = {
            "summary": summarize_missile_threat(records),
            "episodes": [{
                "seed": record["seed"],
                "death_step": record["death_step"],
                "death_reason": record["death_reason"],
                "survived": record["survived"],
            } for record in records],
        }
    result = {
        "config": config,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "episodes_per_scenario": episodes,
        "max_steps": max_steps,
        "scenarios": scenarios,
    }
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tam_missile_threat.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = ["# TAM Missile Threat Audit", ""]
    for name, scenario in scenarios.items():
        summary = scenario["summary"]
        lines.extend([
            f"## {name}", "",
            f"- Red launch/hit: {summary['red_launch_count']}/{summary['red_hit_count']}",
            f"- Blue launch/hit: {summary['blue_launch_count']}/{summary['blue_hit_count']}",
            f"- Warning count: {summary['missile_warning_count']}",
            f"- Warning hit rates: {summary['warning_to_hit_rate']}",
            f"- MAV deaths: {summary['mav_death_reasons']}",
            f"- Red opportunity without launch episodes: {summary['red_launch_opportunity_without_launch_episodes']}",
            f"- Red zero-hit terminations: {summary['red_zero_hit_termination_reasons']}", "",
        ])
    (out_dir / "tam_missile_threat.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_audit(
        args.config, args.output_dir, episodes=args.episodes,
        max_steps=args.max_steps, checkpoint=args.checkpoint,
        device=args.device, seed=args.seed,
    )
    try:
        print(json.dumps({
            name: scenario["summary"] for name, scenario in result["scenarios"].items()
        }, indent=2))
    except OSError:
        # Some Windows JSBSim builds close the inherited stdout descriptor
        # after many simulator lifecycles. Reports are already persisted.
        pass


if __name__ == "__main__":
    main()
