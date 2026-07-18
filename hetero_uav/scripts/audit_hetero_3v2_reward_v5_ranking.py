"""Cross-reward ranking audit for the formal V5 shared team reward.

This script evaluates actor parameters only. It never restores a critic,
optimizer, rollout buffer, or training state from a different reward contract.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy
from scripts.hetero_3v2_v2_audit_common import perturbation
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from uav_env.JSBSim.formal_v2.contract import V5_REWARD_CONTRACT_VERSION
from uav_env.JSBSim.formal_v2.reward_v5 import (
    POTENTIAL_GAMMA,
    discounted_potential_return,
    theoretical_discounted_potential_return,
)

RULE_POLICY_LABEL = "rule_red_policy"


def _undiscounted_task_return(event_return: float, terminal_return: float) -> float:
    return float(event_return) + float(terminal_return)


def _red_combat_counts(events: list[dict], red_kills: int) -> dict[str, int]:
    return {
        "red_launches": sum(
            event.get("event") == "launch"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events),
        "red_hits": sum(
            event.get("event") == "hit"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events),
        "red_kills": int(red_kills),
    }


def _checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label, path


def _paired_scenarios(seed: int, episodes: int) -> list[dict]:
    scenarios = [
        {"seed": seed + episode,
         "perturbation": perturbation(seed + episode)}
        for episode in range(episodes)
    ]
    _validate_scenario_uniqueness(scenarios)
    return scenarios


def _validate_scenario_uniqueness(scenarios: list[dict]) -> int:
    signatures = [
        json.dumps(row["perturbation"], sort_keys=True, separators=(",", ":"))
        for row in scenarios
    ]
    unique_count = len(set(signatures))
    if unique_count != len(scenarios):
        raise ValueError(
            "audit perturbations must be unique: "
            f"episodes={len(scenarios)}, unique_perturbations={unique_count}")
    return unique_count


def _validate_actor_contract(meta: dict) -> None:
    expected = {
        "formal_contract": "hetero_3v2_pure_happo_v2",
        "policy_arch": "pure_happo",
        "algorithm_contract": ALGORITHM_CONTRACT,
        "actor_obs_dim": 73,
        "action_dim": 3,
        "num_agents": 3,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"cross-reward actor contract mismatch {key}: "
                f"{meta.get(key)!r} != {value!r}")


def _load_actor_only(
    checkpoint: Path, device: torch.device,
) -> tuple[PureHAPPOPolicy, dict]:
    meta_path = checkpoint.parent / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"checkpoint meta is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _validate_actor_contract(meta)
    policy = PureHAPPOPolicy(
        actor_obs_dim=73,
        critic_state_dim=219,
        action_dim=3,
        num_agents=3,
        credit_mode="fixed_three_agent_team_mean",
    ).to(device)
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    actor_state = {
        key: value for key, value in source.items()
        if key.startswith("actors.") or key.startswith("action_log_stds.")
    }
    required = {
        key for key in policy.state_dict()
        if key.startswith("actors.") or key.startswith("action_log_stds.")
    }
    if set(actor_state) != required:
        missing = sorted(required - set(actor_state))
        unexpected = sorted(set(actor_state) - required)
        raise ValueError(
            f"incompatible actor state; missing={missing}, unexpected={unexpected}")
    policy.load_state_dict(actor_state, strict=False)
    policy.eval()
    return policy, meta


def _rollout(
    env, label: str, scenarios: list[dict],
    policy: PureHAPPOPolicy | None, rule: PaperGreedyOpponent | None,
) -> dict:
    rows = []
    for episode, scenario in enumerate(scenarios):
        obs, info = env.reset(
            seed=scenario["seed"],
            options={"audit_initial_perturbation": scenario["perturbation"]},
        )
        initial_phi = float(env.v5_phi_previous)
        sums = {
            "undiscounted_shared_event_return": 0.0,
            "undiscounted_terminal_return": 0.0,
            "undiscounted_potential_return": 0.0,
            "undiscounted_team_return": 0.0,
            "discounted_shared_event_return": 0.0,
            "discounted_terminal_return": 0.0,
            "discounted_team_return": 0.0,
            "phi_mav": 0.0,
            "phi_uav_1": 0.0,
            "phi_uav_2": 0.0,
        }
        potential_rewards = []
        red_kills = 0
        steps = 0
        terminated = truncated = False
        while True:
            if rule is not None:
                actions = rule.actions(env, "red")
            else:
                actor_obs = np.stack([
                    obs[agent_id]["flat"] for agent_id in env.red_ids])
                with torch.no_grad():
                    result = policy.act(actor_obs, deterministic=True)
                action_array = result["action"].cpu().numpy()
                action_array *= np.asarray(info["active_mask"])[:, None]
                actions = {
                    agent_id: action_array[index]
                    for index, agent_id in enumerate(env.red_ids)
                }
            obs, rewards, terminations, truncations, info = env.step(actions)
            components = info["reward_components"]
            discount = POTENTIAL_GAMMA ** steps
            event_reward = float(components["shared_event_reward"])
            terminal_reward = float(components["terminal_reward"])
            potential_reward = float(components["potential_shaping_reward"])
            team_reward = float(components["team_reward"])
            sums["undiscounted_shared_event_return"] += event_reward
            sums["undiscounted_terminal_return"] += terminal_reward
            sums["undiscounted_potential_return"] += potential_reward
            sums["undiscounted_team_return"] += team_reward
            sums["discounted_shared_event_return"] += discount * event_reward
            sums["discounted_terminal_return"] += discount * terminal_reward
            sums["discounted_team_return"] += discount * team_reward
            potential_rewards.append(potential_reward)
            red_kills += int(components["red_kill_count"])
            for key in ("phi_mav", "phi_uav_1", "phi_uav_2"):
                sums[key] += components[key]
            if not np.allclose(list(rewards.values()), components["team_reward"]):
                raise ValueError("V5 audit observed non-shared red rewards")
            steps += 1
            terminated = bool(any(terminations.values()))
            truncated = bool(any(truncations.values()))
            if terminated or truncated:
                break
        events = env.event_log
        red_counts = _red_combat_counts(events, red_kills)
        discounted_potential = discounted_potential_return(potential_rewards)
        final_phi_effective = float(components["phi_team_next_effective"])
        theoretical_potential = theoretical_discounted_potential_return(
            initial_phi, final_phi_effective, steps)
        row = {
            "episode": episode,
            "seed": scenario["seed"],
            "initial_perturbation": scenario["perturbation"],
            "outcome": info["outcome"],
            **red_counts,
            "mav_survived": int(info["mav_alive"]),
            "red_attack_survival": info["red_attack_alive"] / 2.0,
            "red_attack_alive_final": int(info["red_attack_alive"]),
            "blue_attack_alive_final": int(info["blue_attack_alive"]),
            "episode_length": steps,
            "phi_initial": initial_phi,
            "phi_final": float(components["phi_team_next"]),
            "phi_final_effective": final_phi_effective,
            "terminated": terminated,
            "truncated": truncated,
            **sums,
            "undiscounted_task_return": _undiscounted_task_return(
                sums["undiscounted_shared_event_return"],
                sums["undiscounted_terminal_return"]),
            "discounted_potential_return": discounted_potential,
            "theoretical_discounted_potential_return": theoretical_potential,
            "potential_telescoping_error": (
                discounted_potential - theoretical_potential),
            "phi_mav_mean": sums["phi_mav"] / steps,
            "phi_uav_1_mean": sums["phi_uav_1"] / steps,
            "phi_uav_2_mean": sums["phi_uav_2"] / steps,
        }
        if not all(np.isfinite(value) for value in row.values()
                   if isinstance(value, (int, float, np.number))):
            raise ValueError(f"non-finite V5 audit row for {label}")
        rows.append(row)

    mean = lambda key: float(np.mean([row[key] for row in rows]))
    return {
        "label": label,
        "episodes": len(scenarios),
        "red_win_rate": float(np.mean([
            row["outcome"] == "red_win" for row in rows])),
        "blue_win_rate": float(np.mean([
            row["outcome"] == "blue_win" for row in rows])),
        "mutual_elimination_rate": float(np.mean([
            row["outcome"] == "mutual_elimination" for row in rows])),
        "timeout_rate": float(np.mean([
            row["outcome"] == "draw" for row in rows])),
        "red_launches": sum(row["red_launches"] for row in rows),
        "red_hits": sum(row["red_hits"] for row in rows),
        "red_kills": sum(row["red_kills"] for row in rows),
        "mav_survival_rate": mean("mav_survived"),
        "red_attack_uav_survival": mean("red_attack_survival"),
        "episode_length_mean": mean("episode_length"),
        "undiscounted_shared_event_return_mean": mean(
            "undiscounted_shared_event_return"),
        "undiscounted_terminal_return_mean": mean(
            "undiscounted_terminal_return"),
        "undiscounted_potential_return_mean": mean(
            "undiscounted_potential_return"),
        "undiscounted_team_return_mean": mean("undiscounted_team_return"),
        "undiscounted_task_return_mean": mean("undiscounted_task_return"),
        "discounted_shared_event_return_mean": mean(
            "discounted_shared_event_return"),
        "discounted_terminal_return_mean": mean("discounted_terminal_return"),
        "discounted_potential_return_mean": mean(
            "discounted_potential_return"),
        "discounted_team_return_mean": mean("discounted_team_return"),
        "potential_telescoping_error_max_abs": float(max(
            abs(row["potential_telescoping_error"]) for row in rows)),
        "phi_mav_mean": mean("phi_mav_mean"),
        "phi_uav_1_mean": mean("phi_uav_1_mean"),
        "phi_uav_2_mean": mean("phi_uav_2_mean"),
        "finite": True,
        "episode_rows": rows,
    }


def _ranking_checks(results: list[dict]) -> dict:
    by_label = {row["label"]: row for row in results}
    episodes = [
        {**row, "policy_label": result["label"]}
        for result in results for row in result["episode_rows"]
    ]

    def condition_check(rows: list[dict], predicate) -> dict:
        return {
            "sample_count": len(rows),
            "pass": all(predicate(row) for row in rows) if rows else None,
        }

    def monotonic_check(
        varied_key: str,
        fixed_keys: tuple[str, ...],
        higher_value_is_better: bool,
    ) -> dict:
        comparisons = []
        for index, left in enumerate(episodes):
            for right in episodes[index + 1:]:
                if left["seed"] != right["seed"]:
                    continue
                if any(left[key] != right[key] for key in fixed_keys):
                    continue
                if left[varied_key] == right[varied_key]:
                    continue
                better, worse = (
                    (left, right) if (
                        left[varied_key] > right[varied_key]
                    ) == higher_value_is_better
                    else (right, left)
                )
                comparisons.append(
                    better["undiscounted_task_return"]
                    > worse["undiscounted_task_return"])
        return {
            "comparison_count": len(comparisons),
            "pass": all(comparisons) if comparisons else None,
        }

    blue_loss = monotonic_check(
        "blue_attack_alive_final",
        ("outcome", "red_attack_alive_final", "mav_survived"),
        False,
    )
    red_loss = monotonic_check(
        "red_attack_alive_final",
        ("outcome", "blue_attack_alive_final", "mav_survived"),
        True,
    )
    mav_death = monotonic_check(
        "mav_survived",
        ("outcome", "blue_attack_alive_final", "red_attack_alive_final"),
        True,
    )
    attack_100k_not_below_250k = None
    if {"checkpoint_100k_attack", "checkpoint_250k_collapsed"} <= set(by_label):
        attack_100k_not_below_250k = (
            by_label["checkpoint_100k_attack"]["discounted_team_return_mean"]
            >= by_label["checkpoint_250k_collapsed"][
                "discounted_team_return_mean"]
        )
    return {
        "blue_win_without_red_kill_negative": condition_check(
            [row for row in episodes
             if row["outcome"] == "blue_win" and row["red_kills"] == 0],
            lambda row: row["undiscounted_task_return"] < 0,
        ),
        "red_win_positive": condition_check(
            [row for row in episodes if row["outcome"] == "red_win"],
            lambda row: row["undiscounted_task_return"] > 0,
        ),
        "more_blue_losses_raise_return_when_matched": blue_loss,
        "more_red_attack_losses_lower_return_when_matched": red_loss,
        "mav_death_lowers_return_when_matched": mav_death,
        "checkpoint_100k_not_below_250k": attack_100k_not_below_250k,
        "potential_telescoping_within_tolerance": condition_check(
            episodes,
            lambda row: abs(row["potential_telescoping_error"]) <= 1e-6,
        ),
    }


def _observational_ranking(results: list[dict]) -> list[dict]:
    return sorted(
        ({"label": row["label"],
          "discounted_team_return_mean": row[
              "discounted_team_return_mean"]} for row in results),
        key=lambda row: row["discounted_team_return_mean"],
        reverse=True,
    )


def _write_report(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward_v5_ranking_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Formal V5 Cross-Reward Ranking Audit",
        "",
        "This is an actor-only cross-reward audit. It is not checkpoint resume.",
        "",
        "## Policy performance",
        "",
        "Discounted team return is the policy-comparison metric.",
        "",
        "| Policy | Source reward | Episodes | R win | B win | Launch | Hit | Kill | Task return | Discounted return |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sources = payload["source_reward_contracts"]
    for row in payload["results"]:
        lines.append(
            f"| {row['label']} | {sources[row['label']]} | {row['episodes']} | "
            f"{row['red_win_rate']:.3f} | {row['blue_win_rate']:.3f} | "
            f"{row['red_launches']} | {row['red_hits']} | {row['red_kills']} | "
            f"{row['undiscounted_task_return_mean']:.4f} | "
            f"{row['discounted_team_return_mean']:.4f} |")
    lines.extend(["", "Observational ranking by mean discounted return:", ""])
    for index, row in enumerate(payload["observational_ranking"], 1):
        lines.append(
            f"{index}. {row['label']}: {row['discounted_team_return_mean']:.6f}")
    task_check_names = (
        "blue_win_without_red_kill_negative",
        "red_win_positive",
        "more_blue_losses_raise_return_when_matched",
        "more_red_attack_losses_lower_return_when_matched",
        "mav_death_lowers_return_when_matched",
    )
    lines.extend([
        "", "## Task reward checks", "",
        "These checks use undiscounted task return (shared event + terminal).",
        "",
        "| Check | Samples/comparisons | Status |",
        "|---|---:|---|",
    ])
    for name in task_check_names:
        check = payload["ranking_checks"][name]
        count = check.get("sample_count", check.get("comparison_count", 0))
        status = "N/A" if check["pass"] is None else (
            "PASS" if check["pass"] else "FAIL")
        lines.append(f"| {name} | {count} | {status} |")
    performance_check = payload["ranking_checks"][
        "checkpoint_100k_not_below_250k"]
    lines.extend([
        "", "100K vs 250K continues to use mean discounted team return: "
        + ("N/A" if performance_check is None else (
            "PASS" if performance_check else "FAIL")),
        "", "## Potential consistency", "",
    ])
    potential_check = payload["ranking_checks"][
        "potential_telescoping_within_tolerance"]
    potential_status = "N/A" if potential_check["pass"] is None else (
        "PASS" if potential_check["pass"] else "FAIL")
    lines.append(
        f"potential_telescoping_within_tolerance: "
        f"samples={potential_check['sample_count']}, status={potential_status}")
    (output_dir / "reward_v5_ranking_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")
    episode_rows = []
    for result in payload["results"]:
        for row in result["episode_rows"]:
            episode_rows.append({
                "label": result["label"],
                **row,
                "initial_perturbation": json.dumps(
                    row["initial_perturbation"], sort_keys=True),
            })
    with (output_dir / "reward_v5_ranking_episodes.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2_reward_v5.yaml")
    parser.add_argument("--checkpoint", action="append", type=_checkpoint_spec, default=[])
    parser.add_argument("--include-initial", action="store_true")
    parser.add_argument("--include-rule", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", default="outputs/formal_v2_reward_v5_ranking_audit")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if not (args.include_initial or args.include_rule or args.checkpoint):
        parser.error("select --include-initial, --include-rule, or --checkpoint")

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    env = make_env(str(config))
    if env.reward_contract != V5_REWARD_CONTRACT_VERSION:
        raise ValueError("ranking audit requires the formal V5 environment")
    device = torch.device(args.device)
    policies = []
    sources = {}
    if args.include_initial:
        torch.manual_seed(args.seed)
        initial = PureHAPPOPolicy(
            73, 219, 3, 3, credit_mode="fixed_three_agent_team_mean").to(device)
        initial.eval()
        policies.append(("initial_policy", initial, None))
        sources["initial_policy"] = "untrained_initialization"
    for label, checkpoint in args.checkpoint:
        policy, meta = _load_actor_only(checkpoint, device)
        policies.append((label, policy, None))
        sources[label] = meta["reward_contract"]
    if args.include_rule:
        policies.append((RULE_POLICY_LABEL, None, PaperGreedyOpponent()))
        sources[RULE_POLICY_LABEL] = "rule_policy_no_checkpoint"

    scenarios = _paired_scenarios(args.seed, args.episodes)
    unique_perturbation_count = _validate_scenario_uniqueness(scenarios)

    results = [
        _rollout(env, label, scenarios, policy, rule)
        for label, policy, rule in policies
    ]
    payload = {
        "audit_reward_contract": env.reward_contract,
        "audit_type": "cross_reward_actor_only",
        "critic_loaded": False,
        "optimizer_or_training_state_loaded": False,
        "source_reward_contracts": sources,
        "results": results,
        "ranking_checks": _ranking_checks(results),
        "paired_scenarios": scenarios,
        "paired_scenario_count": len(scenarios),
        "unique_perturbation_count": unique_perturbation_count,
        "paired_scenarios_all_unique": (
            unique_perturbation_count == len(scenarios)),
        "observational_ranking": _observational_ranking(results),
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    _write_report(output_dir, payload)
    print(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
