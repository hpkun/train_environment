"""Cross-reward ranking audit for the formal V5 shared team reward.

This script evaluates actor parameters only. It never restores a critic,
optimizer, rollout buffer, or training state from a different reward contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from uav_env.JSBSim.formal_v2.contract import V5_REWARD_CONTRACT_VERSION


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
    env, label: str, episodes: int, seed: int,
    policy: PureHAPPOPolicy | None, rule: PaperGreedyOpponent | None,
) -> dict:
    rows = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        sums = {
            "shared_event_return": 0.0,
            "terminal_return": 0.0,
            "potential_shaping_return": 0.0,
            "total_team_return": 0.0,
            "phi_mav": 0.0,
            "phi_uav_1": 0.0,
            "phi_uav_2": 0.0,
        }
        steps = 0
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
            sums["shared_event_return"] += components["shared_event_reward"]
            sums["terminal_return"] += components["terminal_reward"]
            sums["potential_shaping_return"] += components[
                "potential_shaping_reward"]
            sums["total_team_return"] += components["team_reward"]
            for key in ("phi_mav", "phi_uav_1", "phi_uav_2"):
                sums[key] += components[key]
            if not np.allclose(list(rewards.values()), components["team_reward"]):
                raise ValueError("V5 audit observed non-shared red rewards")
            steps += 1
            if any(terminations.values()) or any(truncations.values()):
                break
        events = env.event_log
        red_launches = sum(
            event["event"] == "launch"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events)
        red_hits = sum(
            event["event"] == "hit"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events)
        row = {
            "episode": episode,
            "outcome": info["outcome"],
            "red_launches": red_launches,
            "red_hits": red_hits,
            "red_kills": red_hits,
            "mav_survived": int(info["mav_alive"]),
            "red_attack_survival": info["red_attack_alive"] / 2.0,
            "episode_length": steps,
            **sums,
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
        "episodes": episodes,
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
        "shared_event_return_mean": mean("shared_event_return"),
        "terminal_return_mean": mean("terminal_return"),
        "potential_shaping_return_mean": mean("potential_shaping_return"),
        "total_team_return_mean": mean("total_team_return"),
        "phi_mav_mean": mean("phi_mav_mean"),
        "phi_uav_1_mean": mean("phi_uav_1_mean"),
        "phi_uav_2_mean": mean("phi_uav_2_mean"),
        "finite": True,
        "episode_rows": rows,
    }


def _ranking_checks(results: list[dict]) -> dict:
    by_label = {row["label"]: row for row in results}
    order = [
        "rule_red_success",
        "checkpoint_100k_attack",
        "initial_policy",
        "checkpoint_250k_collapsed",
    ]
    ranking_available = all(label in by_label for label in order)
    ranking_pass = None
    if ranking_available:
        values = [by_label[label]["total_team_return_mean"] for label in order]
        ranking_pass = all(left > right for left, right in zip(values, values[1:]))
    episodes = [row for result in results for row in result["episode_rows"]]
    return {
        "expected_ranking_available": ranking_available,
        "expected_ranking_pass": ranking_pass,
        "blue_win_without_red_kill_all_negative": all(
            row["total_team_return"] < 0
            for row in episodes
            if row["outcome"] == "blue_win" and row["red_kills"] == 0),
        "red_win_all_positive": all(
            row["total_team_return"] > 0
            for row in episodes if row["outcome"] == "red_win"),
    }


def _write_report(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward_v5_ranking_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Formal V5 Cross-Reward Ranking Audit",
        "",
        "This is an actor-only cross-reward audit. It is not checkpoint resume.",
        "",
        "| Policy | Source reward | Episodes | R win | B win | Launch | Hit | Team return |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    sources = payload["source_reward_contracts"]
    for row in payload["results"]:
        lines.append(
            f"| {row['label']} | {sources[row['label']]} | {row['episodes']} | "
            f"{row['red_win_rate']:.3f} | {row['blue_win_rate']:.3f} | "
            f"{row['red_launches']} | {row['red_hits']} | "
            f"{row['total_team_return_mean']:.4f} |")
    lines.extend(["", "## Ranking checks", "", "```json",
                  json.dumps(payload["ranking_checks"], indent=2), "```", ""])
    (output_dir / "reward_v5_ranking_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")


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
        policies.append(("rule_red_success", None, PaperGreedyOpponent()))
        sources["rule_red_success"] = "rule_policy_no_checkpoint"

    results = [
        _rollout(env, label, args.episodes, args.seed, policy, rule)
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
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    _write_report(output_dir, payload)
    print(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
