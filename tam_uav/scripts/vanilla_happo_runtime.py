"""Shared runtime helpers for the formal vanilla-HAPPO baseline."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from algorithms.happo.vanilla_happo import VanillaHAPPOPolicy
from uav_env.make_env import make_env


SCENARIOS = {
    "2v2": "tam_paper_env_v1_2v2.yaml",
    "3v2": "tam_paper_env_v1_3v2.yaml",
    "5v4": "tam_paper_env_v1_5v4.yaml",
}


def make_paper_env(root: Path, scenario: str):
    return make_env(str(root / "uav_env" / "JSBSim" / "configs" / SCENARIOS[scenario]),
                    dynamics_backend="jsbsim")


def infer_policy(env, actor_sharing="independent", hidden_dim=128, device="cpu"):
    obs, _ = env.reset(seed=0)
    obs_dim = len(env.flatten_observation(obs[env.agent_ids[0]]))
    state_dim = len(env.get_state())
    policy = VanillaHAPPOPolicy(env.agent_ids, env.agent_roles, obs_dim, state_dim,
                                hidden_dim=hidden_dim, actor_sharing=actor_sharing)
    return policy.to(device), obs_dim, state_dim


def flattened_obs(env, observations):
    return np.stack([env.flatten_observation(observations[aid])
                     for aid in env.agent_ids]).astype(np.float32)


def deterministic_evaluate(env, policy, episodes, seed, baseline="trained_happo"):
    records = []
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        returns = {aid: 0.0 for aid in env.agent_ids}
        first_detection = first_attack = first_launch = first_hit = None
        switches = violations = 0
        finite = True
        while True:
            available = np.stack([env.get_avail_actions()[aid] for aid in env.agent_ids])
            if baseline == "neutral":
                actions = np.tile(np.array([24, 20, 20, 20]), (env.num_agents, 1))
            elif baseline == "random":
                actions = rng.integers(0, 40, size=(env.num_agents, 4))
            elif baseline == "rule":
                env.prepare_decision_context()
                rule = env.build_rule_actions(env.agent_ids)
                actions = np.stack([rule[aid] for aid in env.agent_ids])
            else:
                with torch.no_grad():
                    actions = policy.act(flattened_obs(env, obs), env.get_state(),
                                         available, deterministic=True)["actions"].cpu().numpy()
            obs, rewards, terminated, truncated, info = env.step(
                {aid: actions[i] for i, aid in enumerate(env.agent_ids)})
            for aid in env.agent_ids:
                returns[aid] += float(rewards[aid])
            finite &= np.isfinite(list(rewards.values())).all() and np.isfinite(env.get_state()).all()
            now = info["simulation_time_s"]
            if first_detection is None and any(value is not None for value in info["current_targets"].values()):
                first_detection = now
            if first_attack is None:
                by_id = {a.agent_id: a for a in env.task.agents}
                if any(tid is not None and np.linalg.norm(by_id[tid].position-by_id[aid].position) <= 14000
                       for aid, tid in info["current_targets"].items()):
                    first_attack = now
            if first_launch is None and info["missiles_fired"]:
                first_launch = now
            if first_hit is None and info["missile_hits"]:
                first_hit = now
            switches += sum(item["target_changed"] for item in info["target_selection"].values())
            violations += len(info["target_consistency_violation"])
            if all(terminated.values()) or all(truncated.values()):
                break
        metrics = info["aircraft_metrics"]
        uav_returns = [returns[aid] for aid in env.agent_ids if env.agent_roles[aid] != "mav"]
        records.append({
            "winner": info["winner"], "termination_reason": info["termination_reason"],
            "episode_steps": info["episode_step"], "episode_return": sum(returns.values()),
            "agent_returns": returns,
            "mav_return": sum(returns[aid] for aid in env.agent_ids if env.agent_roles[aid] == "mav"),
            "uav_mean_return": float(np.mean(uav_returns)) if uav_returns else 0.0,
            "missiles_fired": info["missiles_fired"], "hits": info["missile_hits"],
            "shotdown": info["shotdown"], "crash": info["crashes"],
            "boundary": info["out_of_zone"], "structural_failure": info["structural_failures"],
            "first_detection_time_s": first_detection, "first_attack_range_entry_s": first_attack,
            "first_launch_time_s": first_launch, "first_hit_time_s": first_hit,
            "target_switches": switches, "target_consistency_violations": violations,
            "maximum_speed_mps": max(x["max_speed_mps"] for x in metrics.values()),
            "maximum_load_g": max(x["max_abs_load_factor_g"] for x in metrics.values()),
            "finite": bool(finite),
        })
    numeric = ("episode_steps", "episode_return", "mav_return", "uav_mean_return",
               "missiles_fired", "hits", "shotdown", "crash", "boundary",
               "structural_failure", "target_switches", "target_consistency_violations",
               "maximum_speed_mps", "maximum_load_g")
    summary = {}
    for key in numeric:
        values = np.asarray([record[key] for record in records], dtype=float)
        half = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": float(values.std()),
                        "ci95": [float(values.mean()-half), float(values.mean()+half)]}
    summary["episodes"] = episodes
    summary["win_rate"] = float(np.mean([r["winner"] == "red" for r in records]))
    summary["all_finite"] = all(r["finite"] for r in records)
    return {"baseline": baseline, "episodes_detail": records, "summary": summary}
