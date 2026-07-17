"""Runtime helpers for heterogeneous-reward vanilla HAPPO."""

from __future__ import annotations

import math
from pathlib import Path
import random
import numpy as np
import torch

from algorithms.happo.vanilla_happo import VanillaHAPPOPolicy
from uav_env.make_env import make_env
from uav_env.JSBSim.paper.action_semantics import INACTIVE_ACTION_PLACEHOLDER
from uav_env.JSBSim.paper.protocol import (
    NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL,
    PAPER_5V4_GENERALIZATION_PROTOCOL, validate_generalization_protocol,
    validate_nominal_protocol)


SCENARIOS = {"2v2": "tam_paper_env_v1_2v2.yaml",
             "3v2": "tam_paper_env_v1_3v2.yaml",
             "5v4": "tam_paper_env_v1_5v4.yaml"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_paper_env(root: Path, scenario: str, *, initial_perturbation=None,
                   dynamics_backend="jsbsim",
                   experiment_protocol=PAPER_NOMINAL_PROTOCOL):
    perturbation = initial_perturbation or NOMINAL_PERTURBATION
    if experiment_protocol == PAPER_NOMINAL_PROTOCOL:
        validate_nominal_protocol(scenario, perturbation)
    elif experiment_protocol == PAPER_5V4_GENERALIZATION_PROTOCOL:
        validate_generalization_protocol(scenario, perturbation)
    else:
        raise ValueError(f"unknown paper experiment protocol {experiment_protocol!r}")
    kwargs = {"dynamics_backend": dynamics_backend}
    kwargs.update({"initial_perturbation": perturbation, "scenario": scenario,
                   "experiment_protocol": experiment_protocol})
    return make_env(str(root / "uav_env" / "JSBSim" / "configs" / SCENARIOS[scenario]),
                    **kwargs)


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


def update_side_timing(tracker, info, agents):
    """Update red/blue event timing without allowing one side to fill the other."""
    by_id = {agent.agent_id: agent for agent in agents}
    now = float(info["simulation_time_s"])
    for side in ("red", "blue"):
        side_ids = [agent.agent_id for agent in agents if agent.side == side]
        targets = [info["current_targets"].get(aid) for aid in side_ids]
        if tracker[side]["first_detection_time_s"] is None and any(
                target is not None for target in targets):
            tracker[side]["first_detection_time_s"] = now
        if tracker[side]["first_attack_range_entry_s"] is None:
            for aid, target_id in zip(side_ids, targets):
                if target_id is not None and np.linalg.norm(
                        by_id[target_id].position - by_id[aid].position) <= 14000:
                    tracker[side]["first_attack_range_entry_s"] = now
                    break
        tracker[side]["target_switches"] += sum(
            int(info["target_selection"].get(aid, {}).get("target_changed", False))
            for aid in side_ids)
    for event in info.get("missile_events", []):
        shooter = by_id.get(event.get("shooter_id"))
        if shooter is None:
            continue
        side = shooter.side
        event_time = float(event.get("simulation_time_s", now))
        if event.get("reason") == "launched":
            tracker[side]["missiles_fired"] += 1
            if tracker[side]["first_launch_time_s"] is None:
                tracker[side]["first_launch_time_s"] = event_time
        if event.get("reason") == "hit":
            tracker[side]["hits"] += 1
            if tracker[side]["first_hit_time_s"] is None:
                tracker[side]["first_hit_time_s"] = event_time


def new_side_tracker():
    return {side: {"first_detection_time_s": None,
                   "first_attack_range_entry_s": None,
                   "first_launch_time_s": None, "first_hit_time_s": None,
                   "target_switches": 0, "missiles_fired": 0, "hits": 0}
            for side in ("red", "blue")}


_new_side_tracker = new_side_tracker  # Backward-compatible private alias.


def stack_controlled_rule_actions(env):
    """Stack rule actions in controlled-agent order, filling dead agents only."""
    rule = env.build_rule_actions(env.agent_ids)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    actions = []
    for aid in env.agent_ids:
        agent = by_id.get(aid)
        if agent is None:
            raise RuntimeError(f"controlled agent {aid!r} is missing from env.task.agents")
        if aid in rule:
            action = np.asarray(rule[aid], dtype=np.int64)
        elif agent.alive:
            raise RuntimeError(f"alive controlled agent {aid!r} has no rule action")
        else:
            action = np.asarray(INACTIVE_ACTION_PLACEHOLDER, dtype=np.int64)
        if action.shape != (4,):
            raise RuntimeError(
                f"rule action for {aid!r} must have shape (4,), got {action.shape}")
        actions.append(action)
    return np.stack(actions).astype(np.int64, copy=False)


def deterministic_evaluate(env, policy, episodes, seed, baseline="trained_happo",
                           episode_seeds=None):
    records = []
    rng = np.random.default_rng(seed)
    seeds = list(episode_seeds) if episode_seeds is not None else [
        seed + episode for episode in range(episodes)]
    if len(seeds) != episodes:
        raise ValueError("episode_seeds length must equal episodes")
    for episode in range(episodes):
        obs, _ = env.reset(seed=seeds[episode])
        returns = {aid: 0.0 for aid in env.agent_ids}
        tracker = new_side_tracker()
        violations = 0
        finite = True
        while True:
            available = np.stack([env.get_avail_actions()[aid] for aid in env.agent_ids])
            if baseline == "neutral":
                actions = np.tile(np.array([24, 20, 20, 20]), (env.num_agents, 1))
            elif baseline == "random":
                actions = rng.integers(0, 40, size=(env.num_agents, 4))
            elif baseline == "rule":
                env.prepare_decision_context()
                actions = stack_controlled_rule_actions(env)
            else:
                with torch.no_grad():
                    actions = policy.act(flattened_obs(env, obs), env.get_state(),
                                         available, deterministic=True)["actions"].cpu().numpy()
            obs, rewards, terminated, truncated, info = env.step(
                {aid: actions[i] for i, aid in enumerate(env.agent_ids)})
            for aid in env.agent_ids:
                returns[aid] += float(rewards[aid])
            finite &= bool(np.isfinite(list(rewards.values())).all()
                           and np.isfinite(env.get_state()).all())
            update_side_timing(tracker, info, env.task.agents)
            violations += len(info["target_consistency_violation"])
            if all(terminated.values()) or all(truncated.values()):
                break
        metrics = info["aircraft_metrics"]
        by_side = {side: [a for a in env.task.agents if a.side == side]
                   for side in ("red", "blue")}
        record = {"episode_seed": seeds[episode],
                  "winner": info["winner"], "termination_reason": info["termination_reason"],
                  "episode_steps": info["episode_step"],
                  "target_consistency_violations": violations, "finite": bool(finite),
                  "all_maximum_speed_mps": max(x["max_speed_mps"] for x in metrics.values()),
                  "all_maximum_load_g": max(x["max_abs_load_factor_g"] for x in metrics.values())}
        for side in ("red", "blue"):
            agents = by_side[side]
            side_metrics = [metrics[a.agent_id] for a in agents]
            record.update({f"{side}_{key}": value for key, value in tracker[side].items()})
            record[f"{side}_maximum_speed_mps"] = max(x["max_speed_mps"] for x in side_metrics)
            record[f"{side}_maximum_load_g"] = max(x["max_abs_load_factor_g"] for x in side_metrics)
            record[f"{side}_structural_failures"] = 0
            record[f"{side}_crashes"] = sum(x["death_reason"] == "crash" for x in side_metrics)
            record[f"{side}_boundary"] = sum(x["death_reason"] == "boundary" for x in side_metrics)
            record[f"{side}_survival_rate"] = sum(a.alive for a in agents) / len(agents)
        red_values = list(returns.values())
        record["red_episode_return"] = float(sum(red_values))
        record["red_agent_returns"] = returns
        record["blue_episode_return"] = None  # Blue is rule-controlled and emits no policy reward.
        records.append(record)
    numeric = sorted({key for record in records for key, value in record.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
    summary = {}
    for key in numeric:
        values = np.asarray([record[key] for record in records
                             if isinstance(record.get(key), (int, float))], dtype=float)
        half = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": float(values.std()),
                        "sample_count": int(len(values)),
                        "ci95": [float(values.mean()-half), float(values.mean()+half)]}
    summary.update({"episodes": episodes,
                    "win_rate": float(np.mean([r["winner"] == "red" for r in records])),
                    "all_finite": all(r["finite"] for r in records)})
    metadata_keys = (
        "environment_fidelity_revision", "experiment_protocol", "scenario",
        "initial_perturbation", "dynamics_backend", "paper_nominal_experiment",
        "paper_generalization_experiment", "paper_silent_assumptions_present")
    metadata_keys += (
        "neutral_action_semantics", "blue_policy_fidelity",
        "reference_8_exact_blue_fsm_reproduced")
    return {"baseline": baseline, "episode_seeds": seeds,
            "episodes_detail": records, "summary": summary} | {
                key: env.task.last_info[key] for key in metadata_keys}
