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


def active_action_statistics(actions, active_masks, action_levels=40):
    """Return empirical action-head statistics for active samples only."""
    actions = np.asarray(actions)
    masks = np.asarray(active_masks, dtype=bool)
    if actions.shape[:-1] != masks.shape or actions.shape[-1] != 4:
        raise ValueError("actions must end in four heads and match active_masks")
    active = actions[masks]
    result = {"active_action_sample_count": int(active.shape[0])}
    for head in range(4):
        if active.shape[0] == 0:
            entropy, distribution = None, None
        else:
            counts = np.bincount(active[:, head], minlength=action_levels)
            distribution = counts.astype(float) / counts.sum()
            positive = distribution[distribution > 0]
            entropy = float(-(positive * np.log(positive)).sum())
            distribution = distribution.tolist()
        result[f"active_action_head_{head}_entropy"] = entropy
        result[f"active_action_head_{head}_distribution"] = distribution
    return result


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


def policy_architecture_from_config(config):
    """Resolve explicit paper widths or the legacy hidden_dim-only contract."""
    if ("actor_hidden_sizes" in config) != ("critic_hidden_sizes" in config):
        raise ValueError("actor and critic hidden-size metadata must appear together")
    if "actor_hidden_sizes" in config:
        return {
            "hidden_dim": config.get("hidden_dim"),
            "actor_hidden_sizes": tuple(config["actor_hidden_sizes"]),
            "critic_hidden_sizes": tuple(config["critic_hidden_sizes"]),
        }
    hidden = int(config.get("hidden_dim", 128))
    return {"hidden_dim": hidden, "actor_hidden_sizes": None,
            "critic_hidden_sizes": None}


def infer_policy(env, actor_sharing="independent", hidden_dim=None, device="cpu",
                 actor_hidden_sizes=None, critic_hidden_sizes=None):
    obs, _ = env.reset(seed=0)
    obs_dim = len(env.flatten_observation(obs[env.agent_ids[0]]))
    state_dim = len(env.get_state())
    policy = VanillaHAPPOPolicy(env.agent_ids, env.agent_roles, obs_dim, state_dim,
                                hidden_dim=hidden_dim, actor_sharing=actor_sharing,
                                actor_hidden_sizes=actor_hidden_sizes,
                                critic_hidden_sizes=critic_hidden_sizes)
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


def evaluate_policy_panel(
        env, policy, episodes, environment_seed, baseline="trained_happo",
        environment_seeds=None, policy_action_seeds=None, deterministic=True):
    """Evaluate a policy panel while restoring all caller RNG states."""
    records = []
    env_seeds = (list(environment_seeds) if environment_seeds is not None
                 else [int(environment_seed)] * episodes)
    action_seeds = (list(policy_action_seeds) if policy_action_seeds is not None
                    else [int(environment_seed) + 100000 + i for i in range(episodes)])
    if len(env_seeds) != episodes or len(action_seeds) != episodes:
        raise ValueError("environment and policy seed lists must match episodes")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=env_seeds[episode])
            seed_all(action_seeds[episode])
            random_policy_rng = np.random.default_rng(action_seeds[episode])
            returns = {aid: 0.0 for aid in env.agent_ids}
            tracker, violations, finite = new_side_tracker(), 0, True
            by_side = {side: [a for a in env.task.agents if a.side == side]
                       for side in ("red", "blue")}
            initial_count = {side: len(by_side[side]) for side in by_side}
            initial_combat_count = {side: sum(
                a.aircraft_type.role == "attack_uav" for a in by_side[side])
                for side in by_side}
            active_actions = []
            while True:
                available = np.stack(
                    [env.get_avail_actions()[aid] for aid in env.agent_ids])
                alive = np.asarray([next(
                    a for a in env.task.agents if a.agent_id == aid).alive
                    for aid in env.agent_ids], dtype=bool)
                if baseline == "neutral":
                    actions = np.tile(np.array([24, 20, 20, 20]),
                                      (env.num_agents, 1))
                elif baseline == "random":
                    actions = random_policy_rng.integers(
                        0, 40, size=(env.num_agents, 4))
                elif baseline == "rule":
                    env.prepare_decision_context()
                    actions = stack_controlled_rule_actions(env)
                else:
                    with torch.no_grad():
                        actions = policy.act(
                            flattened_obs(env, obs), env.get_state(), available,
                            deterministic=deterministic)["actions"].cpu().numpy()
                active_actions.extend(np.asarray(actions)[alive].tolist())
                obs, rewards, terminated, truncated, info = env.step(
                    {aid: actions[i] for i, aid in enumerate(env.agent_ids)})
                for aid in env.agent_ids:
                    returns[aid] += float(rewards[aid])
                finite &= bool(np.isfinite(list(rewards.values())).all()
                               and np.isfinite(env.get_state()).all())
                update_side_timing(tracker, info, env.task.agents)
                violations += len(info["target_consistency_violation"])
                if all(bool(terminated[aid] or truncated[aid])
                       for aid in env.agent_ids):
                    break
            metrics = info["aircraft_metrics"]
            record = {
                "episode_seed": int(env_seeds[episode]),
                "environment_seed": int(env_seeds[episode]),
                "policy_action_seed": int(action_seeds[episode]),
                "deterministic": bool(deterministic),
                "winner": info["winner"],
                "termination_reason": info["termination_reason"],
                "episode_steps": int(info["episode_step"]),
                "target_consistency_violations": int(violations),
                "finite": bool(finite),
                "all_maximum_speed_mps": max(
                    x["max_speed_mps"] for x in metrics.values()),
                "all_maximum_load_g": max(
                    x["max_abs_load_factor_g"] for x in metrics.values()),
            }
            action_array = np.asarray(active_actions, dtype=np.int64).reshape(-1, 4)
            action_stats = active_action_statistics(
                action_array, np.ones(len(action_array), dtype=bool))
            record.update(action_stats)
            for side in ("red", "blue"):
                agents = by_side[side]
                combat = [a for a in agents
                          if a.aircraft_type.role == "attack_uav"]
                side_metrics = [metrics[a.agent_id] for a in agents]
                survivors = sum(a.alive for a in agents)
                combat_survivors = sum(a.alive for a in combat)
                record.update({f"{side}_{key}": value
                               for key, value in tracker[side].items()})
                record[f"{side}_initial_count"] = initial_count[side]
                record[f"{side}_initial_combat_count"] = initial_combat_count[side]
                record[f"{side}_survivor_count"] = int(survivors)
                record[f"{side}_combat_survivor_count"] = int(combat_survivors)
                record[f"{side}_maximum_speed_mps"] = max(
                    x["max_speed_mps"] for x in side_metrics)
                record[f"{side}_maximum_load_g"] = max(
                    x["max_abs_load_factor_g"] for x in side_metrics)
                record[f"{side}_structural_failures"] = 0
                record[f"{side}_crashes"] = sum(
                    x["death_reason"] in {"crash", "nonfinite"}
                    for x in side_metrics)
                record[f"{side}_boundary"] = sum(
                    x["death_reason"] == "boundary" for x in side_metrics)
                record[f"{side}_survival_rate"] = (
                    survivors / initial_count[side] if initial_count[side] else 0.0)
                record[f"{side}_combat_survival_rate"] = (
                    combat_survivors / initial_combat_count[side]
                    if initial_combat_count[side] else 0.0)
                record[f"{side}_kills"] = int(info["kills"][side])
                record[f"{side}_hit_rate"] = (
                    record[f"{side}_hits"] / record[f"{side}_missiles_fired"]
                    if record[f"{side}_missiles_fired"] else 0.0)
                record[f"{side}_crash_rate"] = (
                    record[f"{side}_crashes"] / initial_count[side]
                    if initial_count[side] else 0.0)
                record[f"{side}_boundary_rate"] = (
                    record[f"{side}_boundary"] / initial_count[side]
                    if initial_count[side] else 0.0)
            record["red_episode_return"] = float(sum(returns.values()))
            record["red_agent_returns"] = returns
            record["blue_episode_return"] = None
            records.append(record)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if torch.cuda.is_available() and torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
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
    combined_actions = []
    for record in records:
        count = record["active_action_sample_count"]
        if count:
            # Reconstruct aggregate counts exactly from per-episode distributions.
            combined_actions.append((count, [
                record[f"active_action_head_{head}_distribution"]
                for head in range(4)]))
    active_summary = {"active_action_sample_count": int(sum(x[0] for x in combined_actions))}
    for head in range(4):
        if not combined_actions:
            distribution, entropy = None, None
        else:
            total = active_summary["active_action_sample_count"]
            distribution = (sum(count * np.asarray(distributions[head])
                                for count, distributions in combined_actions) / total)
            positive = distribution[distribution > 0]
            entropy = float(-(positive * np.log(positive)).sum())
            distribution = distribution.tolist()
        active_summary[f"active_action_head_{head}_distribution"] = distribution
        active_summary[f"active_action_head_{head}_entropy"] = entropy
    summary.update(active_summary)
    return {"baseline": baseline, "episode_seeds": env_seeds,
            "environment_seeds": env_seeds,
            "policy_action_seeds": action_seeds,
            "deterministic": bool(deterministic),
            "episodes_detail": records, "summary": summary} | {
                key: env.task.last_info[key] for key in metadata_keys}


def deterministic_evaluate(env, policy, episodes, seed, baseline="trained_happo",
                           episode_seeds=None):
    """Backward-compatible deterministic evaluator."""
    seeds = list(episode_seeds) if episode_seeds is not None else [
        seed + episode for episode in range(episodes)]
    return evaluate_policy_panel(
        env, policy, episodes, seed, baseline=baseline,
        environment_seeds=seeds,
        policy_action_seeds=[seed + 100000 + i for i in range(episodes)],
        deterministic=True)
