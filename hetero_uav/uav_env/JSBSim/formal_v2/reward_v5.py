"""Shared task reward with potential shaping for formal V2.

V5 reuses the V4 role-dense formulas as a state potential. It does not alter
the V4 reward implementation or any physical environment contract.
"""
from __future__ import annotations

import math
import numpy as np

from ..formal_v1.geometry import combat_geometry
from .reward import (
    MAV_DENSE_NORMALIZER,
    MAV_SAFETY_WEIGHTS,
    MAV_SUPPORT_WEIGHTS,
    UAV_DENSE_NORMALIZER,
    UAV_WEIGHTS,
    mav_safety_components,
    mav_support_components,
    uav_angle_reward,
    uav_distance_reward,
    uav_dodge_components,
    uav_height_reward,
    uav_speed_reward,
)

REWARD_CONTRACT_VERSION = "task_aligned_shared_potential_reward_v5"
POTENTIAL_GAMMA = 0.99
POTENTIAL_BETA = 0.25
EVENT_SCALE = 200.0
SHARED_EVENT_RAW = {
    "red_kill": 200.0,
    "red_attack_death": -200.0,
    "mav_death": -200.0,
    "out_of_zone": -100.0,
}


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite V5 reward value: {value}")
    return float(np.clip(value, low, high))


def _zero_mav_dense() -> dict[str, float]:
    return {
        "safety": 0.0,
        "safety_distance": 0.0,
        "safety_threat": 0.0,
        "safety_aspect": 0.0,
        "support": 0.0,
        "support_position": 0.0,
        "support_center_distance_m": 0.0,
        "awareness": 0.0,
        "awareness_observed_count": 0.0,
        "shared_information_metric": 0.0,
        "missile_risk": 0.0,
        "dense": 0.0,
        "phi": 0.0,
    }


def _zero_uav_dense() -> dict[str, float]:
    return {
        "height": 0.0,
        "speed": 0.0,
        "angle": 0.0,
        "distance": 0.0,
        "dodge": 0.0,
        "dodge_angle": 0.0,
        "dodge_speed": 0.0,
        "missile_risk": 0.0,
        "dense": 0.0,
        "phi": 0.0,
    }


def role_potential_components(env, selected_targets: dict) -> dict[str, dict]:
    """Evaluate the unchanged V4 role-dense formulas at the current state."""
    details: dict[str, dict] = {}
    mav = env.aircraft["red_0"]
    if not mav.is_alive:
        details["red_0"] = _zero_mav_dense()
    else:
        safety = mav_safety_components(env)
        support = mav_support_components(env)
        dense = float(safety["safety"] + support["support"])
        details["red_0"] = {
            **safety,
            **support,
            "missile_risk": 0.0,
            "dense": dense,
            "phi": _clip(dense / MAV_DENSE_NORMALIZER),
        }

    for agent_id in ("red_1", "red_2"):
        aircraft = env.aircraft[agent_id]
        if not aircraft.is_alive:
            details[agent_id] = _zero_uav_dense()
            continue
        target_id = selected_targets.get(agent_id)
        target = (
            env.aircraft[target_id]
            if target_id and env.aircraft[target_id].is_alive else None
        )
        height = uav_height_reward(aircraft.get_position()[2])
        if target is None:
            speed = angle = distance = 0.0
        else:
            geometry = combat_geometry(aircraft, target)
            speed = uav_speed_reward(
                np.linalg.norm(aircraft.get_velocity()),
                np.linalg.norm(target.get_velocity()),
            )
            angle = uav_angle_reward(
                geometry["ata_rad"], geometry["ta_rad"])
            distance = uav_distance_reward(geometry["range_m"])
        dodge = uav_dodge_components(env, agent_id)
        dense = float(
            UAV_WEIGHTS["height"] * height
            + UAV_WEIGHTS["speed"] * speed
            + UAV_WEIGHTS["angle"] * angle
            + UAV_WEIGHTS["distance"] * distance
            + UAV_WEIGHTS["dodge"] * dodge["dodge"]
        )
        details[agent_id] = {
            "height": height,
            "speed": speed,
            "angle": angle,
            "distance": distance,
            **dodge,
            "dense": dense,
            "phi": _clip(dense / UAV_DENSE_NORMALIZER),
        }
    return details


def _phi_team(details: dict[str, dict]) -> float:
    return float(sum(details[agent_id]["phi"] for agent_id in (
        "red_0", "red_1", "red_2")) / 3.0)


def reset_episode_state(env) -> None:
    details = role_potential_components(env, env.selected_targets)
    env.v5_phi_previous = _phi_team(details)
    env.v5_terminal_applied = False


def _shared_event(env, step_events: list[dict]) -> dict:
    red_kill_events = [
        event for event in step_events
        if event.get("event") == "hit"
        and str(event.get("shooter_id", "")).startswith("red")
        and str(event.get("target_id", "")).startswith("blue")
    ]
    out_of_zone = [
        agent_id for agent_id in env.newly_dead
        if agent_id in env.red_ids
        and env.death_reasons.get(agent_id) == "out_of_zone"
    ]
    red_attack_deaths = [
        agent_id for agent_id in env.newly_dead
        if agent_id in ("red_1", "red_2") and agent_id not in out_of_zone
    ]
    mav_deaths = [
        agent_id for agent_id in env.newly_dead
        if agent_id == "red_0" and agent_id not in out_of_zone
    ]
    raw = (
        SHARED_EVENT_RAW["red_kill"] * len(red_kill_events)
        + SHARED_EVENT_RAW["red_attack_death"] * len(red_attack_deaths)
        + SHARED_EVENT_RAW["mav_death"] * len(mav_deaths)
        + SHARED_EVENT_RAW["out_of_zone"] * len(out_of_zone)
    )
    sources = [{
        "event": "red_kill",
        "shooter_id": event.get("shooter_id", ""),
        "target_id": event.get("target_id", ""),
        "missile_id": event.get("missile_id", ""),
        "death_reason": env.death_reasons.get(event.get("target_id"), ""),
    } for event in red_kill_events]
    sources.extend({
        "event": "red_death",
        "shooter_id": "",
        "target_id": agent_id,
        "missile_id": "",
        "death_reason": env.death_reasons.get(agent_id, ""),
    } for agent_id in (*red_attack_deaths, *mav_deaths, *out_of_zone))
    return {
        "shared_event_raw": float(raw),
        "shared_event_reward": float(raw / EVENT_SCALE),
        "red_kill_count": len(red_kill_events),
        "red_attack_death_count": len(red_attack_deaths),
        "mav_death_count": len(mav_deaths),
        "out_of_zone_count": len(out_of_zone),
        "event_sources": sources,
    }


def _terminal_components(context: dict) -> dict[str, float]:
    if not context["team_done"]:
        return {
            "outcome_bonus": 0.0,
            "terminal_retention_raw": 0.0,
            "terminal_retention_reward": 0.0,
            "terminal_reward": 0.0,
        }
    outcome_bonus = {
        "red_win": 1.0,
        "blue_win": -1.0,
    }.get(context["outcome"], 0.0)
    blue_losses = 2 - int(context["blue_attack_alive"])
    red_attack_losses = 2 - int(context["red_attack_alive"])
    mav_loss = 1 - int(bool(context["mav_alive"]))
    retention_raw = 30.0 * (
        blue_losses - red_attack_losses - mav_loss)
    retention_reward = retention_raw / EVENT_SCALE
    return {
        "outcome_bonus": float(outcome_bonus),
        "terminal_retention_raw": float(retention_raw),
        "terminal_retention_reward": float(retention_reward),
        "terminal_reward": float(outcome_bonus + retention_reward),
    }


def potential_shaping_reward(
    phi_previous: float, phi_next: float, team_done: bool,
) -> tuple[float, float]:
    phi_next_effective = 0.0 if team_done else float(phi_next)
    shaping = POTENTIAL_BETA * (
        POTENTIAL_GAMMA * phi_next_effective - float(phi_previous))
    if not math.isfinite(shaping):
        raise ValueError("non-finite V5 potential shaping")
    return float(shaping), phi_next_effective


def _per_agent_diagnostics(
    role_details: dict[str, dict], team_reward: float,
    shared_event_reward: float,
) -> dict[str, dict]:
    result = {}
    for agent_id, detail in role_details.items():
        common = {
            **detail,
            "event": float(shared_event_reward),
            "total": float(team_reward),
            "raw_role_reward": float(detail["dense"]),
            "normalized_role_reward": float(detail["phi"]),
        }
        if agent_id == "red_0":
            common.update({
                "shared_information": detail["shared_information_metric"],
                "mav_safety": detail["safety"],
                "mav_safety_distance": detail["safety_distance"],
                "mav_safety_threat": detail["safety_threat"],
                "mav_safety_aspect": detail["safety_aspect"],
                "mav_support": detail["support"],
                "mav_support_position": detail["support_position"],
                "mav_awareness": detail["awareness"],
                "mav_shared_information_metric": detail[
                    "shared_information_metric"],
                "mav_event": float(shared_event_reward),
                "mav_total": float(team_reward),
            })
        else:
            common.update({
                "flight": detail["height"],
                "uav_height": detail["height"],
                "uav_speed": detail["speed"],
                "uav_angle": detail["angle"],
                "uav_distance": detail["distance"],
                "uav_dodge": detail["dodge"],
                "uav_dodge_angle": detail["dodge_angle"],
                "uav_dodge_speed": detail["dodge_speed"],
                "uav_event": float(shared_event_reward),
                "uav_total": float(team_reward),
            })
        result[agent_id] = common
    return result


def compute_team_rewards(
    env,
    selected_targets: dict,
    step_events: list[dict],
    transition_context: dict,
) -> tuple[dict, dict]:
    role_details = role_potential_components(env, selected_targets)
    phi_previous = float(env.v5_phi_previous)
    phi_next = _phi_team(role_details)
    team_done = bool(transition_context["team_done"])
    repeated_terminal = team_done and bool(env.v5_terminal_applied)
    shaping, phi_next_effective = potential_shaping_reward(
        phi_previous, phi_next, team_done)
    if repeated_terminal:
        shaping = 0.0

    if env.v5_last_event_step == env.step_count:
        event = {
            "shared_event_raw": 0.0,
            "shared_event_reward": 0.0,
            "red_kill_count": 0,
            "red_attack_death_count": 0,
            "mav_death_count": 0,
            "out_of_zone_count": 0,
            "event_sources": [],
        }
    else:
        event = _shared_event(env, step_events)
        env.v5_last_event_step = env.step_count
    terminal = (
        {"outcome_bonus": 0.0, "terminal_retention_raw": 0.0,
         "terminal_retention_reward": 0.0, "terminal_reward": 0.0}
        if repeated_terminal else _terminal_components(transition_context)
    )
    team_reward = float(
        event["shared_event_reward"]
        + terminal["terminal_reward"]
        + shaping
    )
    if not team_done:
        env.v5_phi_previous = phi_next
    else:
        env.v5_terminal_applied = True

    numeric = [
        phi_previous, phi_next, phi_next_effective, shaping, team_reward,
        event["shared_event_raw"], event["shared_event_reward"],
        *terminal.values(),
    ]
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise ValueError("non-finite V5 shared reward")

    rewards = {agent_id: team_reward for agent_id in env.red_ids}
    components = {
        "reward_contract": REWARD_CONTRACT_VERSION,
        "credit_mode": "fixed_three_agent_team_mean",
        "shared_team_reward": True,
        **event,
        **terminal,
        "phi_mav": float(role_details["red_0"]["phi"]),
        "phi_uav_1": float(role_details["red_1"]["phi"]),
        "phi_uav_2": float(role_details["red_2"]["phi"]),
        "phi_team_previous": phi_previous,
        "phi_team_next": phi_next,
        "phi_team_next_effective": phi_next_effective,
        "potential_gamma": POTENTIAL_GAMMA,
        "potential_beta": POTENTIAL_BETA,
        "potential_shaping_reward": float(shaping),
        "team_reward": team_reward,
        "outcome": transition_context["outcome"],
        "end_reason": transition_context["end_reason"],
    }
    components["per_agent"] = _per_agent_diagnostics(
        role_details, team_reward, event["shared_event_reward"])
    return rewards, components


def reward_metadata() -> dict:
    return {
        "reward_contract": REWARD_CONTRACT_VERSION,
        "potential_gamma": POTENTIAL_GAMMA,
        "potential_beta": POTENTIAL_BETA,
        "event_scale": EVENT_SCALE,
        "shared_event_raw": dict(SHARED_EVENT_RAW),
        "outcome_bonus": {
            "red_win": 1.0,
            "blue_win": -1.0,
            "mutual_elimination": 0.0,
            "draw": 0.0,
            "invalid": 0.0,
        },
        "terminal_retention_formula": (
            "30*(blue_losses-red_attack_losses-mav_loss)/200"),
        "shared_team_reward": True,
        "uav_weights": dict(UAV_WEIGHTS),
        "mav_safety_weights": dict(MAV_SAFETY_WEIGHTS),
        "mav_support_weights": dict(MAV_SUPPORT_WEIGHTS),
    }
