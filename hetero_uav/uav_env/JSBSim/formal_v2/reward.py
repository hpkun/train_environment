"""Paper-aligned role reward for formal V2.

Published category weights and formulas are preserved. Normalization constants
not published by TAM-HAPPO are explicit bounded environment design choices.
"""
from __future__ import annotations

import math
import numpy as np

from ..formal_v1.geometry import combat_geometry
from ..formal_v1.reward import (
    EVENT_REWARDS, shared_information_score, smooth_band,
)

REWARD_CONTRACT_VERSION = "paper_aligned_role_reward_v3"
UAV_WEIGHTS = {
    "height": 10.0,
    "speed": 10.0,
    "angle": 15.0,
    "distance": 10.0,
    "dodge": 30.0,
}
MAV_SAFETY_WEIGHTS = {"distance": 0.5, "threat": 0.3, "aspect": 0.2}
MAV_SUPPORT_WEIGHTS = {"position": 0.6, "awareness": 0.4}
MISSILE_SPEED_NORM_MPS = 1_000.0


def _finite_clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = float(np.nan_to_num(value))
    return float(np.clip(value, low, high))


def uav_height_reward(altitude_m: float) -> float:
    altitude = float(altitude_m)
    minimum, optimum, maximum = 750.0, 6_000.0, 10_000.0
    if altitude < minimum or altitude > maximum:
        return -1.0
    if altitude <= optimum:
        return float((altitude - minimum) / (optimum - minimum))
    return float((maximum - altitude) / (maximum - optimum))


def uav_speed_reward(own_speed_mps: float, target_speed_mps: float) -> float:
    own = max(float(own_speed_mps), 1e-6)
    target = max(float(target_speed_mps), 0.0)
    if target < 0.5 * own:
        return 1.0
    if target <= 1.5 * own:
        return _finite_clip(2.0 - 2.0 * target / own)
    return -1.0


def uav_angle_reward(ata_rad: float, ta_rad: float) -> float:
    # TAM AA is pi - formal TA: rear-aspect target means AA=0.
    aa_rad = math.pi - float(ta_rad)
    return _finite_clip(1.0 - (float(ata_rad) + aa_rad) / math.pi)


def uav_distance_reward(range_m: float) -> float:
    distance_km = float(range_m) / 1_000.0
    if distance_km <= 5.0:
        return 1.0
    if distance_km < 10.0:
        return float(math.exp(-0.921 * (distance_km - 5.0)))
    return -1.0


def uav_dodge_components(env, agent_id: str) -> dict[str, float]:
    aircraft = env.aircraft[agent_id]
    incoming = [
        missile for missile in env.missiles
        if missile.is_launched and missile.target_id == agent_id
    ]
    if not incoming:
        return {
            "dodge": 0.0, "dodge_angle": 0.0, "dodge_speed": 0.0,
            "missile_risk": 0.0,
        }
    missile = min(
        incoming,
        key=lambda item: float(np.linalg.norm(
            item.position - aircraft.get_position())),
    )
    velocity = np.asarray(missile.velocity, dtype=np.float64)
    los = np.asarray(aircraft.get_position(), dtype=np.float64) - missile.position
    speed = float(np.linalg.norm(velocity))
    distance = float(np.linalg.norm(los))
    if speed <= 1e-9 or distance <= 1e-9:
        angle_term = 0.0
    else:
        cos_lambda = float(np.clip(
            np.dot(velocity, los) / (speed * distance), -1.0, 1.0))
        angle_term = -cos_lambda
    previous = env.previous_missile_speed.get(missile.missile_id)
    speed_term = (
        0.0 if previous is None
        else (float(previous) - speed) / MISSILE_SPEED_NORM_MPS
    )
    env.previous_missile_speed[missile.missile_id] = speed
    return {
        "dodge": _finite_clip(angle_term + speed_term),
        "dodge_angle": _finite_clip(angle_term),
        "dodge_speed": _finite_clip(speed_term),
        "missile_risk": float(np.clip(
            math.exp(-distance / 5_000.0), 0.0, 1.0)),
    }


def mav_safety_components(env) -> dict[str, float]:
    mav = env.aircraft["red_0"]
    enemies = [
        env.aircraft[target_id] for target_id in env.blue_ids
        if env.aircraft[target_id].is_alive
    ]
    if not enemies:
        distance_term = threat_term = aspect_term = 0.0
    else:
        nearest = min(float(np.linalg.norm(
            enemy.get_position() - mav.get_position())) for enemy in enemies)
        danger, safe = 8_000.0, 15_000.0
        if nearest <= danger:
            distance_term = -(1.0 - nearest / danger)
        elif nearest < safe:
            distance_term = -0.5 * (
                1.0 - (nearest - danger) / (safe - danger))
        else:
            distance_term = 0.2
        incoming = any(
            missile.is_launched and missile.target_id == "red_0"
            for missile in env.missiles)
        launch_threat = any(
            (geometry := combat_geometry(enemy, mav))["range_m"]
            <= env.attack_range_m
            and geometry["ata_rad"] <= env.launch_ata_rad
            and geometry["ta_rad"] >= env.launch_ta_rad
            for enemy in enemies
        )
        threat_term = -1.0 if incoming else (-0.5 if launch_threat else 0.0)
        aspect_terms = []
        for enemy in enemies:
            # Enemy TA in combat_geometry(mav, enemy) measures enemy heading
            # toward MAV; low values are the published unsafe aspect.
            enemy_toward_mav = combat_geometry(mav, enemy)["ta_rad"]
            aspect_terms.append(
                -(1.0 - enemy_toward_mav / (math.pi / 4.0))
                if enemy_toward_mav < math.pi / 4.0 else 0.0)
        aspect_term = min(aspect_terms, default=0.0)
    distance_term = _finite_clip(distance_term, -1.0, 0.2)
    threat_term = _finite_clip(threat_term, -1.0, 0.0)
    aspect_term = _finite_clip(aspect_term, -1.0, 0.0)
    safety = (
        MAV_SAFETY_WEIGHTS["distance"] * distance_term
        + MAV_SAFETY_WEIGHTS["threat"] * threat_term
        + MAV_SAFETY_WEIGHTS["aspect"] * aspect_term
    )
    return {
        "safety": _finite_clip(safety),
        "safety_distance": distance_term,
        "safety_threat": threat_term,
        "safety_aspect": aspect_term,
    }


def mav_support_components(env) -> dict[str, float]:
    mav = env.aircraft["red_0"]
    position_scores = []
    for agent_id in ("red_1", "red_2"):
        uav = env.aircraft[agent_id]
        if uav.is_alive:
            distance = float(np.linalg.norm(
                mav.get_position() - uav.get_position()))
            position_scores.append(
                smooth_band(distance, 750.0, 4_000.0, 15_000.0, 30_000.0))
    position = float(np.mean(position_scores)) if position_scores else 0.0
    awareness_terms = []
    for target_id in env.blue_ids:
        target = env.aircraft[target_id]
        if not target.is_alive:
            continue
        geometry = combat_geometry(mav, target)
        if geometry["range_m"] <= env.mav_detection_range_m:
            awareness_terms.append(max(
                0.0, 1.0 - geometry["ata_rad"] / (math.pi / 2.0)))
        else:
            awareness_terms.append(0.0)
    awareness = float(np.mean(awareness_terms)) if awareness_terms else 0.0
    support = (
        MAV_SUPPORT_WEIGHTS["position"] * position
        + MAV_SUPPORT_WEIGHTS["awareness"] * awareness
    )
    return {
        "support": _finite_clip(support),
        "support_position": _finite_clip(position),
        "awareness": _finite_clip(awareness, 0.0, 1.0),
        "shared_information_metric": shared_information_score(env),
    }


def _dead_mav_components(event: float) -> dict[str, float]:
    values = {
        "safety": 0.0, "safety_distance": 0.0, "safety_threat": 0.0,
        "safety_aspect": 0.0, "support": 0.0, "support_position": 0.0,
        "awareness": 0.0, "shared_information_metric": 0.0,
        "missile_risk": 0.0, "dense": 0.0, "event": event, "total": event,
    }
    return _with_mav_log_names(values)


def _dead_uav_components(event: float) -> dict[str, float]:
    values = {
        "height": 0.0, "speed": 0.0, "angle": 0.0, "distance": 0.0,
        "dodge": 0.0, "dodge_angle": 0.0, "dodge_speed": 0.0,
        "missile_risk": 0.0, "dense": 0.0, "event": event, "total": event,
    }
    return _with_uav_log_names(values)


def _with_mav_log_names(values: dict[str, float]) -> dict[str, float]:
    return {
        **values,
        "shared_information": values["shared_information_metric"],
        "mav_safety": values["safety"],
        "mav_safety_distance": values["safety_distance"],
        "mav_safety_threat": values["safety_threat"],
        "mav_safety_aspect": values["safety_aspect"],
        "mav_support": values["support"],
        "mav_support_position": values["support_position"],
        "mav_awareness": values["awareness"],
        "mav_shared_information_metric": values["shared_information_metric"],
        "mav_event": values["event"],
        "mav_total": values["total"],
    }


def _with_uav_log_names(values: dict[str, float]) -> dict[str, float]:
    return {
        **values,
        "flight": values["height"],
        "uav_height": values["height"],
        "uav_speed": values["speed"],
        "uav_angle": values["angle"],
        "uav_distance": values["distance"],
        "uav_dodge": values["dodge"],
        "uav_dodge_angle": values["dodge_angle"],
        "uav_dodge_speed": values["dodge_speed"],
        "uav_event": values["event"],
        "uav_total": values["total"],
    }


def compute_role_rewards(
    env, selected_targets: dict, step_events: list[dict],
) -> tuple[dict, dict]:
    rewards, components = {}, {}
    red_hits_by = [
        event.get("shooter_id") for event in step_events
        if event.get("event") == "hit"
        and str(event.get("shooter_id", "")).startswith("red")
    ]
    for agent_id in env.red_ids:
        aircraft = env.aircraft[agent_id]
        event = EVENT_REWARDS["red_kill"] * red_hits_by.count(agent_id)
        if agent_id in env.newly_dead:
            event += EVENT_REWARDS[
                "out_of_zone"
                if env.death_reasons.get(agent_id) == "out_of_zone"
                else ("mav_death" if env.roles[agent_id] == "mav" else "uav_death")
            ]
        if not aircraft.is_alive:
            detail = (
                _dead_mav_components(event)
                if env.roles[agent_id] == "mav"
                else _dead_uav_components(event)
            )
        elif env.roles[agent_id] == "mav":
            safety = mav_safety_components(env)
            support = mav_support_components(env)
            dense = safety["safety"] + support["support"]
            detail = _with_mav_log_names({
                **safety, **support, "missile_risk": 0.0,
                "dense": float(dense), "event": float(event),
                "total": float(dense + event),
            })
        else:
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
                    np.linalg.norm(target.get_velocity()))
                angle = uav_angle_reward(
                    geometry["ata_rad"], geometry["ta_rad"])
                distance = uav_distance_reward(geometry["range_m"])
            dodge = uav_dodge_components(env, agent_id)
            dense = (
                UAV_WEIGHTS["height"] * height
                + UAV_WEIGHTS["speed"] * speed
                + UAV_WEIGHTS["angle"] * angle
                + UAV_WEIGHTS["distance"] * distance
                + UAV_WEIGHTS["dodge"] * dodge["dodge"]
            )
            detail = _with_uav_log_names({
                "height": height, "speed": speed, "angle": angle,
                "distance": distance, **dodge, "dense": float(dense),
                "event": float(event), "total": float(dense + event),
            })
        if not np.isfinite(np.asarray([
                value for value in detail.values()
                if isinstance(value, (int, float, np.number))])).all():
            raise ValueError(f"non-finite formal v2 reward for {agent_id}")
        rewards[agent_id] = float(detail["total"])
        components[agent_id] = detail
    return rewards, {
        "per_agent": components,
        "reward_contract": REWARD_CONTRACT_VERSION,
    }
