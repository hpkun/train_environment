"""Paper-aligned role reward for formal V2.

Published category weights and formulas are preserved. Normalization constants
not published by TAM-HAPPO are explicit bounded environment design choices.
"""
from __future__ import annotations

import math
import numpy as np

from ..formal_v1.geometry import combat_geometry
from ..formal_v1.reward import shared_information_score

REWARD_CONTRACT_VERSION = "paper_aligned_role_reward_v4"
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
UAV_DENSE_NORMALIZER = 75.0
# R_safety is in [-1.2, 0.1] and R_support is in [-0.6, 0.84].
MAV_DENSE_NORMALIZER = 1.8
# The smallest published event magnitude is 100. This maps kill/death to
# +/-2 and out-of-zone to -1, keeping an event larger than one dense step.
EVENT_NORMALIZER = 100.0
EVENT_REWARDS = {
    "red_kill": 200.0,
    "uav_death": -200.0,
    "mav_death": -200.0,
    "out_of_zone": -100.0,
    "mav_team_kill": 100.0,
    "mav_team_kill_cap": 200.0,
}


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
        threat_term = -1.0 if incoming else 0.0
        aspect_terms = []
        for enemy in enemies:
            # Enemy TA in combat_geometry(mav, enemy) measures enemy heading
            # toward MAV; low values are the published unsafe aspect.
            enemy_toward_mav = combat_geometry(mav, enemy)["ta_rad"]
            aspect_terms.append(
                -(1.0 - enemy_toward_mav / (math.pi / 4.0))
                if enemy_toward_mav < math.pi / 4.0 else 0.0)
        aspect_term = sum(aspect_terms)
    distance_term = _finite_clip(distance_term, -1.0, 0.2)
    threat_term = _finite_clip(threat_term, -1.0, 0.0)
    aspect_term = _finite_clip(aspect_term, -2.0, 0.0)
    safety = (
        MAV_SAFETY_WEIGHTS["distance"] * distance_term
        + MAV_SAFETY_WEIGHTS["threat"] * threat_term
        + MAV_SAFETY_WEIGHTS["aspect"] * aspect_term
    )
    return {
        "safety": _finite_clip(safety, -1.2, 0.1),
        "safety_distance": distance_term,
        "safety_threat": threat_term,
        "safety_aspect": aspect_term,
    }


def mav_support_components(env) -> dict[str, float]:
    mav = env.aircraft["red_0"]
    combatants = [
        env.aircraft[agent_id]
        for agent_id in (*env.red_ids[1:], *env.blue_ids)
        if env.aircraft[agent_id].is_alive
    ]
    if combatants:
        battlefield_center = np.mean(
            [aircraft.get_position()[:2] for aircraft in combatants], axis=0)
        center_distance = float(np.linalg.norm(
            mav.get_position()[:2] - battlefield_center))
        d_opt, d_max = 8_000.0, 25_000.0
        if center_distance < d_opt:
            position = center_distance / d_opt - 1.0
        elif center_distance < d_max:
            position = 1.0 - (center_distance - d_opt) / (d_max - d_opt)
        else:
            position = -0.5
    else:
        center_distance = 0.0
        position = 0.0
    awareness = 0.0
    observed_count = 0
    for target_id in env.blue_ids:
        target = env.aircraft[target_id]
        if not target.is_alive:
            continue
        geometry = combat_geometry(mav, target)
        if geometry["range_m"] <= env.mav_detection_range_m:
            observed_count += 1
            if geometry["ata_rad"] < math.pi / 2.0:
                awareness += 0.3 * (
                    1.0 - geometry["ata_rad"] / (math.pi / 2.0))
    support = (
        MAV_SUPPORT_WEIGHTS["position"] * position
        + MAV_SUPPORT_WEIGHTS["awareness"] * awareness
    )
    return {
        "support": _finite_clip(support),
        "support_position": _finite_clip(position),
        "support_center_distance_m": float(center_distance),
        "awareness": _finite_clip(awareness, 0.0, 0.6),
        "awareness_observed_count": float(observed_count),
        "shared_information_metric": shared_information_score(env),
    }


def _scaled_reward_fields(
    raw_dense: float, raw_event: float, dense_normalizer: float,
) -> dict[str, float]:
    normalized_dense = _finite_clip(raw_dense / dense_normalizer)
    normalized_event = float(raw_event / EVENT_NORMALIZER)
    normalized_total = normalized_dense + normalized_event
    values = np.asarray([
        raw_dense, raw_event, normalized_dense, normalized_event,
        normalized_total,
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite formal v2 reward scaling")
    return {
        "raw_dense_reward": float(raw_dense),
        "raw_event_reward": float(raw_event),
        "raw_role_reward": float(raw_dense + raw_event),
        "normalized_dense_reward": normalized_dense,
        "normalized_event_reward": normalized_event,
        "normalized_role_reward": float(normalized_total),
        "dense": float(raw_dense),
        "event": normalized_event,
        "total": float(normalized_total),
    }


def _dead_mav_components(event: float) -> dict[str, float]:
    values = {
        "safety": 0.0, "safety_distance": 0.0, "safety_threat": 0.0,
        "safety_aspect": 0.0, "support": 0.0, "support_position": 0.0,
        "support_center_distance_m": 0.0, "awareness": 0.0,
        "awareness_observed_count": 0.0, "shared_information_metric": 0.0,
        "missile_risk": 0.0, "mav_team_kill_credit_delta": 0.0,
        "mav_team_kill_credit_used": 0.0,
        **_scaled_reward_fields(0.0, event, MAV_DENSE_NORMALIZER),
    }
    return _with_mav_log_names(values)


def _dead_uav_components(event: float) -> dict[str, float]:
    values = {
        "height": 0.0, "speed": 0.0, "angle": 0.0, "distance": 0.0,
        "dodge": 0.0, "dodge_angle": 0.0, "dodge_speed": 0.0,
        "missile_risk": 0.0,
        **_scaled_reward_fields(0.0, event, UAV_DENSE_NORMALIZER),
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
        "mav_event": values["raw_event_reward"],
        "mav_total": values["raw_role_reward"],
        "mav_raw_reward": values["raw_role_reward"],
        "mav_normalized_reward": values["normalized_role_reward"],
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
        "uav_event": values["raw_event_reward"],
        "uav_total": values["raw_role_reward"],
        "uav_raw_reward": values["raw_role_reward"],
        "uav_normalized_reward": values["normalized_role_reward"],
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
            step_attack_kills = sum(
                red_hits_by.count(attacker_id)
                for attacker_id in ("red_1", "red_2"))
            available_credit = max(
                0.0,
                EVENT_REWARDS["mav_team_kill_cap"]
                - float(env.v2_mav_team_credit_used),
            )
            team_credit = min(
                EVENT_REWARDS["mav_team_kill"] * step_attack_kills,
                available_credit,
            )
            env.v2_mav_team_credit_used += team_credit
            event += team_credit
            detail = _with_mav_log_names({
                **safety, **support, "missile_risk": 0.0,
                "mav_team_kill_credit_delta": float(team_credit),
                "mav_team_kill_credit_used": float(
                    env.v2_mav_team_credit_used),
                **_scaled_reward_fields(
                    dense, event, MAV_DENSE_NORMALIZER),
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
                "distance": distance, **dodge,
                **_scaled_reward_fields(
                    dense, event, UAV_DENSE_NORMALIZER),
            })
        if not np.isfinite(np.asarray([
                value for value in detail.values()
                if isinstance(value, (int, float, np.number))])).all():
            raise ValueError(f"non-finite formal v2 reward for {agent_id}")
        rewards[agent_id] = float(detail["normalized_role_reward"])
        components[agent_id] = detail
    team_reward = float(sum(rewards.values()) / 3.0)
    return rewards, {
        "per_agent": components,
        "reward_contract": REWARD_CONTRACT_VERSION,
        "credit_mode": "fixed_three_agent_team_mean",
        "team_reward": team_reward,
    }
