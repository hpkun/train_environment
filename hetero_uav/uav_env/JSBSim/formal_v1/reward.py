"""Bounded role rewards for the formal heterogeneous 3v2 contract.

TAM-HAPPO supplies the role objectives and relative category importance.  The
smooth bounds below are explicit learnability adaptations where the paper does
not publish normalization constants.  No function in this module mutates the
environment.
"""
from __future__ import annotations

import numpy as np

from .geometry import angle, combat_geometry, unit
from .sensing import red_track_sources


GLOBAL_REWARD_SCALE = 1.0
REWARD_CONTRACT_VERSION = "formal_role_reward_v2"
UAV_WEIGHTS = {"flight": 0.20, "speed": 0.15, "angle": 0.25,
               "distance": 0.20, "dodge": 0.20}
MAV_WEIGHTS = {"safety": 0.45, "support_position": 0.25,
               "shared_information": 0.30}
EVENT_REWARDS = {"red_kill": 8.0, "uav_death": -8.0,
                 "mav_death": -10.0, "out_of_zone": -6.0}

assert abs(sum(UAV_WEIGHTS.values()) - 1.0) < 1e-12
assert abs(sum(MAV_WEIGHTS.values()) - 1.0) < 1e-12


def _clip(value: float) -> float:
    return float(np.clip(np.nan_to_num(value), -1.0, 1.0))


def smooth_band(value: float, outer_low: float, inner_low: float,
                inner_high: float, outer_high: float) -> float:
    """Continuous trapezoid: -1 outside outer bounds, +1 in the inner band."""
    value = float(value)
    if value <= outer_low or value >= outer_high:
        return -1.0
    if inner_low <= value <= inner_high:
        return 1.0
    if value < inner_low:
        return _clip(-1.0 + 2.0 * (value - outer_low) / (inner_low - outer_low))
    return _clip(1.0 - 2.0 * (value - inner_high) / (outer_high - inner_high))


def flight_safety_score(altitude_m: float, speed_mps: float) -> float:
    altitude = smooth_band(altitude_m, 100.0, 1_000.0, 9_000.0, 10_500.0)
    speed = smooth_band(speed_mps, 80.0, 160.0, 360.0, 460.0)
    return _clip(0.55 * altitude + 0.45 * speed)


def speed_situation_score(own_speed_mps: float, target_speed_mps: float) -> float:
    """Best near a modest +30 m/s advantage; unsafe speeds suppress advantage."""
    relative = float(own_speed_mps - target_speed_mps)
    relative_score = 2.0 * np.exp(-((relative - 30.0) / 100.0) ** 2) - 1.0
    envelope = 0.5 * (smooth_band(own_speed_mps, 80, 150, 380, 460) + 1.0)
    return _clip(relative_score * envelope - (1.0 - envelope))


def angle_situation_score(ata_rad: float, ta_rad: float) -> float:
    """ATA=0 and TA=pi is best; ATA=pi and TA=0 is worst."""
    return _clip(0.5 * np.cos(ata_rad) - 0.5 * np.cos(ta_rad))


def distance_situation_score(range_m: float) -> float:
    """Smooth engagement band: 3-10 km best, collision and >14 km are poor."""
    return smooth_band(range_m, 500.0, 3_000.0, 10_000.0, 14_000.0)


def missile_risk(position, velocity, missiles) -> float:
    """Most urgent real incoming-missile risk in [0,1]."""
    risks = []
    position = np.asarray(position, dtype=float); velocity = np.asarray(velocity, dtype=float)
    for missile in missiles:
        rel = np.asarray(missile.position, dtype=float) - position
        distance = float(np.linalg.norm(rel))
        missile_velocity = np.asarray(missile.velocity, dtype=float)
        closing = max(0.0, -float(np.dot(missile_velocity - velocity, unit(rel))))
        tgo = distance / max(closing, 1e-6)
        distance_risk = np.exp(-distance / 5_000.0)
        closing_risk = 1.0 - np.exp(-closing / 250.0)
        tgo_risk = np.exp(-tgo / 8.0) if closing > 1e-6 else 0.0
        approach = max(0.0, float(np.dot(unit(missile_velocity), -unit(rel))))
        risks.append(0.30 * distance_risk + 0.25 * closing_risk +
                     0.30 * tgo_risk + 0.15 * approach)
    return float(np.clip(max(risks, default=0.0), 0.0, 1.0))


def dodge_score(previous_risk: float, current_risk: float) -> float:
    if previous_risk <= 0.0 and current_risk <= 0.0:
        return 0.0
    return _clip((previous_risk - current_risk) / 0.20)


def uav_dense_components(position, velocity, target_position, target_velocity,
                         incoming_missiles=(), previous_risk: float = 0.0) -> dict:
    speed = float(np.linalg.norm(velocity))
    flight = flight_safety_score(float(position[2]), speed)
    if target_position is None:
        speed_score = angle_score = distance_score = 0.0
    else:
        rel = np.asarray(target_position) - np.asarray(position)
        target_speed = float(np.linalg.norm(target_velocity))
        ata = angle(velocity, rel)
        ta = angle(target_velocity, -rel)
        speed_score = speed_situation_score(speed, target_speed)
        angle_score = angle_situation_score(ata, ta)
        distance_score = distance_situation_score(float(np.linalg.norm(rel)))
    current_risk = missile_risk(position, velocity, incoming_missiles)
    dodge = dodge_score(previous_risk, current_risk)
    total = (UAV_WEIGHTS["flight"] * flight + UAV_WEIGHTS["speed"] * speed_score +
             UAV_WEIGHTS["angle"] * angle_score + UAV_WEIGHTS["distance"] * distance_score +
             UAV_WEIGHTS["dodge"] * dodge)
    return {"flight": flight, "speed": speed_score, "angle": angle_score,
            "distance": distance_score, "dodge": dodge,
            "missile_risk": current_risk, "dense": _clip(total)}


def shared_information_score(env) -> float:
    mav = env.aircraft["red_0"]
    if not mav.is_alive:
        return 0.0
    total_pairs = shared_only_pairs = 0
    for aid in ("red_1", "red_2"):
        uav = env.aircraft[aid]
        if not uav.is_alive:
            continue
        tracks = red_track_sources(env, aid)
        for bid in env.blue_ids:
            if not env.aircraft[bid].is_alive:
                continue
            total_pairs += 1
            shared_only_pairs += int(
                not tracks[bid]["direct"] and tracks[bid]["mav_shared"])
    if not total_pairs:
        return 0.0
    return float(np.clip(shared_only_pairs / total_pairs, 0.0, 1.0))


def support_position_score(env) -> float:
    mav = env.aircraft["red_0"]
    scores = []
    for aid in ("red_1", "red_2"):
        uav = env.aircraft[aid]
        if uav.is_alive:
            distance = float(np.linalg.norm(mav.get_position() - uav.get_position()))
            scores.append(smooth_band(distance, 750.0, 4_000.0, 15_000.0, 30_000.0))
    return float(np.mean(scores)) if scores else 0.0


def mav_dense_components(env, previous_risk: float) -> dict:
    mav = env.aircraft["red_0"]
    if not mav.is_alive:
        return {"safety": 0.0, "support_position": 0.0, "shared_information": 0.0,
                "missile_risk": 0.0, "dense": 0.0}
    enemies = [env.aircraft[bid] for bid in env.blue_ids if env.aircraft[bid].is_alive]
    incoming = [m for m in env.missiles if m.is_launched and m.target_id == "red_0"]
    risk = missile_risk(mav.get_position(), mav.get_velocity(), incoming)
    flight = flight_safety_score(mav.get_position()[2], np.linalg.norm(mav.get_velocity()))
    if enemies:
        distances = [float(np.linalg.norm(enemy.get_position() - mav.get_position())) for enemy in enemies]
        distance_safety = float(np.tanh((min(distances) - 8_000.0) / 8_000.0))
        attack_threat = max(
            max(0.0, np.cos(combat_geometry(enemy, mav)["ata_rad"])) *
            np.exp(-np.linalg.norm(enemy.get_position() - mav.get_position()) / 14_000.0)
            for enemy in enemies)
        aspect_safety = -float(attack_threat)
    else:
        distance_safety = aspect_safety = 0.0
    risk_trend = dodge_score(previous_risk, risk)
    safety = _clip(0.25 * flight + 0.25 * distance_safety +
                   0.20 * aspect_safety - 0.15 * risk + 0.15 * risk_trend)
    position = support_position_score(env)
    information = shared_information_score(env)
    total = (MAV_WEIGHTS["safety"] * safety + MAV_WEIGHTS["support_position"] * position +
             MAV_WEIGHTS["shared_information"] * information)
    return {"safety": safety, "support_position": position,
            "shared_information": information, "missile_risk": risk, "dense": _clip(total)}


def compute_role_rewards(env, selected_targets: dict, step_events: list[dict]) -> tuple[dict, dict]:
    """Return distinct role rewards. Pure HAPPO performs the only team aggregation."""
    rewards, components = {}, {}
    red_hits_by = [e.get("shooter_id") for e in step_events
                   if e.get("event") == "hit" and str(e.get("shooter_id", "")).startswith("red")]
    for aid in env.red_ids:
        aircraft = env.aircraft[aid]
        if not aircraft.is_alive:
            if env.roles[aid] == "mav":
                dense_components = {
                    "safety": 0.0, "support_position": 0.0,
                    "shared_information": 0.0, "missile_risk": 0.0, "dense": 0.0,
                }
            else:
                dense_components = {
                    "flight": 0.0, "speed": 0.0, "angle": 0.0,
                    "distance": 0.0, "dodge": 0.0,
                    "missile_risk": 0.0, "dense": 0.0,
                }
        elif env.roles[aid] == "mav":
            dense_components = mav_dense_components(env, env.previous_missile_risk[aid])
        else:
            target_id = selected_targets.get(aid)
            target = env.aircraft[target_id] if target_id and env.aircraft[target_id].is_alive else None
            incoming = [m for m in env.missiles if m.is_launched and m.target_id == aid]
            dense_components = uav_dense_components(
                aircraft.get_position(), aircraft.get_velocity(),
                None if target is None else target.get_position(),
                None if target is None else target.get_velocity(), incoming,
                env.previous_missile_risk[aid])
        event = EVENT_REWARDS["red_kill"] * red_hits_by.count(aid)
        if aid in env.newly_dead:
            if env.death_reasons.get(aid) == "out_of_zone":
                event += EVENT_REWARDS["out_of_zone"]
            else:
                event += EVENT_REWARDS["mav_death" if env.roles[aid] == "mav" else "uav_death"]
        dense = float(dense_components.get("dense", 0.0))
        total = GLOBAL_REWARD_SCALE * (dense + event)
        if not np.isfinite(total):
            raise ValueError(f"non-finite formal reward for {aid}")
        rewards[aid] = float(total)
        components[aid] = {**dense_components, "event": float(event), "total": float(total)}
    return rewards, {"per_agent": components, "global_reward_scale": GLOBAL_REWARD_SCALE}
