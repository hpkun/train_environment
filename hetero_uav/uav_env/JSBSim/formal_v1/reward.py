"""Bounded TAM role reward with one shared team-reward output."""
from __future__ import annotations

import numpy as np

from .geometry import combat_geometry


def _height(aircraft) -> float:
    altitude = float(aircraft.get_position()[2])
    if altitude < 750.0:
        return -1.0
    return float(np.clip(1.0 - abs(altitude - 6000.0) / 5250.0, -1.0, 1.0))


def _uav_dense(env, agent_id: str, target_id: str | None) -> tuple[float, dict]:
    aircraft = env.aircraft[agent_id]
    if not aircraft.is_alive:
        return 0.0, {key: 0.0 for key in ("height", "speed", "angle", "distance", "avoidance")}
    target = env.aircraft[target_id] if target_id else None
    height = _height(aircraft)
    speed = 0.0
    angle = 0.0
    distance = 0.0
    if target is not None and target.is_alive:
        geom = combat_geometry(aircraft, target)
        own_speed = np.linalg.norm(aircraft.get_velocity())
        target_speed = np.linalg.norm(target.get_velocity())
        speed = float(np.clip((own_speed - target_speed) / 150.0, -1.0, 1.0))
        angle = float(np.clip(1.0 - (geom["ata_rad"] + geom["ta_rad"]) / np.pi, -1.0, 1.0))
        r_km = geom["range_m"] / 1000.0
        distance = 1.0 if r_km <= 5 else (float(np.exp(-0.921 * (r_km - 5))) if r_km < 10 else -1.0)
    threats = [m for m in env.missiles if m.is_launched and m.target_id == agent_id]
    avoidance = 0.0
    if threats:
        nearest = min(threats, key=lambda m: np.linalg.norm(m.position - aircraft.get_position()))
        rel = nearest.position - aircraft.get_position()
        closing = -float(np.dot(nearest.velocity - aircraft.get_velocity(), rel)) / max(np.linalg.norm(rel), 1e-6)
        avoidance = float(np.clip(closing / 600.0, 0.0, 1.0))
    raw = 10 * height + 10 * speed + 15 * angle + 10 * distance + 30 * avoidance
    return raw, {"height": height, "speed": speed, "angle": angle,
                 "distance": distance, "avoidance": avoidance}


def _mav_dense(env) -> tuple[float, dict]:
    mav = env.aircraft["red_0"]
    if not mav.is_alive:
        return 0.0, {"safety": 0.0, "support": 0.0}
    enemies = [env.aircraft[bid] for bid in env.blue_ids if env.aircraft[bid].is_alive]
    nearest = min((np.linalg.norm(e.get_position() - mav.get_position()) for e in enemies), default=80_000.0)
    r_dist = float(np.clip((nearest - 5_000.0) / 15_000.0, -1.0, 1.0))
    incoming = any(m.is_launched and m.target_id == "red_0" for m in env.missiles)
    r_threat = -1.0 if incoming else 0.2
    r_aspect = 0.0
    for enemy in enemies:
        ta = combat_geometry(mav, enemy)["ta_rad"]
        if ta < np.pi / 4:
            r_aspect -= 1.0 - ta / (np.pi / 4)
    r_aspect = float(np.clip(r_aspect, -1.0, 0.0))
    safety = float(np.clip(0.5 * r_dist + 0.3 * r_threat + 0.2 * r_aspect, -1.0, 1.0))
    red_uavs = [env.aircraft[aid] for aid in ("red_1", "red_2") if env.aircraft[aid].is_alive]
    if red_uavs:
        centre = np.mean([u.get_position() for u in red_uavs], axis=0)
        r_pos = float(np.clip(1.0 - np.linalg.norm(mav.get_position() - centre) / 20_000.0, -1.0, 1.0))
    else:
        r_pos = 0.0
    observable = sum(
        np.linalg.norm(env.aircraft[bid].get_position() - mav.get_position()) <= env.mav_detection_range_m
        for bid in env.blue_ids if env.aircraft[bid].is_alive)
    alive_blue = sum(env.aircraft[bid].is_alive for bid in env.blue_ids)
    r_aware = observable / max(alive_blue, 1)
    support = float(np.clip(0.6 * r_pos + 0.4 * r_aware, -1.0, 1.0))
    return 75.0 * (0.5 * safety + 0.5 * support), {"safety": safety, "support": support}


def compute_team_reward(env, selected_targets: dict, step_events: list[dict]) -> tuple[dict, dict]:
    kill_by = {event.get("shooter_id") for event in step_events if event.get("event") == "hit"}
    newly_dead = env.newly_dead
    totals, components = {}, {}
    for aid in env.red_ids:
        if env.roles[aid] == "mav":
            dense, detail = _mav_dense(env)
        else:
            dense, detail = _uav_dense(env, aid, selected_targets.get(aid))
        event = 200.0 * float(aid in kill_by) - 200.0 * float(aid in newly_dead)
        if aid in newly_dead and env.death_reasons.get(aid) == "out_of_zone":
            event = -100.0
        if env.roles[aid] == "mav" and aid in newly_dead:
            event = -200.0
        # A single total normalization keeps a full episode of bounded dense
        # shaping on the same order as one TAM event, instead of multiplying
        # dense reward by the 1000-step horizon.
        total = float(np.clip((dense / env.max_steps + event) / 200.0, -2.0, 2.0))
        totals[aid] = total
        components[aid] = {**detail, "event": event, "role_total": total}
    team = float(np.mean(list(totals.values())))
    return {aid: team for aid in env.red_ids}, {"per_agent": components, "team_reward": team}
