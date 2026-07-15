"""Structured entity observation and red-available centralized state."""
from __future__ import annotations

import numpy as np

from .contract import (
    ACTOR_OBS_DIM, ALTITUDE_SCALE_M, POSITION_SCALE_M, RELATIVE_SPEED_SCALE_MPS,
    SPEED_SCALE_MPS,
)
from .geometry import combat_geometry
from .sensing import red_track_sources


def _role(role: str) -> np.ndarray:
    return np.asarray([role == "mav", role == "attack_uav"], dtype=np.float32)


def _finite_clip(value, low=-1.0, high=1.0) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(value, dtype=np.float32)), low, high)


def build_actor_observation(env, agent_id: str) -> dict:
    ego = env.aircraft[agent_id]
    alive = float(ego.is_alive)
    position = ego.get_position() if ego.is_alive else np.zeros(3)
    velocity = ego.get_velocity() if ego.is_alive else np.zeros(3)
    roll, pitch, yaw = ego.get_rpy() if ego.is_alive else np.zeros(3)
    speed = float(np.linalg.norm(velocity))
    missile_ratio = ego.num_left_missiles / max(ego.num_missiles, 1) if ego.num_missiles else 0.0
    ego_state = _finite_clip([
        position[2] / ALTITUDE_SCALE_M, speed / SPEED_SCALE_MPS,
        roll / np.pi, pitch / (np.pi / 2), np.sin(yaw), np.cos(yaw),
        velocity[2] / SPEED_SCALE_MPS, missile_ratio,
        *_role(env.roles[agent_id]), alive,
    ])

    allies = []
    for other_id in (aid for aid in env.red_ids if aid != agent_id):
        other = env.aircraft[other_id]
        valid = float(ego.is_alive and other.is_alive)
        rel_pos = (other.get_position() - position) if valid else np.zeros(3)
        rel_vel = (other.get_velocity() - velocity) if valid else np.zeros(3)
        allies.append(_finite_clip([
            *(rel_pos / POSITION_SCALE_M), *(rel_vel / RELATIVE_SPEED_SCALE_MPS),
            np.linalg.norm(rel_pos) / POSITION_SCALE_M, *_role(env.roles[other_id]),
            float(other.is_alive), valid,
        ]))

    tracks = red_track_sources(env, agent_id)
    enemies = []
    for target_id in env.blue_ids:
        target = env.aircraft[target_id]
        track = tracks[target_id]
        valid = float(ego.is_alive and target.is_alive and track["observable"])
        if valid:
            geom = combat_geometry(ego, target)
            rel_pos = geom["relative_position"]
            rel_vel = target.get_velocity() - velocity
            values = [
                *(rel_pos / POSITION_SCALE_M), *(rel_vel / RELATIVE_SPEED_SCALE_MPS),
                geom["range_m"] / POSITION_SCALE_M,
                geom["ata_rad"] / np.pi, geom["ta_rad"] / np.pi,
                1.0, float(track["direct"]), float(track["mav_shared"]),
                float(target.is_alive), 1.0,
            ]
        else:
            values = [0.0] * 14
        enemies.append(_finite_clip(values))

    incoming = np.zeros(7, dtype=np.float32)
    threats = [m for m in env.missiles if m.is_launched and m.target_id == agent_id]
    if ego.is_alive and threats:
        missile = min(threats, key=lambda m: np.linalg.norm(m.position - position))
        rel = missile.position - position
        distance = float(np.linalg.norm(rel))
        closing = max(0.0, -float(np.dot(missile.velocity - velocity, rel)) / max(distance, 1e-6))
        tgo = distance / max(closing, 1e-6)
        incoming = _finite_clip([
            *(rel / POSITION_SCALE_M), distance / POSITION_SCALE_M,
            closing / RELATIVE_SPEED_SCALE_MPS, tgo / 60.0, 1.0,
        ])

    flat = np.concatenate([ego_state, *allies, *enemies, incoming]).astype(np.float32)
    if flat.shape != (ACTOR_OBS_DIM,) or not np.isfinite(flat).all():
        raise ValueError(f"invalid formal actor observation for {agent_id}: {flat.shape}")
    return {"ego": ego_state, "allies": np.stack(allies), "enemies": np.stack(enemies),
            "incoming_missile": incoming, "flat": flat}


def build_team_observations(env) -> tuple[dict, np.ndarray]:
    observations = {aid: build_actor_observation(env, aid) for aid in env.red_ids}
    critic = np.concatenate([
        observations[aid]["flat"] if env.aircraft[aid].is_alive else np.zeros(ACTOR_OBS_DIM, np.float32)
        for aid in env.red_ids
    ]).astype(np.float32)
    if not np.isfinite(critic).all():
        raise ValueError("non-finite formal critic state")
    return observations, critic
