"""Prototype extractor for strict paper-style 10-dim observations.

This module does not modify ``UavCombatEnv.observation_space`` and is not wired
into any training script.  It is a first pass at constructing Table 1 / Table 2
style observations from simulator state for later validation.
"""
from __future__ import annotations

import numpy as np


def _rotation_inertial_to_body(roll, pitch, heading) -> np.ndarray:
    """Return a 3x3 NEU-inertial to body-frame rotation matrix.

    Assumption: the project uses an inertial NEU frame, where x is north, y is
    east, and z is up.  ``roll``, ``pitch``, and ``heading`` are radians from
    ``AircraftSimulator.get_rpy()``.  This prototype uses a conventional
    yaw-pitch-roll composition and returns the transpose of body-to-inertial.
    The sign convention should be validated against JSBSim before training use.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(heading), np.sin(heading)

    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)
    r_y = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)
    r_z = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    body_to_inertial = r_z @ r_y @ r_x
    return body_to_inertial.T.astype(np.float64)


def _get_alpha_beta_placeholder(sim) -> tuple[float, float]:
    """Return alpha/beta if exposed by simulator, otherwise placeholder zeros."""
    alpha = None
    beta = None
    for name in ("get_alpha", "get_attack_angle"):
        getter = getattr(sim, name, None)
        if callable(getter):
            alpha = float(getter())
            break
    for name in ("get_beta", "get_sideslip_angle"):
        getter = getattr(sim, name, None)
        if callable(getter):
            beta = float(getter())
            break
    # Placeholder: AircraftSimulator currently does not expose alpha/beta.
    # A strict implementation should read JSBSim aerodynamic properties.
    return float(alpha if alpha is not None else 0.0), float(beta if beta is not None else 0.0)


def extract_self_state(sim) -> np.ndarray:
    """Extract paper Table 1 style self state.

    Output: [x, y, h, V, roll, pitch, heading, alpha, beta, Vd].
    ``Vd`` is down velocity; with NEU z-up velocity, ``Vd = -v_up``.
    Alpha/beta are placeholder zeros unless the simulator exposes getters.
    """
    position = np.asarray(sim.get_position(), dtype=np.float64)
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, heading = np.asarray(sim.get_rpy(), dtype=np.float64)
    alpha, beta = _get_alpha_beta_placeholder(sim)

    speed = float(np.linalg.norm(velocity))
    down_velocity = float(-velocity[2])
    return np.array([
        position[0],
        position[1],
        position[2],
        speed,
        roll,
        pitch,
        heading,
        alpha,
        beta,
        down_velocity,
    ], dtype=np.float32)


def extract_relative_state(observer_sim, target_sim,
                           radar_detected: bool = True) -> np.ndarray:
    """Extract paper Table 2 style relative observation.

    Output:
    [x_body, y_body, z_body, theta_v_body, psi_v_body, V_target,
     theta_LOS_body, psi_LOS_body, q_LOS, d]

    ``q_LOS`` is currently a placeholder line-of-sight angle defined as
    arccos(clamp(x_body / d, -1, 1)).  This needs review against the paper's
    exact geometric definition before training use.
    """
    obs_pos = np.asarray(observer_sim.get_position(), dtype=np.float64)
    tgt_pos = np.asarray(target_sim.get_position(), dtype=np.float64)
    obs_vel = np.asarray(observer_sim.get_velocity(), dtype=np.float64)
    tgt_vel = np.asarray(target_sim.get_velocity(), dtype=np.float64)
    roll, pitch, heading = np.asarray(observer_sim.get_rpy(), dtype=np.float64)
    r_bi = _rotation_inertial_to_body(roll, pitch, heading)

    rel_pos_body = r_bi @ (tgt_pos - obs_pos)
    x_body, y_body, z_body = [float(v) for v in rel_pos_body]
    d = float(np.linalg.norm(rel_pos_body))
    horizontal = float(np.linalg.norm(rel_pos_body[:2]))
    theta_los_body = float(np.arctan2(z_body, horizontal))
    psi_los_body = float(np.arctan2(y_body, x_body))
    if d <= 1e-8:
        q_los = 0.0
    else:
        q_los = float(np.arccos(np.clip(x_body / d, -1.0, 1.0)))

    rel_vel_body = r_bi @ (tgt_vel - obs_vel)
    rel_vel_horizontal = float(np.linalg.norm(rel_vel_body[:2]))
    theta_v_body = float(np.arctan2(rel_vel_body[2], rel_vel_horizontal))
    psi_v_body = float(np.arctan2(rel_vel_body[1], rel_vel_body[0]))
    target_speed = float(np.linalg.norm(tgt_vel))

    if not radar_detected:
        theta_v_body = 0.0
        psi_v_body = 0.0
        target_speed = 0.0

    return np.array([
        x_body,
        y_body,
        z_body,
        theta_v_body,
        psi_v_body,
        target_speed,
        theta_los_body,
        psi_los_body,
        q_los,
        d,
    ], dtype=np.float32)


def _ordered_team_sims(env, agent_id: str):
    if agent_id.startswith("blue"):
        own_ids = getattr(env, "blue_ids", list(env.blue_planes.keys()))
        enemy_ids = getattr(env, "red_ids", list(env.red_planes.keys()))
        own_planes = env.blue_planes
        enemy_planes = env.red_planes
    else:
        own_ids = getattr(env, "red_ids", list(env.red_planes.keys()))
        enemy_ids = getattr(env, "blue_ids", list(env.blue_planes.keys()))
        own_planes = env.red_planes
        enemy_planes = env.blue_planes
    allies = [(aid, own_planes[aid]) for aid in own_ids if aid != agent_id]
    enemies = [(aid, enemy_planes[aid]) for aid in enemy_ids]
    return allies, enemies


def _is_valid_sim(sim) -> bool:
    return bool(sim is not None and getattr(sim, "is_alive", False))


def build_strict_paper_entity_observation(env, agent_id: str):
    """Build prototype paper Table 1/Table 2 entity observation from env state.

    Returns:
        entities: shape (N_entities, 10)
        mask: shape (N_entities,), where 1 means invalid/dead and 0 means valid
        meta: schema and placeholder warnings
    """
    ego_sim = env._get_sim(agent_id) if hasattr(env, "_get_sim") else None
    if ego_sim is None:
        raise KeyError(f"Unknown agent_id: {agent_id}")

    allies, enemies = _ordered_team_sims(env, agent_id)
    rows = []
    mask = []

    if _is_valid_sim(ego_sim):
        rows.append(extract_self_state(ego_sim))
        mask.append(0)
    else:
        rows.append(np.zeros(10, dtype=np.float32))
        mask.append(1)

    radar_mode = "true_or_existing_env_method"
    for _aid, sim in allies + enemies:
        if not _is_valid_sim(ego_sim) or not _is_valid_sim(sim):
            rows.append(np.zeros(10, dtype=np.float32))
            mask.append(1)
            continue
        radar_detected = True
        if hasattr(env, "_is_detected_by_radar"):
            radar_detected = bool(env._is_detected_by_radar(ego_sim, sim))
        rows.append(extract_relative_state(ego_sim, sim, radar_detected=radar_detected))
        mask.append(0)

    entities = np.stack(rows).astype(np.float32)
    entity_mask = np.asarray(mask, dtype=np.int64)
    meta = {
        "entity_dim": 10,
        "schema": "paper_table1_table2_prototype",
        "alpha_beta": "placeholder_zero_if_unavailable",
        "q_los": "placeholder_definition_needs_review",
        "radar_detected": radar_mode,
        "layout": {
            "n_ego": 1,
            "n_allies": len(allies),
            "n_enemies": len(enemies),
            "n_entities": int(entities.shape[0]),
        },
    }
    return entities, entity_mask, meta
