"""Extractor for strict paper-style 10-dim observations.

``UavCombatEnv`` uses this module on its ``obs_mode="paper_strict"`` reset/step
observation path. It builds paper Table 1 / Table 2 style observations from
simulator state while keeping the legacy engineering observation path separate.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "_get_alpha_beta_with_source",
    "_get_alpha_beta_placeholder",
    "_is_valid_sim",
    "_ordered_team_sims",
    "_read_jsbsim_angle_property",
    "_rotation_inertial_to_body",
    "body_angles_from_neu_vector",
    "body_vector_to_inertial_neu",
    "build_strict_paper_entity_observation",
    "compute_body_x_q_los_from_body",
    "compute_q_los_placeholder",
    "describe_paper_entities",
    "extract_relative_state",
    "extract_self_state",
    "extract_self_state_with_meta",
    "ordered_entity_slots",
    "slot_aligned_alive_mask",
]


def _rotation_inertial_to_body(roll, pitch, heading) -> np.ndarray:
    """Return a 3x3 NEU-inertial to aerospace body-frame rotation.

    Environment vectors are NEU (z up), while the aircraft body frame follows
    the aerospace convention x forward, y right, z down. The Euler rotation is
    therefore formed in NED and preceded by the NEU-to-NED sign conversion.
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

    body_to_ned = r_z @ r_y @ r_x
    neu_to_ned = np.diag([1.0, 1.0, -1.0])
    return (body_to_ned.T @ neu_to_ned).astype(np.float64)


def body_vector_to_inertial_neu(vector_body, roll, pitch, heading) -> np.ndarray:
    """Convert aerospace body vector (x forward, y right, z down) to NEU."""
    r_bi = _rotation_inertial_to_body(roll, pitch, heading)
    return r_bi.T @ np.asarray(vector_body, dtype=np.float64)


def body_angles_from_neu_vector(vector_neu, roll, pitch, heading):
    """Return body vector, up-positive elevation, right-positive azimuth, qLOS."""
    r_bi = _rotation_inertial_to_body(roll, pitch, heading)
    body = r_bi @ np.asarray(vector_neu, dtype=np.float64)
    horizontal = float(np.hypot(body[0], body[1]))
    elevation = float(np.arctan2(-body[2], horizontal))
    azimuth = float(np.arctan2(body[1], body[0]))
    q_los = compute_body_x_q_los_from_body(body)
    return body, elevation, azimuth, q_los


def _get_alpha_beta_with_source(sim) -> tuple[float, float, str, str]:
    """Return alpha/beta plus source labels for diagnostics."""
    alpha = None
    beta = None
    alpha_source = "placeholder:0"
    beta_source = "placeholder:0"
    for name in ("get_alpha", "get_attack_angle"):
        getter = getattr(sim, name, None)
        if callable(getter):
            try:
                alpha = float(getter())
                alpha_source = f"getter:{name}"
            except Exception:
                alpha = None
            break
    for name in ("get_beta", "get_sideslip_angle"):
        getter = getattr(sim, name, None)
        if callable(getter):
            try:
                beta = float(getter())
                beta_source = f"getter:{name}"
            except Exception:
                beta = None
            break

    if alpha is None:
        alpha, alpha_source = _read_jsbsim_angle_property(
            sim, rad_name="aero/alpha-rad", deg_name="aero/alpha-deg",
            fallback_source="placeholder:0")
    if beta is None:
        beta, beta_source = _read_jsbsim_angle_property(
            sim, rad_name="aero/beta-rad", deg_name="aero/beta-deg",
            fallback_source="placeholder:0")

    return (
        float(alpha if alpha is not None else 0.0),
        float(beta if beta is not None else 0.0),
        alpha_source,
        beta_source,
    )


def _get_alpha_beta_placeholder(sim) -> tuple[float, float]:
    """Return alpha/beta if exposed by simulator, otherwise placeholder zeros."""
    alpha, beta, _alpha_source, _beta_source = _get_alpha_beta_with_source(sim)
    return alpha, beta


def _read_jsbsim_angle_property(sim, rad_name: str, deg_name: str,
                                fallback_source: str) -> tuple[float | None, str]:
    getter = getattr(sim, "get_property_value", None)
    if not callable(getter):
        return None, fallback_source
    try:
        return float(getter(rad_name)), f"jsbsim:{rad_name}"
    except Exception:
        pass
    try:
        return float(np.deg2rad(getter(deg_name))), f"jsbsim:{deg_name}"
    except Exception:
        return None, fallback_source


def extract_self_state(sim) -> np.ndarray:
    """Extract paper Table 1 style self state.

    Output: [x, y, h, V, roll, pitch, heading, alpha, beta, Vd].
    ``Vd`` is down velocity; with NEU z-up velocity, ``Vd = -v_up``.
    Alpha/beta are placeholder zeros unless the simulator exposes getters.
    """
    return extract_self_state_with_meta(sim)[0]


def extract_self_state_with_meta(sim) -> tuple[np.ndarray, dict]:
    """Extract self state and alpha/beta source diagnostics."""
    position = np.asarray(sim.get_position(), dtype=np.float64)
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, heading = np.asarray(sim.get_rpy(), dtype=np.float64)
    alpha, beta, alpha_source, beta_source = _get_alpha_beta_with_source(sim)

    speed = float(np.linalg.norm(velocity))
    down_velocity = float(-velocity[2])
    state = np.array([
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
    return state, {
        "alpha_source": alpha_source,
        "beta_source": beta_source,
    }


def extract_relative_state(observer_sim, target_sim,
                           radar_detected: bool = True) -> np.ndarray:
    """Extract paper Table 2 style relative observation.

    Output:
    [x_body, y_body, z_body, theta_v_body, psi_v_body, V_target,
     theta_LOS_body, psi_LOS_body, q_LOS, d]

    ``q_LOS`` is the three-dimensional angle between observer velocity and the
    observer-to-target LOS.  This keeps directional q_ij/q_ji semantics
    consistent across Table 2 and Eq.20-22.
    """
    obs_pos = np.asarray(observer_sim.get_position(), dtype=np.float64)
    tgt_pos = np.asarray(target_sim.get_position(), dtype=np.float64)
    obs_vel = np.asarray(observer_sim.get_velocity(), dtype=np.float64)
    tgt_vel = np.asarray(target_sim.get_velocity(), dtype=np.float64)
    roll, pitch, heading = np.asarray(observer_sim.get_rpy(), dtype=np.float64)
    rel_pos_body, theta_los_body, psi_los_body, _body_axis_q_los = \
        body_angles_from_neu_vector(
            tgt_pos - obs_pos, roll, pitch, heading)
    x_body, y_body, z_body = [float(v) for v in rel_pos_body]
    d = float(np.linalg.norm(rel_pos_body))

    rel_vel_body, theta_v_body, psi_v_body, _ = \
        body_angles_from_neu_vector(
            tgt_vel - obs_vel, roll, pitch, heading)
    target_speed = float(np.linalg.norm(tgt_vel))
    los_neu = tgt_pos - obs_pos
    los_norm = float(np.linalg.norm(los_neu))
    obs_speed = float(np.linalg.norm(obs_vel))
    if los_norm <= 1e-8 or obs_speed <= 1e-8:
        q_los = 0.0
    else:
        q_los = float(np.arccos(np.clip(
            float(np.dot(obs_vel, los_neu)) / (obs_speed * los_norm), -1.0, 1.0)))

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


def compute_body_x_q_los_from_body(rel_pos_body: np.ndarray) -> float:
    """Return the body-x to observer-to-target LOS angle in ``[0, pi]``."""
    rel_pos_body = np.asarray(rel_pos_body, dtype=np.float64)
    distance = float(np.linalg.norm(rel_pos_body))
    if distance <= 1e-8:
        return 0.0
    return float(np.arccos(np.clip(float(rel_pos_body[0]) / distance, -1.0, 1.0)))


def compute_q_los_placeholder(rel_pos_body: np.ndarray) -> float:
    """Deprecated compatibility alias for ``compute_body_x_q_los_from_body``."""
    return compute_body_x_q_los_from_body(rel_pos_body)


def ordered_entity_slots(env, agent_id: str):
    """Return ``(ego, allies, enemies)`` slots in observation tensor order."""
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
    if agent_id not in own_planes:
        raise KeyError(f"Unknown agent_id: {agent_id}")
    ego = [(agent_id, own_planes[agent_id])]
    allies = [(aid, own_planes[aid]) for aid in own_ids if aid != agent_id]
    enemies = [(aid, enemy_planes[aid]) for aid in enemy_ids]
    return ego, allies, enemies


def slot_aligned_alive_mask(env, agent_id: str) -> np.ndarray:
    """Return 1=valid/alive in exact ``ego, allies, enemies`` slot order."""
    ego, allies, enemies = ordered_entity_slots(env, agent_id)
    ego_valid = _is_valid_sim(ego[0][1])
    return np.asarray([
        int(ego_valid and _is_valid_sim(sim))
        for _aid, sim in ego + allies + enemies
    ], dtype=np.int64)


def _is_valid_sim(sim) -> bool:
    return bool(sim is not None and getattr(sim, "is_alive", False))


def build_strict_paper_entity_observation(env, agent_id: str):
    """Build the paper Table 1/Table 2 entity observation from env state.

    Returns:
        entities: shape (N_entities, 10)
        mask: shape (N_entities,), where 1 means invalid/dead and 0 means valid
        meta: schema and placeholder warnings
    """
    ego_sim = env._get_sim(agent_id) if hasattr(env, "_get_sim") else None
    if ego_sim is None:
        raise KeyError(f"Unknown agent_id: {agent_id}")

    ego_slots, allies, enemies = ordered_entity_slots(env, agent_id)
    rows = []
    mask = []

    if _is_valid_sim(ego_sim):
        self_state, self_meta = extract_self_state_with_meta(ego_sim)
        rows.append(self_state)
        mask.append(0)
    else:
        self_meta = {
            "alpha_source": "placeholder:0",
            "beta_source": "placeholder:0",
        }
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
    expected_mask = 1 - slot_aligned_alive_mask(env, agent_id)
    if not np.array_equal(entity_mask, expected_mask):
        raise RuntimeError("strict entity mask does not match entity slot order")
    meta = {
        "entity_dim": 10,
        "schema": "paper_table1_table2_v1",
        "alpha_beta": {
            "alpha_source": self_meta["alpha_source"],
            "beta_source": self_meta["beta_source"],
        },
        "q_los": "observer_velocity_to_target_los_3d_angle_interpretation",
        "radar_detected": radar_mode,
        "layout": {
            "n_ego": 1,
            "n_allies": len(allies),
            "n_enemies": len(enemies),
            "n_entities": int(entities.shape[0]),
        },
    }
    return entities, entity_mask, meta


def describe_paper_entities(entities: np.ndarray, mask: np.ndarray,
                            meta: dict | None = None) -> str:
    """Return a readable diagnostic dump for paper-style entity tensors."""
    entities = np.asarray(entities)
    mask = np.asarray(mask)
    lines = [
        f"entities.shape: {tuple(entities.shape)}",
        f"mask: {mask.tolist()}",
    ]
    if entities.shape[0] > 0:
        names = ["x", "y", "h", "V", "roll", "pitch",
                 "heading", "alpha", "beta", "Vd"]
        values = ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in zip(names, entities[0])
        )
        lines.append(f"self[0]: {values}")

    rel_names = [
        "x_body", "y_body", "z_body", "theta_v_body", "psi_v_body",
        "V", "theta_LOS_body", "psi_LOS_body", "q_LOS", "d",
    ]
    for idx in range(1, entities.shape[0]):
        values = ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in zip(rel_names, entities[idx])
        )
        lines.append(f"relative[{idx}]: {values}")

    if meta is not None:
        if "alpha_beta" in meta:
            alpha_beta = meta["alpha_beta"]
            if isinstance(alpha_beta, dict):
                lines.append(f"meta.alpha_source: {alpha_beta.get('alpha_source')}")
                lines.append(f"meta.beta_source: {alpha_beta.get('beta_source')}")
            else:
                lines.append(f"meta.alpha_beta: {alpha_beta}")
        if "q_los" in meta:
            lines.append(f"meta.q_los: {meta['q_los']}")
    return "\n".join(lines)
