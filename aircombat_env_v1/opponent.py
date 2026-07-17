"""Finite-manoeuvre greedy opponent adapted from the TAM paper environment."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .combat import DECISION_DT, pursuit_action
from .geometry import in_range_rad


MANOEUVRE_NAMES = (
    "level_hold", "pursuit", "left_turn", "right_turn",
    "climb", "descend", "accelerate", "decelerate",
)


def height_reward(altitude_m):
    """Engineering approximation because the paper gives no height curve."""
    altitude = float(altitude_m)
    if altitude < 750.0:
        return -1.0
    if altitude <= 6000.0:
        return float(1.0 - ((6000.0 - altitude) / 5250.0) ** 2)
    return float(np.clip(1.0 - ((altitude - 6000.0) / 4000.0) ** 2,
                         -1.0, 1.0))


def speed_reward(own_speed_mps, target_speed_mps):
    own = max(float(own_speed_mps), 1e-6)
    target = float(target_speed_mps)
    if target < 0.5 * own:
        return 1.0
    if target <= 1.5 * own:
        return float(2.0 - 2.0 * target / own)
    return -1.0


def angle_reward(ata_rad, aa_rad):
    return float(1.0 - (float(ata_rad) + float(aa_rad)) / np.pi)


def distance_reward(distance_m):
    distance_km = float(distance_m) / 1000.0
    if distance_km <= 5.0:
        return 1.0
    if distance_km < 10.0:
        return float(np.exp(-0.921 * (distance_km - 5.0)))
    return -1.0


def candidate_actions(own_state, target_state):
    """Return the deterministic engineering mapping to the 3D target action."""
    pursuit = pursuit_action(own_state, target_state)
    return OrderedDict((
        ("level_hold", np.array([0.0, 0.0, 0.0], np.float32)),
        ("pursuit", pursuit.astype(np.float32)),
        ("left_turn", np.array([0.0, -0.35, 0.0], np.float32)),
        ("right_turn", np.array([0.0, 0.35, 0.0], np.float32)),
        ("climb", np.array([0.25, 0.0, 0.0], np.float32)),
        ("descend", np.array([-0.25, 0.0, 0.0], np.float32)),
        ("accelerate", np.array([0.0, 0.0, 1.0], np.float32)),
        ("decelerate", np.array([0.0, 0.0, -0.4], np.float32)),
    ))


def _velocity(speed, pitch, heading):
    cp = np.cos(pitch)
    return np.array([
        speed * cp * np.cos(heading), speed * cp * np.sin(heading),
        -speed * np.sin(pitch),
    ], dtype=float)


def _angle(a, b):
    return float(np.arccos(np.clip(
        np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8),
        -1.0, 1.0)))


def paper_structured_engineering_score(own_state, target_state, action):
    """Score one 0.2 s high-level candidate using paper reward structure.

    The component formulas and weights follow the TAM repository. The target
    command prediction and safety penalties are project engineering choices.
    """
    action = np.asarray(action, dtype=float)
    pitch = float(action[0] * np.deg2rad(20.0))
    heading = float(in_range_rad(
        own_state["heading"] + action[1] * np.deg2rad(60.0)))
    speed = float(250.0 + action[2] * 50.0)
    velocity = _velocity(speed, pitch, heading)
    target_velocity = np.array([
        target_state["v_north"], target_state["v_east"],
        target_state["v_down"],
    ], dtype=float)
    # The current LOS is reconstructed from range/angles by the caller's
    # states only through geodetic conversion in combat; defer that import to
    # avoid a module cycle.
    from .combat import relative_geometry
    geometry = relative_geometry(own_state, target_state)
    # Candidate own velocity must be used here: r_next = r + (v_t-v_c)dt.
    los = geometry["relative_ned"] + (target_velocity - velocity) * DECISION_DT
    predicted_altitude = own_state["altitude"] - velocity[2] * DECISION_DT
    ata, aa = _angle(velocity, los), _angle(target_velocity, los)
    score = (
        10.0 * height_reward(predicted_altitude)
        + 10.0 * speed_reward(speed, target_state["true_airspeed"])
        + 15.0 * angle_reward(ata, aa)
        + 10.0 * distance_reward(np.linalg.norm(los))
    )
    # Explicit engineering safety guard: immediate greedy prediction alone
    # cannot see a ground impact hundreds of decisions ahead.
    if own_state["altitude"] < 3500.0 and pitch < 0.0:
        score -= 100.0
    if own_state["altitude"] < 1800.0 and pitch <= 0.0:
        score -= 200.0
    if own_state["true_airspeed"] < 180.0 and speed < 250.0:
        score -= 200.0
    if abs(own_state.get("load_factor", 1.0)) > 8.0 and abs(action[1]) > 0.1:
        score -= 100.0
    return float(score)


def paper_greedy_action(own_state, target_state, return_details=False):
    actions = candidate_actions(own_state, target_state)
    scores = OrderedDict(
        (name, paper_structured_engineering_score(
            own_state, target_state, action))
        for name, action in actions.items())
    selected = max(
        actions, key=lambda name: (scores[name], -MANOEUVRE_NAMES.index(name)))
    action = actions[selected].copy()
    if return_details:
        return action, {"manoeuvre": selected, "scores": dict(scores)}
    return action
