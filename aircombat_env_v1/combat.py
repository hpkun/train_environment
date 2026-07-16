"""Minimal 1v1 geometry, attack dwell, and pure-pursuit rules."""

from __future__ import annotations

import numpy as np

from .geometry import LLA2NEU, in_range_rad, ned_to_body_matrix
from .scenario import ORIGIN


ATTACK_MIN_M = 500.0
ATTACK_MAX_M = 2500.0
ATTACK_ANGLE_RAD = np.deg2rad(20.0)
ATTACK_DWELL_S = 1.0
DECISION_DT = 0.2


def local_neu(state):
    return LLA2NEU(
        state["longitude"], state["latitude"], state["altitude"], *ORIGIN)


def relative_geometry(own_state, enemy_state):
    own_neu, enemy_neu = local_neu(own_state), local_neu(enemy_state)
    relative_neu = enemy_neu - own_neu
    relative_ned = np.array(
        [relative_neu[0], relative_neu[1], -relative_neu[2]], dtype=float)
    distance = max(float(np.linalg.norm(relative_ned)), 1e-6)
    body = ned_to_body_matrix(
        own_state["roll"], own_state["pitch"], own_state["heading"]) @ relative_ned
    horizontal = max(float(np.hypot(relative_ned[0], relative_ned[1])), 1e-6)
    los_heading = float(np.arctan2(relative_ned[1], relative_ned[0]))
    los_pitch = float(np.arctan2(-relative_ned[2], horizontal))
    boresight = float(np.arccos(np.clip(body[0] / distance, -1.0, 1.0)))
    own_velocity = np.array([
        own_state["v_north"], own_state["v_east"], own_state["v_down"]])
    enemy_velocity = np.array([
        enemy_state["v_north"], enemy_state["v_east"], enemy_state["v_down"]])
    closure = float(-np.dot(relative_ned, enemy_velocity - own_velocity) / distance)
    return {
        "relative_neu": relative_neu,
        "relative_ned": relative_ned,
        "relative_body": body,
        "distance_m": distance,
        "los_heading": los_heading,
        "los_pitch": los_pitch,
        "boresight_angle": boresight,
        "closure_mps": closure,
        "relative_heading": float(in_range_rad(
            enemy_state["heading"] - own_state["heading"])),
    }


def in_attack_zone(geometry, own_alive=True, enemy_alive=True):
    return bool(own_alive and enemy_alive
                and ATTACK_MIN_M <= geometry["distance_m"] <= ATTACK_MAX_M
                and geometry["boresight_angle"] <= ATTACK_ANGLE_RAD)


def update_attack_dwell(current, condition):
    return min(ATTACK_DWELL_S, current + DECISION_DT) if condition else 0.0


def hit_event(red_dwell, blue_dwell):
    red_hit = red_dwell >= ATTACK_DWELL_S
    blue_hit = blue_dwell >= ATTACK_DWELL_S
    if red_hit and blue_hit:
        return "draw_simultaneous_hit"
    if red_hit:
        return "red_hit"
    if blue_hit:
        return "blue_hit"
    return None


def action_to_targets(action, current_heading):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        raise ValueError("action must be a finite vector with shape (3,)")
    action = np.clip(action, -1.0, 1.0)
    pitch = float(action[0] * np.deg2rad(20.0))
    heading = float(in_range_rad(
        current_heading + action[1] * np.deg2rad(60.0)))
    speed = float(250.0 + action[2] * 50.0)
    return pitch, heading, speed


def pursuit_action(own_state, enemy_state):
    geometry = relative_geometry(own_state, enemy_state)
    heading_offset = in_range_rad(
        geometry["los_heading"] - own_state["heading"])
    pitch_action = geometry["los_pitch"] / np.deg2rad(20.0)
    heading_action = heading_offset / np.deg2rad(60.0)
    distance = geometry["distance_m"]
    target_speed = 300.0 if distance > 3000.0 else (
        250.0 if distance > 1500.0 else 230.0)
    speed_action = (target_speed - 250.0) / 50.0
    return np.clip(
        np.array([pitch_action, heading_action, speed_action], dtype=np.float32),
        -1.0, 1.0)
