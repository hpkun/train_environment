"""Compact 26-dimensional aircraft plus weapon observation."""

from __future__ import annotations

import numpy as np

from .combat import local_neu, relative_geometry


def build_observation(red_state, blue_state, weapon=None, *legacy):
    geometry = relative_geometry(red_state, blue_state)
    red_neu = local_neu(red_state)
    relative_neu = geometry["relative_neu"]
    red_velocity = np.array([
        red_state["v_north"], red_state["v_east"], red_state["v_down"]])
    blue_velocity = np.array([
        blue_state["v_north"], blue_state["v_east"], blue_state["v_down"]])
    weapon = weapon if isinstance(weapon, dict) else {}
    values = np.array([
        red_neu[0] / 10000.0, red_neu[1] / 10000.0,
        red_state["altitude"] / 10000.0,
        red_state["true_airspeed"] / 400.0,
        np.sin(red_state["pitch"]), np.cos(red_state["pitch"]),
        np.sin(red_state["heading"]), np.cos(red_state["heading"]),
        np.sin(red_state["roll"]), np.cos(red_state["roll"]),
        np.linalg.norm(red_velocity - blue_velocity) / 400.0,
        (red_state["altitude"] - blue_state["altitude"]) / 10000.0,
        geometry["distance_m"] / 14000.0,
        geometry["ata_rad"] / np.pi, geometry["aa_rad"] / np.pi,
        relative_neu[0] / 14000.0, relative_neu[1] / 14000.0,
        relative_neu[2] / 14000.0,
        weapon.get("own_inventory_ratio", 0.0),
        weapon.get("cooldown_ratio", 0.0),
        weapon.get("enemy_inventory_ratio", 0.0),
        weapon.get("incoming", 0.0),
        weapon.get("incoming_range_ratio", 1.0),
        weapon.get("incoming_closing_ratio", 0.0),
        weapon.get("incoming_los_angle_ratio", 1.0),
        weapon.get("incoming_ttg_ratio", 1.0),
    ], dtype=np.float32)
    return np.clip(values, -1.0, 1.0).astype(np.float32)
