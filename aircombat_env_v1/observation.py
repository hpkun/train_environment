"""Fixed 20-dimensional observation construction."""

from __future__ import annotations

import numpy as np

from .combat import relative_geometry


def build_observation(red_state, blue_state, red_dwell, blue_dwell):
    geometry = relative_geometry(red_state, blue_state)
    body = geometry["relative_body"] / 10000.0
    relative_heading = geometry["relative_heading"]
    values = np.array([
        (red_state["altitude"] - 6000.0) / 4000.0,
        (red_state["true_airspeed"] - 250.0) / 100.0,
        np.sin(red_state["roll"]), np.cos(red_state["roll"]),
        np.sin(red_state["pitch"]), np.cos(red_state["pitch"]),
        body[0], body[1], body[2],
        geometry["distance_m"] / 10000.0,
        geometry["closure_mps"] / 300.0,
        (blue_state["true_airspeed"] - 250.0) / 100.0,
        np.sin(relative_heading), np.cos(relative_heading),
        red_dwell, blue_dwell,
        np.sin(red_state["heading"]), np.cos(red_state["heading"]),
        -red_state["v_down"] / 100.0,
        -blue_state["v_down"] / 100.0,
    ], dtype=np.float32)
    return np.clip(values, -1.0, 1.0).astype(np.float32)
