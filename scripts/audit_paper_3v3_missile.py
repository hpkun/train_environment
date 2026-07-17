"""Deterministic in-memory trajectory audit for the formal 3V3 missile."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_3v3_spec import (
    PAPER_ENVIRONMENT_CONFIG,
    PAPER_MISSILE_GUIDANCE_MODE,
)
from my_uav_env.simulator import MissileSimulator


class Aircraft:
    dt = 1.0 / 60.0
    lon0, lat0, alt0 = 120.0, 60.0, 6000.0

    def __init__(self, uid, color, position, velocity, heading):
        self.uid = uid
        self.color = color
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.rpy = np.array([0.0, 0.0, heading], dtype=np.float64)
        self.is_alive = True
        self.launch_missiles = []
        self.under_missiles = []

    def get_geodetic(self):
        return np.array([self.lon0, self.lat0, self.position[2]])

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_rpy(self):
        return self.rpy

    def shotdown(self):
        self.is_alive = False

    def advance(self):
        self.position += self.velocity * self.dt


def run_case(name, position, velocity, heading):
    parent = Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0], 0.0)
    target = Aircraft("blue_0", "Blue", position, velocity, heading)
    missile = MissileSimulator.create(
        parent, target, name, guidance_mode=PAPER_MISSILE_GUIDANCE_MODE,
        config=PAPER_ENVIRONMENT_CONFIG.missile,
        rng=np.random.default_rng(0), launch_speed_mps=800.0,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    missile._roll_hit_probability = lambda: True
    launch_guard = None
    finite = True
    speed_error = 0.0
    while not missile.is_done:
        target.advance()
        missile.run()
        launch_guard = missile.is_alive if launch_guard is None else launch_guard
        state = np.concatenate([missile.get_position(), missile.get_velocity()])
        finite &= bool(np.all(np.isfinite(state)))
        speed_error = max(
            speed_error, abs(float(np.linalg.norm(missile.get_velocity())) - 800.0))
    return {
        "scenario": name,
        "termination_reason": missile._termination_reason,
        "finite": finite,
        "launch_frame_no_hit": bool(launch_guard),
        "maximum_speed_error_mps": speed_error,
        "maximum_command_g": float(missile._maximum_command_g),
        "physics_frames": int(round(missile._t / missile.dt)),
    }


def run_audit():
    cases = [
        ("head_on_1km", [1000, 0, 6000], [-300, 0, 0], math.pi),
        ("head_on_3km", [3000, 0, 6000], [-300, 0, 0], math.pi),
        ("head_on_5km", [5000, 0, 6000], [-300, 0, 0], math.pi),
        ("head_on_8km", [8000, 0, 6000], [-300, 0, 0], math.pi),
        ("head_on_10km", [10000, 0, 6000], [-300, 0, 0], math.pi),
        ("off_axis", [4800, 1200, 6700], [-250, 0, 0], math.pi),
        ("tail_chase", [5000, 0, 6000], [200, 0, 0], 0.0),
        ("crossing", [5000, -1000, 6000], [0, 300, 0], math.pi / 2),
    ]
    rows = [run_case(*case) for case in cases]
    passed = all(
        row["finite"] and row["launch_frame_no_hit"]
        and row["maximum_speed_error_mps"] < 1e-6
        and row["maximum_command_g"] <= 30.0 + 1e-9
        for row in rows)
    return {"profile": PAPER_MISSILE_GUIDANCE_MODE,
            "status": "PASS" if passed else "FAIL", "cases": rows}


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
