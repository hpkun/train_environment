"""Short deterministic trajectory audit for paper_learnable_point_mass_v1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_learnable_3v3_spec import (
    LEARNABLE_PAPER_ENVIRONMENT_CONFIG,
)
from my_uav_env.simulator import MissileSimulator


class _Aircraft:
    dt = 1.0 / 60.0
    lon0, lat0, alt0 = 120.0, 60.0, 0.0
    is_alive = True

    def __init__(self, uid, color, position, velocity, heading):
        self.uid = uid
        self.color = color
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.rpy = np.array([0.0, 0.0, heading], dtype=np.float64)
        self.launch_missiles = []
        self.under_missiles = []

    def get_geodetic(self):
        return np.array([120.0, 60.0, self.position[2]])

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_rpy(self):
        return self.rpy

    def shotdown(self):
        self.is_alive = False


def audit_range(range_m: float) -> dict:
    parent = _Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0], 0.0)
    target = _Aircraft(
        "blue_0", "Blue", [range_m, 0.0, 6000.0], [-300.0, 0.0, 0.0],
        math.pi)
    samples = []
    missile = MissileSimulator.create(
        parent, target, f"m{int(range_m)}",
        guidance_mode="paper_learnable_point_mass_v1",
        config=LEARNABLE_PAPER_ENVIRONMENT_CONFIG.missile,
        rng=np.random.default_rng(0), launch_speed_mps=800.0,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    missile._trajectory_sink = samples.append
    while not missile.is_done:
        missile.run()
    speeds = [float(row["missile_speed_mps"]) for row in samples]
    commands = [
        math.hypot(float(row["PN_lateral_command_g"]),
                   float(row["PN_vertical_command_g"]))
        for row in samples]
    return {
        "range_m": float(range_m),
        "termination_reason": missile._termination_reason,
        "flight_time_s": float(missile._t),
        "frames": int(missile._pn_guidance_frames),
        "nonzero_pn_frames": int(missile._pn_nonzero_command_frames),
        "maximum_command_g": max(commands) if commands else 0.0,
        "maximum_speed_error_mps": max(
            (abs(speed - 800.0) for speed in speeds), default=0.0),
        "finite": bool(all(np.isfinite(value) for value in speeds + commands)),
        "one_frame_termination": bool(missile._pn_guidance_frames <= 1),
        "lifetime_over_0_2_s": bool(missile._t > 0.2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = [audit_range(value) for value in (1000.0, 3000.0, 5000.0, 8000.0)]
    report = {
        "profile": "paper_learnable_point_mass_v1",
        "rows": rows,
        "all_finite": all(row["finite"] for row in rows),
        "no_one_frame_termination": all(
            not row["one_frame_termination"] for row in rows),
        "median_lifetime_s": float(np.median([
            row["flight_time_s"] for row in rows])),
        "some_lifetime_over_0_2_s": any(
            row["lifetime_over_0_2_s"] for row in rows),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
