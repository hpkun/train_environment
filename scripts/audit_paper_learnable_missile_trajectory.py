"""Deterministic moving-target trajectory audit for the learnable missile."""
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


DT = 1.0 / 60.0


class _Aircraft:
    dt = DT
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

    def advance(self):
        self.position += self.velocity * self.dt


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
    cosine = float(np.dot(first, second) / denominator)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def audit_case(name: str, target_position: list[float]) -> dict:
    parent = _Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0], 0.0)
    target = _Aircraft(
        "blue_0", "Blue", target_position, [-300.0, 0.0, 0.0], math.pi)
    initial_target_position = target.position.copy()
    initial_los = target.position - parent.position
    samples: list[dict] = []
    missile = MissileSimulator.create(
        parent, target, name,
        guidance_mode="paper_learnable_point_mass_v1",
        config=LEARNABLE_PAPER_ENVIRONMENT_CONFIG.missile,
        rng=np.random.default_rng(0), launch_speed_mps=800.0,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    missile._roll_hit_probability = lambda: True
    missile._trajectory_sink = samples.append
    initial_velocity = missile.get_velocity().copy()
    while not missile.is_done:
        target.advance()
        missile.run()

    speeds = [float(row["missile_speed_mps"]) for row in samples]
    commands = [
        math.hypot(float(row["PN_lateral_command_g"]),
                   float(row["PN_vertical_command_g"]))
        for row in samples]
    ranges = [float(row["range_m"]) for row in samples]
    missile_positions = [
        np.asarray(row["missile_position"], dtype=np.float64) for row in samples]
    target_positions = [
        np.asarray(row["target_position"], dtype=np.float64) for row in samples]
    velocities = [
        np.asarray(row["missile_velocity"], dtype=np.float64) for row in samples]
    finite_values = speeds + commands + ranges
    finite = (
        all(np.isfinite(value) for value in finite_values)
        and all(np.all(np.isfinite(value)) for value in missile_positions)
        and all(np.all(np.isfinite(value)) for value in target_positions)
        and all(np.all(np.isfinite(value)) for value in velocities)
    )
    direction_changes = [_angle_deg(initial_velocity, value) for value in velocities]
    target_displacement = float(np.linalg.norm(target.position - initial_target_position))
    return {
        "scenario": name,
        "initial_range_m": float(np.linalg.norm(initial_los)),
        "termination_reason": missile._termination_reason,
        "flight_time_s": float(missile._t),
        "physics_frames": len(samples),
        "pn_guidance_frames": int(missile._pn_guidance_frames),
        "pn_nonzero_command_frames": int(missile._pn_nonzero_command_frames),
        "maximum_command_g": max(commands) if commands else 0.0,
        "maximum_speed_error_mps": max(
            (abs(speed - 800.0) for speed in speeds), default=0.0),
        "initial_velocity_to_los_angle_deg": _angle_deg(
            initial_velocity, initial_los),
        "final_velocity_direction_change_deg": (
            direction_changes[-1] if direction_changes else 0.0),
        "maximum_velocity_direction_change_deg": max(
            direction_changes, default=0.0),
        "target_position_updated": bool(target_displacement > 0.0),
        "target_displacement_m": target_displacement,
        "all_state_range_speed_commands_finite": bool(finite),
        "one_frame_termination": len(samples) <= 1,
        "hit_before_arming": bool(
            missile._termination_reason == "hit" and missile._t < 0.25 - 1e-12),
    }


def run_audit() -> dict:
    rows = [
        audit_case("head_on_1km", [1000.0, 0.0, 6000.0]),
        audit_case("head_on_3km", [3000.0, 0.0, 6000.0]),
        audit_case("head_on_5km", [5000.0, 0.0, 6000.0]),
        audit_case("head_on_8km", [8000.0, 0.0, 6000.0]),
        audit_case("off_axis_5km", [4800.0, 1200.0, 6700.0]),
    ]
    off_axis = next(row for row in rows if row["scenario"] == "off_axis_5km")
    report = {
        "profile": "paper_learnable_point_mass_v1",
        "dt_s": DT,
        "target_motion_model": "position_plus_equals_velocity_times_same_dt",
        "hit_probability_override": "forced_success_for_guidance_isolation_only",
        "hit_probability_scope_warning": (
            "This audit validates kinematics and guidance, not statistical hit rate."),
        "rows": rows,
        "all_finite": all(
            row["all_state_range_speed_commands_finite"] for row in rows),
        "constant_speed_ok": all(
            row["maximum_speed_error_mps"] <= 1e-6 for row in rows),
        "command_limit_ok": all(
            row["maximum_command_g"] <= 30.0 + 1e-6 for row in rows),
        "targets_moved": all(row["target_position_updated"] for row in rows),
        "no_one_frame_termination": all(
            not row["one_frame_termination"] for row in rows),
        "no_pre_arming_hit": all(not row["hit_before_arming"] for row in rows),
        "off_axis_nonzero_direction_change": bool(
            off_axis["maximum_velocity_direction_change_deg"] > 1e-6),
    }
    report["MissileTrajectoryHealth"] = "PASS" if all((
        report["all_finite"], report["constant_speed_ok"],
        report["command_limit_ok"], report["targets_moved"],
        report["no_one_frame_termination"], report["no_pre_arming_hit"],
        report["off_axis_nonzero_direction_change"],
    )) else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    encoded = json.dumps(run_audit(), indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
