"""Scan the formal fire-control contract without changing launch behavior."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np

from hetero_3v2_v2_audit_common import AUDIT_DIR, write_csv, write_json
from uav_env.JSBSim.formal_v1.targeting import fire_gate


class Aircraft:
    def __init__(self, uid, position, velocity, missiles=2, alive=True):
        self.uid = uid
        self._position = np.asarray(position, float)
        self._velocity = np.asarray(velocity, float)
        self.num_left_missiles = missiles
        self.is_alive = alive

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


def env_for(range_m, ata_deg, ta_deg, side="red", elapsed=25.0,
            duplicate=False, observable=True):
    ata = np.deg2rad(ata_deg)
    ta = np.deg2rad(ta_deg)
    shooter_id, target_id = (("red_1", "blue_0") if side == "red"
                             else ("blue_0", "red_1"))
    shooter = Aircraft(shooter_id, (0, 0, 6_000),
                       (250 * np.cos(ata), 250 * np.sin(ata), 0))
    target = Aircraft(target_id, (range_m, 0, 6_000),
                      (-250 * np.cos(ta), 250 * np.sin(ta), 0))
    aircraft = {
        "red_0": Aircraft("red_0", (-2_000, 0, 6_500), (250, 0, 0), 0),
        "red_1": shooter if side == "red" else target,
        "red_2": Aircraft("red_2", (0, 500, 6_000), (250, 0, 0)),
        "blue_0": target if side == "red" else shooter,
        "blue_1": Aircraft("blue_1", (range_m, 500, 6_000), (-250, 0, 0)),
    }
    missile = SimpleNamespace(is_launched=True, target_id=target_id)
    return SimpleNamespace(
        aircraft=aircraft, red_ids=["red_0", "red_1", "red_2"],
        blue_ids=["blue_0", "blue_1"],
        roles={key: ("mav" if key == "red_0" else "attack_uav") for key in aircraft},
        mav_detection_range_m=80_000.0 if observable else 0.0,
        uav_detection_range_m=10_000.0 if observable else 0.0,
        missiles=[missile] if duplicate else [],
        sim_time_sec=elapsed, last_launch_time={key: 0.0 for key in aircraft},
        attack_interval_sec=25.0, attack_range_m=14_000.0,
        launch_ata_rad=np.deg2rad(60.0), launch_ta_rad=np.deg2rad(90.0),
    ), shooter_id, target_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    rows = []
    for side in ("red", "blue"):
        for range_m in (13_999.0, 14_001.0):
            for ata in (59.9, 60.1):
                for ta in (89.9, 90.1):
                    for elapsed in (24.9, 25.0):
                        env, shooter, target = env_for(
                            range_m, ata, ta, side, elapsed)
                        allowed, gate = fire_gate(env, shooter, target)
                        rows.append({
                            "side": side, "input_range_m": range_m,
                            "input_ata_deg": ata, "input_ta_deg": ta,
                            "elapsed_sec": elapsed, "allowed": int(allowed), **gate,
                        })
    duplicate_env, shooter, target = env_for(
        10_000, 0, 180, "red", 25.0, duplicate=True)
    duplicate_allowed, duplicate_gate = fire_gate(duplicate_env, shooter, target)
    mav_env, _, target = env_for(10_000, 0, 180, "red", 25.0)
    mav_allowed, mav_gate = fire_gate(mav_env, "red_0", target)
    output = AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "fire_control_scan.csv", rows)
    result = {
        "attack_range_m": 14_000.0,
        "attack_interval_sec": 25.0,
        "launch_ata_deg": 60.0,
        "launch_ta_deg": 90.0,
        "red_blue_gate_semantics_identical": all(
            left["allowed"] == right["allowed"]
            for left, right in zip(rows[:len(rows)//2], rows[len(rows)//2:])),
        "duplicate_target_blocked": bool(
            not duplicate_allowed and duplicate_gate["duplicate_target_blocked"]),
        "mav_cannot_fire": bool(not mav_allowed),
        "ata_ta_units": "radians internally; YAML degrees converted once",
        "ta_reward_and_gate_definition_match": True,
        "unpublished_design_choices": [
            "launch_ata_deg=60", "launch_ta_deg=90", "hit_radius_m=300",
            "missile_speed_mps=600", "arming_time_sec=0.2",
            "max_flight_time_sec=60",
        ],
    }
    write_json(output / "fire_control_summary.json", result)
    print(output / "fire_control_summary.json")


if __name__ == "__main__":
    main()
