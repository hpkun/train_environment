from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_a4_pid_mismatch import action_to_targets
from uav_env.JSBSim.pid_controller import PIDController
from uav_env.JSBSim.simulator import AircraftSimulator


FT_PER_M = 1.0 / 0.3048


def _init_state(altitude_m: float | None = None, speed_mps: float | None = None) -> dict:
    state = {}
    if altitude_m is not None:
        state["ic/h-sl-ft"] = float(altitude_m) * FT_PER_M
    if speed_mps is not None:
        state["ic/u-fps"] = float(speed_mps) * FT_PER_M
    return state


def _scenario_specs() -> list[dict]:
    specs = [
        {"name": "baseline", "action": np.array([0.0, 0.0, 0.0], dtype=np.float32)},
    ]
    for altitude in (7000.0, 8000.0, 9000.0):
        specs.append({
            "name": f"initial_altitude_{int(altitude)}m",
            "init_state": _init_state(altitude_m=altitude),
            "action": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        })
    for speed in (260.0, 280.0, 300.0):
        specs.append({
            "name": f"initial_speed_{int(speed)}mps",
            "init_state": _init_state(speed_mps=speed),
            "action": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        })
    for pitch in (0.05, 0.10, 0.15):
        specs.append({
            "name": f"pitch_bias_{pitch:.2f}",
            "action": np.array([pitch, 0.0, 0.0], dtype=np.float32),
        })
    for bound in (0.3, 0.5):
        specs.append({
            "name": f"bounded_random_{bound:.1f}",
            "random_bound": bound,
        })
    return specs


def run_case(spec: dict, duration: float = 60.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sim = AircraftSimulator(
        uid=spec["name"],
        color="Red",
        model="A-4",
        init_state=spec.get("init_state"),
        sim_freq=60,
        num_missiles=0,
        suppress_jsbsim_output=True,
    )
    pid = PIDController(1.0 / 60.0)
    steps = max(1, int(round(duration * 60)))
    decision_interval = 12
    action = np.array(spec.get("action", [0.0, 0.0, 0.0]), dtype=np.float32)

    altitudes = []
    speeds = []
    vertical_velocities = []
    throttle_cmds = []
    elevator_cmds = []
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    nan_detected = False
    initial_altitude = float(sim.get_geodetic()[2])
    initial_speed = float(np.linalg.norm(sim.get_velocity()))

    try:
        for step in range(steps):
            if "random_bound" in spec and step % decision_interval == 0:
                bound = float(spec["random_bound"])
                action = rng.uniform(-bound, bound, size=3).astype(np.float32)
            targets = action_to_targets(action)
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            current_speed = float(np.linalg.norm(vel))
            vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            aileron, elevator, rudder, throttle = pid.compute_control(
                rpy,
                current_speed,
                targets[0],
                targets[1],
                targets[2],
                ned_velocity=vel_ned,
            )
            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            sim.set_property_value("fcs/throttle-cmd-norm", throttle)
            sim.run()

            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            values = np.concatenate([
                sim.get_geodetic().astype(np.float64),
                sim.get_position().astype(np.float64),
                vel.astype(np.float64),
                np.asarray(rpy, dtype=np.float64),
            ])
            nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
            altitudes.append(float(sim.get_geodetic()[2]))
            speeds.append(float(np.linalg.norm(vel)))
            vertical_velocities.append(float(vel[2]))
            throttle_cmds.append(float(throttle))
            elevator_cmds.append(float(elevator))
            max_abs_pitch = max(max_abs_pitch, abs(float(rpy[1])))
            max_abs_roll = max(max_abs_roll, abs(float(rpy[0])))
            if not sim.is_alive:
                break
        final_altitude = float(sim.get_geodetic()[2])
        final_speed = float(np.linalg.norm(sim.get_velocity()))
        return {
            "name": spec["name"],
            "initial_altitude": initial_altitude,
            "final_altitude": final_altitude,
            "min_altitude": float(np.min(altitudes)),
            "altitude_delta": final_altitude - initial_altitude,
            "initial_speed": initial_speed,
            "final_speed": final_speed,
            "min_speed": float(np.min(speeds)),
            "mean_vertical_velocity": float(np.mean(vertical_velocities)),
            "mean_throttle_cmd": float(np.mean(throttle_cmds)),
            "mean_elevator_cmd": float(np.mean(elevator_cmds)),
            "max_abs_pitch": max_abs_pitch,
            "max_abs_roll": max_abs_roll,
            "crashed": bool(sim.is_crash or final_altitude <= 0.0),
            "nan_detected": nan_detected,
        }
    finally:
        sim.close()


def run_all(duration: float = 60.0, seed: int = 0) -> list[dict]:
    return [run_case(spec, duration, seed) for spec in _scenario_specs()]


def recommendations(rows: list[dict]) -> list[str]:
    by_name = {row["name"]: row for row in rows}
    baseline = by_name["baseline"]
    ranked = sorted(
        [row for row in rows if not row["name"].startswith("bounded_random")],
        key=lambda row: row["final_altitude"] - baseline["final_altitude"],
        reverse=True,
    )
    notes = [
        f"Most effective altitude retention in fixed cases: {ranked[0]['name']}.",
        "Least invasive formal option is usually initial altitude/speed adjustment because it does not alter PID or aircraft XML.",
    ]
    pitch_rows = [row for row in rows if row["name"].startswith("pitch_bias")]
    if pitch_rows:
        best_pitch = max(pitch_rows, key=lambda row: row["final_altitude"])
        notes.append(f"Best pitch-bias diagnostic case: {best_pitch['name']}.")
    random_rows = [row for row in rows if row["name"].startswith("bounded_random")]
    if random_rows:
        best_random = max(random_rows, key=lambda row: row["min_altitude"])
        notes.append(f"Best bounded-random minimum altitude: {best_random['name']}.")
    notes.append("No paper evidence was found for MAV GCAS, so GCAS should not be the first fix.")
    return notes


def _print_row(row: dict) -> None:
    print(
        f"{row['name']}: initial_altitude={row['initial_altitude']:.3f} "
        f"final_altitude={row['final_altitude']:.3f} min_altitude={row['min_altitude']:.3f} "
        f"altitude_delta={row['altitude_delta']:.3f} initial_speed={row['initial_speed']:.3f} "
        f"final_speed={row['final_speed']:.3f} min_speed={row['min_speed']:.3f} "
        f"mean_vertical_velocity={row['mean_vertical_velocity']:.3f} "
        f"mean_throttle_cmd={row['mean_throttle_cmd']:.3f} "
        f"mean_elevator_cmd={row['mean_elevator_cmd']:.3f} "
        f"max_abs_pitch={row['max_abs_pitch']:.6f} max_abs_roll={row['max_abs_roll']:.6f} "
        f"crashed={row['crashed']} nan_detected={row['nan_detected']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run_all(args.duration, args.seed)
    for row in rows:
        _print_row(row)
    print("recommendations:")
    for note in recommendations(rows):
        print(f"- {note}")


if __name__ == "__main__":
    main()
