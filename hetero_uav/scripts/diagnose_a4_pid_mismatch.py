from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.env import UavCombatEnv
from uav_env.JSBSim.pid_controller import PIDController
from uav_env.JSBSim.simulator import AircraftSimulator


SCENARIOS = {
    "a4_level": ("A-4", np.array([0.0, 0.0, 0.0], dtype=np.float32), "fixed"),
    "a4_mild_climb": ("A-4", np.array([0.1, 0.0, 0.0], dtype=np.float32), "fixed"),
    "a4_higher_speed_level": ("A-4", np.array([0.0, 0.0, 0.5], dtype=np.float32), "fixed"),
    "a4_bounded_random": ("A-4", np.array([0.0, 0.0, 0.0], dtype=np.float32), "bounded_random"),
    "f16_level": ("f16", np.array([0.0, 0.0, 0.0], dtype=np.float32), "fixed"),
}


def action_to_targets(action: np.ndarray) -> tuple[float, float, float]:
    target_pitch = float(action[0]) * math.radians(UavCombatEnv.PITCH_DEG)
    target_heading = float(action[1]) * math.pi
    target_velocity = UavCombatEnv.VELOCITY_MIN + (float(action[2]) + 1.0) * 0.5 * (
        UavCombatEnv.VELOCITY_MAX - UavCombatEnv.VELOCITY_MIN
    )
    return target_pitch, target_heading, target_velocity


def _safe_property(sim: AircraftSimulator, name: str) -> float:
    try:
        value = sim.get_property_value(name)
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [x for x in values if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def _min(values: list[float]) -> float:
    finite = [x for x in values if np.isfinite(x)]
    return float(np.min(finite)) if finite else float("nan")


def diagnose_scenario(name: str, duration: float = 60.0, seed: int = 0) -> dict:
    model, fixed_action, mode = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    sim = AircraftSimulator(
        uid=name,
        color="Red",
        model=model,
        num_missiles=0,
        sim_freq=60,
        suppress_jsbsim_output=True,
    )
    pid = PIDController(1.0 / 60.0)
    steps = max(1, int(round(duration * 60.0)))
    initial_altitude = float(sim.get_geodetic()[2])
    initial_speed = float(np.linalg.norm(sim.get_velocity()))
    altitudes: list[float] = []
    speeds: list[float] = []
    vertical_velocities: list[float] = []
    throttle_cmds: list[float] = []
    elevator_cmds: list[float] = []
    aileron_cmds: list[float] = []
    alphas: list[float] = []
    nz_values: list[float] = []
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    nan_detected = False

    try:
        for _ in range(steps):
            action = fixed_action
            if mode == "bounded_random":
                action = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
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
            altitude = float(sim.get_geodetic()[2])
            speed = float(np.linalg.norm(vel))
            values = np.concatenate([
                sim.get_geodetic().astype(np.float64),
                sim.get_position().astype(np.float64),
                vel.astype(np.float64),
                np.asarray(rpy, dtype=np.float64),
                np.asarray([aileron, elevator, throttle], dtype=np.float64),
            ])
            nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
            altitudes.append(altitude)
            speeds.append(speed)
            vertical_velocities.append(float(vel[2]))
            throttle_cmds.append(float(throttle))
            elevator_cmds.append(float(elevator))
            aileron_cmds.append(float(aileron))
            alphas.append(_safe_property(sim, "aero/alpha-rad"))
            nz_values.append(_safe_property(sim, "accelerations/n-pilot-z-norm"))
            max_abs_pitch = max(max_abs_pitch, abs(float(rpy[1])))
            max_abs_roll = max(max_abs_roll, abs(float(rpy[0])))
            if not sim.is_alive:
                break

        final_altitude = float(sim.get_geodetic()[2])
        final_speed = float(np.linalg.norm(sim.get_velocity()))
        return {
            "scenario": name,
            "model": model,
            "initial_altitude": initial_altitude,
            "final_altitude": final_altitude,
            "min_altitude": _min(altitudes),
            "initial_speed": initial_speed,
            "final_speed": final_speed,
            "min_speed": _min(speeds),
            "max_abs_pitch": max_abs_pitch,
            "max_abs_roll": max_abs_roll,
            "vertical_velocity_mean": _mean(vertical_velocities),
            "vertical_velocity_min": _min(vertical_velocities),
            "throttle_command_mean": _mean(throttle_cmds),
            "elevator_command_mean": _mean(elevator_cmds),
            "aileron_command_mean": _mean(aileron_cmds),
            "alpha_mean": _mean(alphas),
            "n_pilot_z_norm_mean": _mean(nz_values),
            "crash": bool(sim.is_crash or final_altitude <= 0.0),
            "nan_detected": nan_detected,
        }
    finally:
        sim.close()


def classify(results: dict[str, dict]) -> str:
    a4 = results["a4_level"]
    f16 = results["f16_level"]
    mild = results["a4_mild_climb"]
    fast = results["a4_higher_speed_level"]
    random = results["a4_bounded_random"]
    notes = []
    if a4["final_altitude"] < f16["final_altitude"] - 500.0:
        notes.append("A-4 loses substantially more altitude than f16 under the same level target.")
    if mild["final_altitude"] > a4["final_altitude"]:
        notes.append("A mild positive pitch command recovers much of the A-4 altitude loss.")
    if fast["final_altitude"] > a4["final_altitude"]:
        notes.append("Higher target speed improves A-4 altitude retention, suggesting energy/speed margin matters.")
    if a4["vertical_velocity_mean"] < -1.0:
        notes.append("A-4 level run has sustained negative vertical velocity.")
    if random["max_abs_roll"] > a4["max_abs_roll"] or random["max_abs_pitch"] > a4["max_abs_pitch"]:
        notes.append("Bounded random action increases attitude excursions relative to level flight.")
    if random["min_altitude"] < a4["min_altitude"]:
        notes.append("Random action reduces the minimum altitude margin.")
    notes.append(
        "Most likely causes: A-4 model and BRMA F-16-oriented PID/target-speed defaults are not trimmed for level A-4 flight; "
        "aggressive/random actions further reduce energy and altitude margins."
    )
    return " ".join(notes)


def _print_row(row: dict) -> None:
    print(
        f"{row['scenario']}: model={row['model']} "
        f"initial_altitude={row['initial_altitude']:.3f} "
        f"final_altitude={row['final_altitude']:.3f} "
        f"min_altitude={row['min_altitude']:.3f} "
        f"initial_speed={row['initial_speed']:.3f} "
        f"final_speed={row['final_speed']:.3f} "
        f"min_speed={row['min_speed']:.3f} "
        f"max_abs_pitch={row['max_abs_pitch']:.6f} "
        f"max_abs_roll={row['max_abs_roll']:.6f} "
        f"vertical_velocity_mean={row['vertical_velocity_mean']:.3f} "
        f"vertical_velocity_min={row['vertical_velocity_min']:.3f} "
        f"throttle_command_mean={row['throttle_command_mean']:.3f} "
        f"elevator_command_mean={row['elevator_command_mean']:.3f} "
        f"aileron_command_mean={row['aileron_command_mean']:.3f} "
        f"alpha_mean={row['alpha_mean']:.6f} "
        f"n_pilot_z_norm_mean={row['n_pilot_z_norm_mean']:.6f} "
        f"crash={row['crash']} nan_detected={row['nan_detected']}"
    )


def main() -> None:
    results = {name: diagnose_scenario(name) for name in SCENARIOS}
    for row in results.values():
        _print_row(row)
    print(f"conclusion: {classify(results)}")


if __name__ == "__main__":
    main()
