"""Tune F22 MAV 3D PID profile via grid sweep of PID gains and elevator sign.

Runs a small F22-only environment with fixed actions, scores each candidate
profile on speed maintenance, altitude safety, and control surface health.
Writes the best profile to outputs/environment_audit/f22_pid_profile_tuning.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product

import numpy as np

# Ensure the hetero_uav package is importable
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput
from uav_env.JSBSim.pid_controller import F22MavEnergyPIDController

# ---- Search space ----
ELEVATOR_SIGNS = [+1, -1]
PITCH_KP_CANDIDATES = [0.8, 1.2, 1.8]
ROLL_KP_CANDIDATES = [0.08, 0.12, 0.15]
VEL_KP_CANDIDATES = [0.04, 0.06, 0.08]

# F22-specific fixed gains (not swept)
F22_ROLL_KI = 0.45
F22_ROLL_KD = 0.04
F22_PITCH_KI = 0.45
F22_PITCH_KD = 0.08
F22_VEL_KI = 0.012
F22_VEL_KD = 0.003
F22_THROTTLE_MIN = 0.65
F22_THROTTLE_MAX = 1.0
F22_LOW_SPEED_THROTTLE_FLOOR = 0.90

# Simulation settings
SIM_FREQ = 60
PHYSICS_DT = 1.0 / SIM_FREQ
AGENT_INTERACTION_STEPS = 12
SIM_TIME_SEC = 60.0
NUM_PHYSICS_FRAMES = int(SIM_TIME_SEC * SIM_FREQ)
NUM_ENV_STEPS = NUM_PHYSICS_FRAMES // AGENT_INTERACTION_STEPS

# Test actions (normalised [-1, 1])
# [target_pitch, target_heading, target_velocity]
FIXED_ACTIONS = [
    ("level",             np.array([0.0,  0.0,   0.3],  dtype=np.float32)),
    ("speed_up",          np.array([0.0,  0.0,   1.0],  dtype=np.float32)),
    ("shallow_climb",     np.array([0.2,  0.0,   0.5],  dtype=np.float32)),
    ("shallow_turn_left", np.array([0.0, -0.3,   0.5],  dtype=np.float32)),
    ("shallow_turn_right",np.array([0.0,  0.3,   0.5],  dtype=np.float32)),
]

OUTPUT_DIR = os.path.join(_root, "outputs", "environment_audit")


def _action_to_targets(act: np.ndarray):
    """Map normalised [-1,1]^3 to physical setpoints (same as env._parse_actions layer 3)."""
    PITCH_DEG = 90.0
    VELOCITY_MIN = 102.0
    VELOCITY_MAX = 408.0
    target_pitch = float(act[0]) * np.deg2rad(PITCH_DEG)
    target_heading = float(act[1]) * np.pi
    target_velocity = VELOCITY_MIN + (float(act[2]) + 1.0) / 2.0 * (VELOCITY_MAX - VELOCITY_MIN)
    return target_pitch, target_heading, target_velocity


def run_one_candidate(elevator_sign, pitch_kp, roll_kp, vel_kp, rng_seed=42):
    """Run all fixed actions with one PID candidate. Returns score dict."""
    from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput

    with SuppressOutput():
        sim = AircraftSimulator(
            uid="red_0", color="Red", model="f22",
            sim_freq=SIM_FREQ, num_missiles=0,
            init_state={
                "latitude_deg": 59.98, "longitude_deg": 120.02,
                "altitude_ft": 6000.0 / 0.3048,
                "heading_deg": 0.0,
                "speed_fps": 250.0 / 0.3048,
            },
            suppress_jsbsim_output=True,
        )

    pid = F22MavEnergyPIDController(
        PHYSICS_DT, debug=False,
        roll_kp=roll_kp, roll_ki=F22_ROLL_KI, roll_kd=F22_ROLL_KD,
        pitch_kp=pitch_kp, pitch_ki=F22_PITCH_KI, pitch_kd=F22_PITCH_KD,
        vel_kp=vel_kp, vel_ki=F22_VEL_KI, vel_kd=F22_VEL_KD,
        elevator_sign=elevator_sign,
        throttle_min=F22_THROTTLE_MIN,
        throttle_max=F22_THROTTLE_MAX,
        low_speed_throttle_floor=F22_LOW_SPEED_THROTTLE_FLOOR,
    )

    rng = np.random.default_rng(rng_seed)
    all_results = []

    for action_name, act in FIXED_ACTIONS:
        with SuppressOutput():
            sim.reload(new_state={
                "latitude_deg": 59.98, "longitude_deg": 120.02,
                "altitude_ft": 6000.0 / 0.3048,
                "heading_deg": 0.0,
                "speed_fps": 250.0 / 0.3048,
            })
        pid.reset()

        speeds = []
        altitudes = []
        surface_saturations = []  # fraction of frames where any surface is |val| > 0.98
        crashed = False
        nonfinite = False
        energy_guard_activations = 0

        for step in range(NUM_ENV_STEPS):
            target_pitch, target_heading, target_velocity = _action_to_targets(act)

            for _ in range(AGENT_INTERACTION_STEPS):
                if not sim.is_alive:
                    crashed = True
                    break

                rpy = sim.get_rpy()
                vel = sim.get_velocity()
                current_speed = float(np.linalg.norm(vel))
                vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

                ail, elev, rud, thr = pid.compute_control(
                    rpy, current_speed,
                    target_pitch, target_heading, target_velocity,
                    ned_velocity=vel_ned,
                )

                # Check nonfinite
                if not (np.isfinite(ail) and np.isfinite(elev)
                        and np.isfinite(rud) and np.isfinite(thr)):
                    nonfinite = True
                    break

                # Clamp and set
                ail = float(np.clip(ail, -1, 1))
                elev = float(np.clip(elev, -1, 1))
                rud = float(np.clip(rud, -1, 1))
                thr = float(np.clip(thr, 0, 1))

                sim.set_property_value("fcs/aileron-cmd-norm", ail)
                sim.set_property_value("fcs/elevator-cmd-norm", elev)
                sim.set_property_value("fcs/rudder-cmd-norm", rud)
                sim.set_property_value("fcs/throttle-cmd-norm", thr)

                sim.run()

                sat_count = sum(1 for v in [ail, elev, thr] if abs(v) > 0.98)
                surface_saturations.append(sat_count / 3.0)

                if pid.last_energy_guard_active:
                    energy_guard_activations += 1

            if crashed or nonfinite:
                break

            alt_m = sim.get_geodetic()[2]
            vel = sim.get_velocity()
            speed = float(np.linalg.norm(vel))
            speeds.append(speed)
            altitudes.append(alt_m)

        # ---- Score this action ----
        if crashed:
            score = -1000.0
        elif nonfinite or len(speeds) == 0:
            score = -2000.0
        else:
            mean_speed = float(np.mean(speeds))
            min_alt = float(np.min(altitudes))
            mean_saturation = float(np.mean(surface_saturations))
            speed_penalty = max(0.0, 200.0 - mean_speed) * 0.5
            alt_penalty = max(0.0, 5000.0 - min_alt) * 2.0
            sat_penalty = mean_saturation * 100.0
            eg_bonus = -min(energy_guard_activations * 0.01, 5.0)  # fewer is better
            score = mean_speed - speed_penalty - alt_penalty - sat_penalty + eg_bonus

        all_results.append({
            "action": action_name,
            "crashed": crashed,
            "nonfinite": nonfinite,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "min_altitude": float(np.min(altitudes)) if altitudes else 0.0,
            "mean_saturation": float(np.mean(surface_saturations)) if surface_saturations else 1.0,
            "energy_guard_activations": energy_guard_activations,
            "score": float(score),
        })

    # Aggregate score across actions
    total_score = sum(r["score"] for r in all_results)
    any_crashed = any(r["crashed"] for r in all_results)
    any_nonfinite = any(r["nonfinite"] for r in all_results)

    return {
        "params": {
            "elevator_sign": elevator_sign,
            "pitch_kp": pitch_kp,
            "roll_kp": roll_kp,
            "vel_kp": vel_kp,
            "roll_ki": F22_ROLL_KI, "roll_kd": F22_ROLL_KD,
            "pitch_ki": F22_PITCH_KI, "pitch_kd": F22_PITCH_KD,
            "vel_ki": F22_VEL_KI, "vel_kd": F22_VEL_KD,
            "throttle_min": F22_THROTTLE_MIN,
            "throttle_max": F22_THROTTLE_MAX,
            "low_speed_throttle_floor": F22_LOW_SPEED_THROTTLE_FLOOR,
        },
        "total_score": total_score,
        "any_crashed": any_crashed,
        "any_nonfinite": any_nonfinite,
        "per_action": all_results,
    }


def determine_elevator_sign(pid_controller, gain_kwargs) -> int:
    """Determine the F22 elevator sign through a quick level-flight test.

    Runs level flight at [0, 0, 0.3] (slight speed-up) for 2 seconds
    with elevator_sign=+1 and elevator_sign=-1. The sign that produces
    a higher mean speed (implying the correct pitch tracking and less
    energy loss) is selected.

    If both signs are stable (difference < 15 m/s), defaults to +1
    (F-22 convention: positive elevator → pitch UP).
    """
    from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput

    results = {}
    for sign in [+1, -1]:
        kwargs = dict(gain_kwargs)
        kwargs["elevator_sign"] = sign
        with SuppressOutput():
            sim = AircraftSimulator(
                uid="red_0", color="Red", model="f22",
                sim_freq=SIM_FREQ, num_missiles=0,
                init_state={
                    "latitude_deg": 59.98, "longitude_deg": 120.02,
                    "altitude_ft": 6000.0 / 0.3048,
                    "heading_deg": 0.0,
                    "speed_fps": 250.0 / 0.3048,
                },
                suppress_jsbsim_output=True,
            )
        pid = pid_controller(**kwargs)
        act = np.array([0.0, 0.0, 0.3], dtype=np.float32)  # level, slight speed
        target_pitch, target_heading, target_velocity = _action_to_targets(act)
        speeds = []
        crashed = False

        NUM_TEST_FRAMES = int(2.0 * SIM_FREQ)
        for _ in range(NUM_TEST_FRAMES // AGENT_INTERACTION_STEPS):
            for _ in range(AGENT_INTERACTION_STEPS):
                if not sim.is_alive:
                    crashed = True
                    break
                rpy = sim.get_rpy()
                vel = sim.get_velocity()
                current_speed = float(np.linalg.norm(vel))
                vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
                ail, elev, rud, thr = pid.compute_control(
                    rpy, current_speed,
                    target_pitch, target_heading, target_velocity,
                    ned_velocity=vel_ned,
                )
                ail = float(np.clip(ail, -1, 1))
                elev = float(np.clip(elev, -1, 1))
                thr = float(np.clip(thr, 0, 1))
                sim.set_property_value("fcs/aileron-cmd-norm", ail)
                sim.set_property_value("fcs/elevator-cmd-norm", elev)
                sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
                sim.set_property_value("fcs/throttle-cmd-norm", thr)
                sim.run()
            if crashed:
                break
            vel = sim.get_velocity()
            speeds.append(float(np.linalg.norm(vel)))

        if crashed:
            results[sign] = -1e9
        else:
            results[sign] = float(np.mean(speeds)) if speeds else 0.0

    # Choose the sign with higher mean speed
    if results[+1] > results[-1] + 15.0:
        chosen = +1
    elif results[-1] > results[+1] + 15.0:
        chosen = -1
    else:
        # Both similar — default to +1 (F-22 convention)
        chosen = +1

    return chosen, results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Determine elevator sign
    print("=== F22 PID Profile Tuning ===\n")
    print("[1/3] Determining F22 elevator sign...")
    default_kwargs = dict(
        dt=PHYSICS_DT, debug=False,
        roll_kp=0.12, roll_ki=F22_ROLL_KI, roll_kd=F22_ROLL_KD,
        pitch_kp=1.2, pitch_ki=F22_PITCH_KI, pitch_kd=F22_PITCH_KD,
        vel_kp=0.06, vel_ki=F22_VEL_KI, vel_kd=F22_VEL_KD,
        throttle_min=F22_THROTTLE_MIN,
        throttle_max=F22_THROTTLE_MAX,
        low_speed_throttle_floor=F22_LOW_SPEED_THROTTLE_FLOOR,
    )
    chosen_sign, sign_results = determine_elevator_sign(
        F22MavEnergyPIDController, default_kwargs
    )
    print(f"  sign=+1 mean speed: {sign_results[+1]:.1f} m/s")
    print(f"  sign=-1 mean speed: {sign_results[-1]:.1f} m/s")
    print(f"  Chosen elevator_sign = {chosen_sign}\n")

    # Step 2: Grid sweep
    print("[2/3] Running PID gain grid sweep...")
    candidates = list(product(
        ELEVATOR_SIGNS if chosen_sign is None else [chosen_sign],
        PITCH_KP_CANDIDATES,
        ROLL_KP_CANDIDATES,
        VEL_KP_CANDIDATES,
    ))
    print(f"  Total candidates: {len(candidates)}")

    best = None
    best_score = -1e9
    all_candidate_results = []

    for idx, (elev_sign, pkp, rkp, vkp) in enumerate(candidates):
        print(f"  [{idx + 1}/{len(candidates)}] "
              f"elev_sign={elev_sign:+d} pkp={pkp:.2f} rkp={rkp:.2f} vkp={vkp:.2f} ...",
              end=" ", flush=True)
        t0 = time.time()
        result = run_one_candidate(elev_sign, pkp, rkp, vkp)
        elapsed = time.time() - t0
        all_candidate_results.append(result)

        status = ""
        if result["any_crashed"]:
            status = "CRASHED"
        elif result["any_nonfinite"]:
            status = "NONFINITE"
        else:
            status = f"score={result['total_score']:.1f}"

        print(f"{status} ({elapsed:.1f}s)")

        if not result["any_crashed"] and not result["any_nonfinite"]:
            if result["total_score"] > best_score:
                best_score = result["total_score"]
                best = result

    # Step 3: Write outputs
    print("\n[3/3] Writing outputs...")

    if best is None:
        print("  ERROR: No valid candidate found! Writing fallback defaults.")
        best = {
            "params": {
                "elevator_sign": chosen_sign,
                "pitch_kp": 1.2, "pitch_ki": F22_PITCH_KI, "pitch_kd": F22_PITCH_KD,
                "roll_kp": 0.12, "roll_ki": F22_ROLL_KI, "roll_kd": F22_ROLL_KD,
                "vel_kp": 0.06, "vel_ki": F22_VEL_KI, "vel_kd": F22_VEL_KD,
                "throttle_min": F22_THROTTLE_MIN,
                "throttle_max": F22_THROTTLE_MAX,
                "low_speed_throttle_floor": F22_LOW_SPEED_THROTTLE_FLOOR,
            },
            "total_score": -9999,
            "any_crashed": True,
            "any_nonfinite": True,
        }

    # JSON output
    tuning_json_path = os.path.join(OUTPUT_DIR, "f22_pid_profile_tuning.json")
    with open(tuning_json_path, "w") as f:
        json.dump({
            "best_profile": best,
            "all_candidates": all_candidate_results,
            "elevator_sign_test": {
                "chosen": chosen_sign,
                "results": {str(k): v for k, v in sign_results.items()},
            },
            "search_space": {
                "elevator_signs": ELEVATOR_SIGNS,
                "pitch_kp": PITCH_KP_CANDIDATES,
                "roll_kp": ROLL_KP_CANDIDATES,
                "vel_kp": VEL_KP_CANDIDATES,
            },
        }, f, indent=2)
    print(f"  → {tuning_json_path}")

    # Markdown output
    md_path = os.path.join(OUTPUT_DIR, "f22_pid_profile_tuning.md")
    bp = best["params"]
    lines = [
        "# F22 MAV 3D PID Profile Tuning Result",
        "",
        "## Elevator Sign",
        f"- sign=+1 mean speed: {sign_results[+1]:.1f} m/s",
        f"- sign=-1 mean speed: {sign_results[-1]:.1f} m/s",
        f"- **Chosen: `elevator_sign = {chosen_sign:+d}`**",
        "",
        "## Best PID Gains",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| elevator_sign | {bp['elevator_sign']:+d} |",
        f"| pitch_kp | {bp['pitch_kp']:.3f} |",
        f"| pitch_ki | {bp['pitch_ki']:.3f} |",
        f"| pitch_kd | {bp['pitch_kd']:.3f} |",
        f"| roll_kp | {bp['roll_kp']:.3f} |",
        f"| roll_ki | {bp['roll_ki']:.3f} |",
        f"| roll_kd | {bp['roll_kd']:.3f} |",
        f"| vel_kp | {bp['vel_kp']:.3f} |",
        f"| vel_ki | {bp['vel_ki']:.3f} |",
        f"| vel_kd | {bp['vel_kd']:.3f} |",
        f"| throttle_min | {bp['throttle_min']:.2f} |",
        f"| throttle_max | {bp['throttle_max']:.2f} |",
        f"| low_speed_throttle_floor | {bp['low_speed_throttle_floor']:.2f} |",
        "",
        "## Best Score",
        f"- total_score: {best['total_score']:.1f}",
        f"- any_crashed: {best['any_crashed']}",
        f"- any_nonfinite: {best['any_nonfinite']}",
        "",
        "## Per-Action Results (Best Candidate)",
    ]
    for r in best["per_action"]:
        lines.append(f"- **{r['action']}**: score={r['score']:.1f}, "
                     f"mean_speed={r['mean_speed']:.1f}, "
                     f"min_alt={r['min_altitude']:.0f}, "
                     f"crashed={r['crashed']}, "
                     f"saturation={r['mean_saturation']:.3f}")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {md_path}")

    print("\n=== Tuning complete ===")
    print(f"Best elevator_sign = {bp['elevator_sign']:+d}")
    print(f"Best pitch_kp = {bp['pitch_kp']:.3f}")
    print(f"Best roll_kp  = {bp['roll_kp']:.3f}")
    print(f"Best vel_kp   = {bp['vel_kp']:.3f}")
    print(f"Best score    = {best['total_score']:.1f}")


if __name__ == "__main__":
    main()
