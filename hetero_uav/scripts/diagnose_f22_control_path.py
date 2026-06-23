"""Diagnose F22 vs F16 control path: direct FCS sweep, surface position, reward trace.

Tests whether F22 actually responds to fcs/elevator-cmd-norm and compares with F16.
Bypasses actor and PID — writes raw control surface commands directly.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput
from uav_env.JSBSim.pid_controller import PIDController, F22MavEnergyPIDController

SIM_FREQ = 60
DT = 1.0 / SIM_FREQ
AGENT_STEPS = 12  # frames per "env step"
PITCH_DEG = 90.0
VEL_MIN, VEL_MAX = 102.0, 408.0

F22_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"

INIT_STATE = {
    "latitude_deg": 59.98, "longitude_deg": 120.02,
    "altitude_ft": 6000.0 / 0.3048,
    "heading_deg": 0.0, "speed_fps": 250.0 / 0.3048,
}


def _read_surface(sim, prop: str) -> float:
    try:
        return float(sim.get_property_value(prop))
    except Exception:
        return float("nan")


def _pitch_act_to_target(act):
    """Map normalised action to physical targets (same as env._parse_actions layer 3)."""
    tp = float(act[0]) * np.deg2rad(PITCH_DEG)
    th = float(act[1]) * np.pi
    tv = VEL_MIN + (float(act[2]) + 1.0) / 2.0 * (VEL_MAX - VEL_MIN)
    return tp, th, tv


def run_direct_fcs_test(model: str, elev_cmds: list, ail_cmds: list,
                        thr_cmds: list, duration_sec: float = 10.0,
                        pid_sign=None):
    """Run direct FCS sweep — write raw surface commands, no actor, no PID."""
    results = {}
    for elev in elev_cmds:
        for ail in ail_cmds:
            for thr in thr_cmds:
                label = f"elev={elev:+.0f}_ail={ail:+.0f}_thr={thr:.1f}"
                with SuppressOutput():
                    sim = AircraftSimulator(
                        uid="test", color="Red", model=model,
                        sim_freq=SIM_FREQ, num_missiles=0,
                        init_state=dict(INIT_STATE),
                        suppress_jsbsim_output=True,
                    )

                rows = []
                crashed = False
                n_frames = int(duration_sec * SIM_FREQ)

                for f in range(n_frames):
                    if not sim.is_alive:
                        crashed = True
                        break
                    # Set raw FCS commands
                    sim.set_property_value("fcs/elevator-cmd-norm", float(elev))
                    sim.set_property_value("fcs/aileron-cmd-norm", float(ail))
                    sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
                    sim.set_property_value("fcs/throttle-cmd-norm", float(thr))
                    sim.run()

                    if f % AGENT_STEPS == 0:
                        rpy = sim.get_rpy()
                        vel = sim.get_velocity()
                        alt = sim.get_geodetic()[2]
                        spd = float(np.linalg.norm(vel))
                        rows.append({
                            "frame": f,
                            "elevator_cmd": float(elev),
                            "pitch_deg": round(np.rad2deg(rpy[1]), 2),
                            "roll_deg": round(np.rad2deg(rpy[0]), 2),
                            "yaw_deg": round(np.rad2deg(rpy[2]), 2),
                            "speed_mps": round(spd, 1),
                            "altitude_m": round(alt, 1),
                            "alpha_deg": round(np.rad2deg(_read_surface(sim, "aero/alpha-rad")), 2),
                            "beta_deg": round(np.rad2deg(_read_surface(sim, "aero/beta-rad")), 2),
                            "qbar": round(_read_surface(sim, "aero/qbar-psf"), 1),
                            "elev_pos_rad": round(_read_surface(sim, "fcs/elevator-pos-rad"), 4),
                            "ail_pos_rad": round(_read_surface(sim, "fcs/left-aileron-pos-rad"), 4),
                            "thrust": round(_read_surface(sim, "propulsion/engine/thrust-lbs"), 1),
                        })

                results[label] = {
                    "crashed": crashed,
                    "rows": rows,
                    "model": model,
                    "elev_cmd": elev,
                    "ail_cmd": ail,
                    "thr_cmd": thr,
                }
                # if crashed:
                #     break  # skip further sweeps for this elev
    return results


def _f22_pid_kwargs():
    return {
        "elevator_sign": -1,
        "pitch_kp": 1.0,
        "pitch_ki": 0.0,
        "pitch_kd": 1.0,
        "roll_kp": 0.06,
        "roll_ki": 0.08,
        "roll_kd": 0.03,
        "vel_kp": 0.04,
        "vel_ki": 0.006,
        "vel_kd": 0.002,
        "throttle_min": 0.72,
        "throttle_max": 1.0,
        "low_speed_throttle_floor": 0.95,
    }


def run_pid_action_test(model: str, action: np.ndarray, duration_sec: float = 30.0,
                        label: str = "pid", pid_class=PIDController, **pid_kwargs):
    """Run PID-controlled test with a fixed high-level action."""
    with SuppressOutput():
        sim = AircraftSimulator(
            uid="test", color="Red", model=model,
            sim_freq=SIM_FREQ, num_missiles=0,
            init_state=dict(INIT_STATE),
            suppress_jsbsim_output=True,
        )
    pid = pid_class(DT, **pid_kwargs)
    target_pitch, target_heading, target_velocity = _pitch_act_to_target(action)

    rows = []
    n_frames = int(duration_sec * SIM_FREQ)
    crashed = False

    for f in range(n_frames):
        if not sim.is_alive:
            crashed = True
            break
        rpy = sim.get_rpy()
        vel = sim.get_velocity()
        spd = float(np.linalg.norm(vel))
        vn = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
        aileron, elevator, rudder, throttle = pid.compute_control(
            rpy, spd, target_pitch, target_heading, target_velocity, ned_velocity=vn)
        sim.set_property_value("fcs/aileron-cmd-norm", float(np.clip(aileron, -1, 1)))
        sim.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1, 1)))
        sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
        sim.set_property_value("fcs/throttle-cmd-norm", float(np.clip(throttle, 0, 1)))
        sim.run()

        if f % AGENT_STEPS == 0:
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            alt = sim.get_geodetic()[2]
            spd = float(np.linalg.norm(vel))
            rows.append({
                "frame": f,
                "pitch_deg": round(np.rad2deg(rpy[1]), 2),
                "roll_deg": round(np.rad2deg(rpy[0]), 2),
                "yaw_deg": round(np.rad2deg(rpy[2]), 2),
                "speed_mps": round(spd, 1),
                "altitude_m": round(alt, 1),
                "alpha_deg": round(np.rad2deg(_read_surface(sim, "aero/alpha-rad")), 2),
                "elevator_cmd": round(float(elevator), 4),
                "aileron_cmd": round(float(aileron), 4),
                "throttle_cmd": round(float(throttle), 4),
                "elev_pos_rad": round(_read_surface(sim, "fcs/elevator-pos-rad"), 4),
            })

    return {"label": label, "model": model, "crashed": crashed, "rows": rows}


def main():
    OUT = "outputs/environment_audit/f22_control_path_diag"
    os.makedirs(OUT, exist_ok=True)

    # ============================================================
    # TEST 1: Direct FCS sweep — F22 vs F16
    # ============================================================
    print("=" * 60)
    print("TEST 1: Direct FCS surface sweep (no actor, no PID)")
    print("=" * 60)

    for model in ["f22", "f16"]:
        print(f"\n--- {model.upper()} ---")
        # Minimal sweep: neutral, full up, full down
        results = run_direct_fcs_test(
            model,
            elev_cmds=[-1.0, 0.0, 1.0],
            ail_cmds=[0.0],
            thr_cmds=[0.8],
            duration_sec=8.0,
        )
        for label, r in results.items():
            rows = r["rows"]
            if rows:
                p0 = rows[0]["pitch_deg"]
                p_end = rows[-1]["pitch_deg"]
                s0 = rows[0]["speed_mps"]
                s_end = rows[-1]["speed_mps"]
                alpha = rows[-1]["alpha_deg"]
                elev_pos = rows[-1]["elev_pos_rad"]
                print(f"  {label:30s} pitch: {p0:6.1f}->{p_end:6.1f}  speed: {s0:5.0f}->{s_end:5.0f}  alpha: {alpha:6.1f}  elev_pos: {elev_pos:+.4f}  crash={r['crashed']}")

    # ============================================================
    # TEST 2: PID-controlled level flight — F22 vs F16
    # ============================================================
    print()
    print("=" * 60)
    print("TEST 2: PID-controlled level flight (zero action)")
    print("=" * 60)
    zero_action = np.array([0.0, 0.0, 0.3], dtype=np.float32)

    for model in ["f22", "f16"]:
        label = f"{model}_pid_zero"
        pid_kwargs = {}
        pid_class = PIDController
        if model == "f22":
            pid_kwargs = _f22_pid_kwargs()
            pid_class = F22MavEnergyPIDController
        r = run_pid_action_test(model, zero_action, duration_sec=20.0,
                                label=label, pid_class=pid_class, **pid_kwargs)
        rows = r["rows"]
        if rows:
            pitches = [x["pitch_deg"] for x in rows]
            speeds = [x["speed_mps"] for x in rows]
            alts = [x["altitude_m"] for x in rows]
            print(f"  {label:30s} pitch: {pitches[0]:5.1f}->{pitches[-1]:5.1f} "
                  f"(max={max(pitches):.0f}) speed: {speeds[0]:.0f}->{speeds[-1]:.0f} "
                  f"alt: {alts[0]:.0f}->{alts[-1]:.0f} crash={r['crashed']}")

    print()
    print("=" * 60)
    print("TEST 2B: F22 PID 200-frame zero/safe action stability")
    print("=" * 60)
    fixed_actions = {
        "f22_zero_action_200": np.array([0.0, 0.0, 0.3], dtype=np.float32),
        "f22_safe_action_200": np.array([0.05, 0.0, 0.4], dtype=np.float32),
    }
    for label, action in fixed_actions.items():
        r = run_pid_action_test(
            "f22",
            action,
            duration_sec=200.0 / SIM_FREQ,
            label=label,
            pid_class=F22MavEnergyPIDController,
            **_f22_pid_kwargs(),
        )
        rows = r["rows"]
        if rows:
            pitches = [x["pitch_deg"] for x in rows]
            rolls = [x["roll_deg"] for x in rows]
            speeds = [x["speed_mps"] for x in rows]
            alts = [x["altitude_m"] for x in rows]
            print(
                f"  {label:30s} pitch={pitches[0]:5.1f}->{pitches[-1]:5.1f} "
                f"max_abs_pitch={max(abs(x) for x in pitches):.1f} "
                f"max_abs_roll={max(abs(x) for x in rolls):.1f} "
                f"speed={speeds[0]:.0f}->{speeds[-1]:.0f} "
                f"min_speed={min(speeds):.0f} alt={alts[0]:.0f}->{alts[-1]:.0f} "
                f"crash={r['crashed']}"
            )

    # ============================================================
    # TEST 3: F22 direct FCS with +1 elevator for 10 sec → read actual alpha/pitch response
    # ============================================================
    print()
    print("=" * 60)
    print("TEST 3: F22 direct FCS detailed trace (elev=-1, 0, +1)")
    print("=" * 60)
    detailed = run_direct_fcs_test(
        "f22",
        elev_cmds=[-1.0, 0.0, 1.0],
        ail_cmds=[0.0],
        thr_cmds=[0.8],
        duration_sec=12.0,
    )
    csv_path = os.path.join(OUT, "f22_direct_fcs_detailed.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["label", "frame", "pitch_deg", "roll_deg", "yaw_deg",
                      "elevator_cmd", "speed_mps", "altitude_m", "alpha_deg", "beta_deg",
                      "qbar", "elev_pos_rad", "ail_pos_rad", "thrust"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, r in detailed.items():
            for row in r["rows"]:
                row["label"] = label
                w.writerow(row)
    print(f"Saved: {csv_path}")

    # Print first 6 steps for each elev command
    for label, r in detailed.items():
        print(f"\n  {label}:")
        for row in r["rows"][:4]:
            print(f"    t={row['frame']/SIM_FREQ:.1f}s pitch={row['pitch_deg']:6.1f} "
                  f"alpha={row['alpha_deg']:6.1f} spd={row['speed_mps']:5.0f} "
                  f"elev_pos={row['elev_pos_rad']:+.4f} qbar={row['qbar']:6.0f}")

    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
