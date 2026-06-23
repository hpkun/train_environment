"""Audit F22 trimmed PID stability with fixed actions, blue_zero opponent."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput
from uav_env.JSBSim.pid_controller import TrimmedPIDController

SIM_FREQ = 60
DT = 1.0 / SIM_FREQ
AG = 12

INIT_STATE = {
    "latitude_deg": 59.98, "longitude_deg": 120.02,
    "altitude_ft": 6000.0 / 0.3048,
    "heading_deg": 0.0, "speed_fps": 250.0 / 0.3048,
}


def _read(sim, prop, default=float("nan")):
    try: return float(sim.get_property_value(prop))
    except Exception: return default


def _act_to_targets(act):
    tp = float(act[0]) * np.deg2rad(90.0)
    th = float(act[1]) * np.pi
    tv = 102.0 + (float(act[2]) + 1.0) / 2.0 * (408.0 - 102.0)
    return tp, th, tv


def run_case(case_name, action, n_env_steps, pid_kwargs):
    with SuppressOutput():
        sim = AircraftSimulator(
            uid="test", color="Red", model="f22", sim_freq=SIM_FREQ,
            num_missiles=0, init_state=dict(INIT_STATE), suppress_jsbsim_output=True)
    pid = TrimmedPIDController(DT, **pid_kwargs)
    tp, th, tv = _act_to_targets(action)

    pitches, rolls, speeds, alts = [], [], [], []
    thr_cmds, elev_cmds, elev_pos = [], [], []
    qbars = []
    death_reason = ""

    for step in range(n_env_steps):
        if not sim.is_alive:
            death_reason = "crash_or_shotdown"
            break
        for _ in range(AG):
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            spd = float(np.linalg.norm(vel))
            vn = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            ail, elev, rud, thr = pid.compute_control(rpy, spd, tp, th, tv, ned_velocity=vn)
            sim.set_property_value("fcs/aileron-cmd-norm", float(np.clip(ail, -1, 1)))
            sim.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elev, -1, 1)))
            sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
            sim.set_property_value("fcs/throttle-cmd-norm", float(np.clip(thr, 0, 1)))
            sim.run()
        rpy = sim.get_rpy(); vel = sim.get_velocity()
        pitches.append(np.rad2deg(rpy[1]))
        rolls.append(np.rad2deg(rpy[0]))
        speeds.append(float(np.linalg.norm(vel)))
        alts.append(sim.get_geodetic()[2])
        thr_cmds.append(float(np.clip(thr, 0, 1)))
        elev_cmds.append(float(np.clip(elev, -1, 1)))
        elev_pos.append(_read(sim, "fcs/elevator-pos-rad"))
        qbars.append(_read(sim, "aero/qbar-psf"))

    return {
        "case": case_name,
        "alive": bool(sim.is_alive), "death_reason": death_reason,
        "max_abs_pitch_deg": float(round(max(abs(p) for p in pitches), 1)) if pitches else 0.0,
        "max_abs_roll_deg": float(round(max(abs(r) for r in rolls), 1)) if rolls else 0.0,
        "min_speed_mps": float(round(min(speeds), 1)) if speeds else 0.0,
        "mean_speed_mps": float(round(float(np.mean(speeds)), 1)) if speeds else 0.0,
        "min_altitude_m": float(round(min(alts), 0)) if alts else 0.0,
        "final_altitude_m": float(round(alts[-1], 0)) if alts else 0.0,
        "throttle_cmd_mean": float(round(float(np.mean(thr_cmds)), 3)),
        "elevator_cmd_mean": float(round(float(np.mean(elev_cmds)), 3)),
        "elevator_pos_rad_mean": float(round(float(np.mean(elev_pos)), 4)),
        "qbar_min": float(round(min(qbars), 1)) if qbars else 0.0,
    }


def main():
    OUT = "outputs/environment_audit/f22_trimmed_pid_stability"
    os.makedirs(OUT, exist_ok=True)

    # PID params from config
    pid_kwargs = {
        "elevator_sign": -1, "throttle_trim": 0.85,
        "throttle_min": 0.0, "throttle_max": 1.0,
        "elevator_trim": 0.0, "aileron_trim": 0.0, "pitch_trim_deg": 0.0,
        "roll_kp": 0.06, "roll_ki": 0.08, "roll_kd": 0.03,
        "pitch_kp": 1.0, "pitch_ki": 0.0, "pitch_kd": 1.0,
        "vel_kp": 0.04, "vel_ki": 0.006, "vel_kd": 0.002,
    }

    actions = [
        ("action_0_0_0",       np.array([0.0,  0.0,  0.0])),
        ("action_0_0_0_3",     np.array([0.0,  0.0,  0.3])),
        ("action_0_05_0_0_3",  np.array([0.05, 0.0,  0.3])),
        ("action_m0_05_0_0_3", np.array([-0.05, 0.0,  0.3])),
        ("action_0_0_2_0_3",   np.array([0.0,  0.2,  0.3])),
    ]

    results = []
    for name, act in actions:
        print(f"=== {name} ===", flush=True)
        r = run_case(name, act, 1000, pid_kwargs)
        results.append(r)
        passed = (r["alive"] and r["min_speed_mps"] > 180
                  and r["max_abs_pitch_deg"] < 45 and r["max_abs_roll_deg"] < 80
                  and r["min_altitude_m"] > 4000)
        r["pass"] = bool(passed)
        print(f"  alive={r['alive']} spd_min={r['min_speed_mps']} pitch_max={r['max_abs_pitch_deg']} "
              f"roll_max={r['max_abs_roll_deg']} alt_min={r['min_altitude_m']} -> {'PASS' if passed else 'FAIL'}")

    # Convert numpy types for JSON serialisation
    clean_results = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.bool_): cr[k] = bool(v)
            else: cr[k] = v
        clean_results.append(cr)
    json.dump({"results": clean_results, "pid_kwargs": pid_kwargs},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # Report
    lines = ["# F22 Trimmed PID Stability Audit", ""]
    lines.append("## PID Parameters")
    for k, v in pid_kwargs.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("| case | alive | min_spd | max_pitch | max_roll | min_alt | final_alt | thr_mean | elev_mean | elev_pos | qbar_min | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['case']} | {r['alive']} | {r['min_speed_mps']} | {r['max_abs_pitch_deg']} | {r['max_abs_roll_deg']} | {r['min_altitude_m']} | {r['final_altitude_m']} | {r['throttle_cmd_mean']} | {r['elevator_cmd_mean']} | {r['elevator_pos_rad_mean']} | {r['qbar_min']} | {'PASS' if r['pass'] else 'FAIL'} |")

    all_pass = all(r["pass"] for r in results)
    lines.append(f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**")
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Trace CSV
    csv_path = os.path.join(OUT, "trace.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case","alive","death_reason","max_abs_pitch_deg",
            "max_abs_roll_deg","min_speed_mps","mean_speed_mps","min_altitude_m",
            "final_altitude_m","throttle_cmd_mean","elevator_cmd_mean",
            "elevator_pos_rad_mean","qbar_min","pass"])
        w.writeheader(); w.writerows(results)

    print(f"\nSaved: {OUT}")
    for r in results:
        print(f"  {r['case']}: {'PASS' if r['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
