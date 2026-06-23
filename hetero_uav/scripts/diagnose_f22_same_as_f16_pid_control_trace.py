"""F22 control-chain trace: action -> PID -> FCS cmd -> surface -> thrust -> aero -> state.

Tests direct FCS and PID modes to determine where the F22 control chain breaks.
"""
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
from uav_env.JSBSim.pid_controller import PIDController, F22MavEnergyPIDController

SIM_FREQ = 60
DT = 1.0 / SIM_FREQ
AG = 12
PITCH_DEG = 90.0
VEL_MIN, VEL_MAX = 102.0, 408.0

INIT_STATE = {
    "latitude_deg": 59.98, "longitude_deg": 120.02,
    "altitude_ft": 6000.0 / 0.3048,
    "heading_deg": 0.0, "speed_fps": 250.0 / 0.3048,
}

FIELDNAMES = [
    "case", "frame", "sim_time_sec",
    "action_pitch", "action_heading", "action_speed",
    "target_pitch_deg", "target_heading_deg", "target_velocity_mps",
    "pid_aileron_cmd", "pid_elevator_cmd", "pid_rudder_cmd", "pid_throttle_cmd",
    "fcs_aileron_cmd_norm", "fcs_elevator_cmd_norm", "fcs_rudder_cmd_norm",
    "fcs_throttle_cmd_norm",
    "fcs_elevator_pos_rad", "fcs_left_aileron_pos_rad", "fcs_right_aileron_pos_rad",
    "fcs_throttle_pos_norm_0", "fcs_throttle_pos_norm_1",
    "thrust_lbs_0", "thrust_lbs_1",
    "aero_alpha_rad", "aero_beta_rad", "aero_qbar_psf",
    "pitch_deg", "roll_deg", "yaw_deg",
    "altitude_m", "speed_mps", "vertical_speed_mps",
    "alive",
]


def _read(sim, prop, default=float("nan")):
    try:
        return float(sim.get_property_value(prop))
    except Exception:
        return default


def _act_to_targets(act):
    tp = float(act[0]) * np.deg2rad(PITCH_DEG)
    th = float(act[1]) * np.pi
    tv = VEL_MIN + (float(act[2]) + 1.0) / 2.0 * (VEL_MAX - VEL_MIN)
    return tp, th, tv


def run_case(case_name, duration_sec, model="f22",
             direct_elev=None, direct_ail=0.0, direct_thr=0.8,
             pid_action=None, pid_class=None, pid_kwargs=None):
    """Run one test case, return list of row dicts."""
    rows = []
    with SuppressOutput():
        sim = AircraftSimulator(
            uid="test", color="Red", model=model, sim_freq=SIM_FREQ,
            num_missiles=0, init_state=dict(INIT_STATE),
            suppress_jsbsim_output=True,
        )

    pid = None
    if pid_action is not None:
        pc = pid_class or PIDController
        pk = dict(pid_kwargs or {})
        pid = pc(DT, **pk)
        tp, th, tv = _act_to_targets(pid_action)

    n_frames = int(duration_sec * SIM_FREQ)
    for f in range(n_frames):
        if not sim.is_alive:
            break
        ail_cmd = elev_cmd = rud_cmd = thr_cmd = 0.0
        act_p = act_h = act_s = 0.0

        if direct_elev is not None:
            # Direct FCS mode — bypass PID
            sim.set_property_value("fcs/elevator-cmd-norm", float(direct_elev))
            sim.set_property_value("fcs/aileron-cmd-norm", float(direct_ail))
            sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
            sim.set_property_value("fcs/throttle-cmd-norm", float(direct_thr))
            elev_cmd = float(direct_elev)
            ail_cmd = float(direct_ail)
            thr_cmd = float(direct_thr)
            tgt_pitch = tgt_head = 0.0
            tgt_vel = 0.0
        elif pid is not None:
            # PID mode
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            spd = float(np.linalg.norm(vel))
            vn = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            ail_cmd, elev_cmd, rud_cmd, thr_cmd = pid.compute_control(
                rpy, spd, tp, th, tv, ned_velocity=vn)
            sim.set_property_value("fcs/aileron-cmd-norm", float(np.clip(ail_cmd, -1, 1)))
            sim.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elev_cmd, -1, 1)))
            sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
            sim.set_property_value("fcs/throttle-cmd-norm", float(np.clip(thr_cmd, 0, 1)))
            act_p = pid_action[0]
            act_h = pid_action[1]
            act_s = pid_action[2]
            tgt_pitch = np.rad2deg(tp)
            tgt_head = np.rad2deg(th)
            tgt_vel = tv

        sim.run()

        if f % AG == 0:
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            spd = float(np.linalg.norm(vel))
            rows.append({
                "case": case_name, "frame": f,
                "sim_time_sec": round(f / SIM_FREQ, 2),
                "action_pitch": round(float(act_p), 6),
                "action_heading": round(float(act_h), 6),
                "action_speed": round(float(act_s), 6),
                "target_pitch_deg": round(float(tgt_pitch), 2),
                "target_heading_deg": round(float(tgt_head), 2),
                "target_velocity_mps": round(float(tgt_vel), 1),
                "pid_aileron_cmd": round(float(ail_cmd), 6),
                "pid_elevator_cmd": round(float(elev_cmd), 6),
                "pid_rudder_cmd": round(float(rud_cmd), 6),
                "pid_throttle_cmd": round(float(thr_cmd), 6),
                "fcs_aileron_cmd_norm": round(_read(sim, "fcs/aileron-cmd-norm"), 6),
                "fcs_elevator_cmd_norm": round(_read(sim, "fcs/elevator-cmd-norm"), 6),
                "fcs_rudder_cmd_norm": round(_read(sim, "fcs/rudder-cmd-norm"), 6),
                "fcs_throttle_cmd_norm": round(_read(sim, "fcs/throttle-cmd-norm"), 6),
                "fcs_elevator_pos_rad": round(_read(sim, "fcs/elevator-pos-rad"), 6),
                "fcs_left_aileron_pos_rad": round(_read(sim, "fcs/left-aileron-pos-rad"), 6),
                "fcs_right_aileron_pos_rad": round(_read(sim, "fcs/right-aileron-pos-rad"), 6),
                "fcs_throttle_pos_norm_0": round(_read(sim, "fcs/throttle-pos-norm"), 6),
                "fcs_throttle_pos_norm_1": round(_read(sim, "fcs/throttle-pos-norm[1]", 0), 6),
                "thrust_lbs_0": round(_read(sim, "propulsion/engine/thrust-lbs"), 1),
                "thrust_lbs_1": round(_read(sim, "propulsion/engine[1]/thrust-lbs", 0), 1),
                "aero_alpha_rad": round(_read(sim, "aero/alpha-rad"), 4),
                "aero_beta_rad": round(_read(sim, "aero/beta-rad"), 4),
                "aero_qbar_psf": round(_read(sim, "aero/qbar-psf"), 1),
                "pitch_deg": round(float(np.rad2deg(rpy[1])), 2),
                "roll_deg": round(float(np.rad2deg(rpy[0])), 2),
                "yaw_deg": round(float(np.rad2deg(rpy[2])), 2),
                "altitude_m": round(sim.get_geodetic()[2], 1),
                "speed_mps": round(spd, 1),
                "vertical_speed_mps": round(float(vel[2]), 2),
                "alive": int(sim.is_alive),
            })
    return rows


def _generate_report(all_rows, out_dir):
    lines = ["# F22 same-as-F16 PID Control Trace Report", ""]

    for case_name in sorted(set(r["case"] for r in all_rows)):
        rows = [r for r in all_rows if r["case"] == case_name]
        first = rows[0]
        last = rows[-1]
        pitch_end = last["pitch_deg"]
        spd_end = last["speed_mps"]
        elev_cmd_nonzero = any(abs(r["pid_elevator_cmd"]) > 0.01 or abs(r["fcs_elevator_cmd_norm"]) > 0.01 for r in rows)
        elev_pos_responding = any(abs(r["fcs_elevator_pos_rad"]) > 0.0001 for r in rows)
        elev_cmd_sign = "positive" if any(r["pid_elevator_cmd"] > 0.01 or r["fcs_elevator_cmd_norm"] > 0.01 for r in rows) else "negative/zero"
        thrust_engine0 = any(r["thrust_lbs_0"] > 100 for r in rows)
        thrust_engine1 = any(r["thrust_lbs_1"] > 100 for r in rows)
        qbar_end = last["aero_qbar_psf"]
        alpha_end = last["aero_alpha_rad"]

        lines.append(f"## {case_name}")
        lines.append(f"- elevator cmd non-zero: {elev_cmd_nonzero}")
        lines.append(f"- elevator-pos-rad responds: {elev_pos_responding}")
        lines.append(f"- elevator cmd dominant sign: {elev_cmd_sign}")
        lines.append(f"- throttle → engine0 thrust: {thrust_engine0}")
        lines.append(f"- throttle → engine1 thrust: {thrust_engine1}")
        lines.append(f"- final pitch: {pitch_end:.1f} deg")
        lines.append(f"- final speed: {spd_end:.1f} m/s")
        lines.append(f"- final qbar: {qbar_end:.1f} psf")
        lines.append(f"- final alpha: {np.rad2deg(alpha_end):.1f} deg")
        lines.append(f"- aircraft alive: {bool(last['alive'])}")

    # Root cause verdict
    direct_neg = [r for r in all_rows if "direct_fcs_elev_minus1" in r["case"]]
    direct_pos = [r for r in all_rows if "direct_fcs_elev_plus1" in r["case"]]
    pid_0_0_3 = [r for r in all_rows if "pid_f16_default_action_0_0_0_3" == r["case"]]
    pid_plus_pitch = [r for r in all_rows if "action_plus0_1" in r["case"]]
    pid_minus_pitch = [r for r in all_rows if "action_minus0_1" in r["case"]]

    lines.append("")
    lines.append("## Root cause verdict")
    if direct_neg and direct_pos:
        dn_end = direct_neg[-1]["pitch_deg"]
        dp_end = direct_pos[-1]["pitch_deg"]
        lines.append(f"- direct FCS elev=-1 → pitch {dn_end:.1f} deg")
        lines.append(f"- direct FCS elev=+1 → pitch {dp_end:.1f} deg")
        if abs(dn_end - dp_end) < 5:
            lines.append("- **elevator has NO effect on pitch** — surface or FBW channel broken")
        elif dp_end < dn_end:
            lines.append("- **positive elevator → pitch DOWN** — sign convention normal")
        else:
            lines.append("- **positive elevator → pitch UP** — sign convention reversed")

    if pid_0_0_3:
        r = pid_0_0_3[-1]
        lines.append(f"- PID [0,0,0.3] → final pitch {r['pitch_deg']:.1f}, speed {r['speed_mps']:.1f}")
        pid_elev_cmds = [x["pid_elevator_cmd"] for x in pid_0_0_3]
        lines.append(f"- PID elevator range: [{min(pid_elev_cmds):.3f}, {max(pid_elev_cmds):.3f}]")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {os.path.join(out_dir, 'report.md')}")


def main():
    OUT = "outputs/environment_audit/f22_same_as_f16_pid_control_trace"
    os.makedirs(OUT, exist_ok=True)

    all_rows = []

    # ---- Direct FCS cases ----
    for elev in [-1.0, 0.0, 1.0]:
        name = f"direct_fcs_elev_{int(elev):+d}_thr_0_8"
        print(f"=== {name} ===", flush=True)
        rows = run_case(name, 20.0, direct_elev=elev)
        all_rows.extend(rows)

    # ---- PID with various actions ----
    pid_cases = [
        ("pid_f16_default_action_0_0_0_3", np.array([0.0, 0.0, 0.3])),
        ("pid_f16_default_action_0_0_1_0", np.array([0.0, 0.0, 1.0])),
        ("pid_f16_default_action_minus0_1_0_0_3", np.array([-0.1, 0.0, 0.3])),
        ("pid_f16_default_action_plus0_1_0_0_3", np.array([0.1, 0.0, 0.3])),
    ]
    for name, act in pid_cases:
        print(f"=== {name} ===", flush=True)
        rows = run_case(name, 20.0, pid_action=act)
        all_rows.extend(rows)

    # ---- F22 energy PID comparison ----
    print(f"=== pid_f22_energy_action_0_0_0_3 ===", flush=True)
    rows = run_case("pid_f22_energy_action_0_0_0_3", 20.0,
                    pid_action=np.array([0.0, 0.0, 0.3]),
                    pid_class=F22MavEnergyPIDController,
                    pid_kwargs={"elevator_sign": -1})
    all_rows.extend(rows)

    # Write CSV
    csv_path = os.path.join(OUT, "trace.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved: {csv_path} ({len(all_rows)} rows)")

    # Summary JSON
    summary = {
        "init_state": INIT_STATE,
        "sim_freq": SIM_FREQ,
        "agent_interaction_steps": AG,
        "cases": sorted(set(r["case"] for r in all_rows)),
        "total_rows": len(all_rows),
    }
    json_path = os.path.join(OUT, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    _generate_report(all_rows, OUT)


if __name__ == "__main__":
    main()
