"""Audit whether f16/f22 satisfy powered-aircraft admissibility under parent init.

Compares f16 and f22 using the same AircraftSimulator construction and
initialization path shared by the parent BRMA-MAPPO project and hetero_uav.
Does NOT modify reward, missile, PID, XML, blue rule, action space, or
observation dim.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_CONFIG_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
DEFAULT_CONFIG_5V4 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"
SIM_FREQ = 60
PHYSICS_STEPS_PER_ENV_STEP = 12
DT = 1.0 / SIM_FREQ

FIXED_ACTIONS = {
    "level":     np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "speed_up":  np.array([0.0, 0.0, 0.8], dtype=np.float32),
    "climb":     np.array([0.4, 0.0, 0.0], dtype=np.float32),
}

FIELDS = [
    "model", "config", "action_name", "env_step", "sim_time",
    "altitude_m", "speed_mps", "north_m", "east_m", "up_m",
    "roll_rad", "pitch_rad", "yaw_rad",
    "throttle_cmd_norm", "throttle_pos_norm",
    "engine_0_thrust_lbs", "engine_1_thrust_lbs", "combined_thrust_lbs",
    "engine_0_running", "engine_1_running",
    "engine_0_n1", "engine_0_n2",
    "elevator_pos_rad", "left_aileron_pos_rad",
    "right_aileron_pos_rad", "rudder_pos_rad",
    "nonfinite", "crash", "death_reason",
]


def _env_step_record(env, action_name: str, step: int,
                     model: str, config_name: str) -> dict:
    """Snapshot per-step aircraft state from env red_planes."""
    sim = env.red_planes.get("red_0")
    if sim is None:
        return {"model": model, "config": config_name,
                "action_name": action_name, "env_step": step,
                "nonfinite": True, "crash": True, "death_reason": "nonexistent"}
    try:
        geo = sim.get_geodetic()
    except Exception:
        geo = np.zeros(3)
    try:
        pos = sim.get_position()
    except Exception:
        pos = np.zeros(3)
    try:
        vel = sim.get_velocity()
    except Exception:
        vel = np.zeros(3)
    try:
        rpy = sim.get_rpy()
    except Exception:
        rpy = np.zeros(3)
    speed = float(np.linalg.norm(vel))

    tcmd = _prop(sim, "fcs/throttle-cmd-norm")
    tpos = _prop(sim, "fcs/throttle-pos-norm")
    t0 = _prop(sim, "propulsion/engine[0]/thrust-lbs")
    t1 = _prop(sim, "propulsion/engine[1]/thrust-lbs")
    n1_0 = _prop(sim, "propulsion/engine[0]/n1")
    n2_0 = _prop(sim, "propulsion/engine[0]/n2")
    running_0 = _prop(sim, "propulsion/engine[0]/set-running") if tcmd > 0 else ""
    running_1 = _prop(sim, "propulsion/engine[1]/set-running") if tcmd > 0 else ""
    elev = _prop(sim, "fcs/elevator-pos-rad")
    lail = _prop(sim, "fcs/left-aileron-pos-rad")
    rail = _prop(sim, "fcs/right-aileron-pos-rad")
    rudd = _prop(sim, "fcs/rudder-pos-rad")

    nonfinite = not (np.isfinite(pos).all() and np.isfinite(vel).all()
                     and np.isfinite(rpy).all())
    crashed = not sim.is_alive
    death = ""
    if crashed:
        death = "crash"
        alt = float(geo[2]) if np.isfinite(geo[2]) else 0.0
        if alt < 100.0:
            death = "ground_collision"

    return {
        "model": model, "config": config_name, "action_name": action_name,
        "env_step": step, "sim_time": 0.0,
        "altitude_m": float(geo[2]) if geo.size >= 3 else 0.0,
        "speed_mps": speed,
        "north_m": float(pos[0]), "east_m": float(pos[1]), "up_m": float(pos[2]),
        "roll_rad": float(rpy[0]), "pitch_rad": float(rpy[1]),
        "yaw_rad": float(rpy[2]),
        "throttle_cmd_norm": tcmd, "throttle_pos_norm": tpos,
        "engine_0_thrust_lbs": t0, "engine_1_thrust_lbs": t1,
        "combined_thrust_lbs": t0 + t1 if np.isfinite(t0) and np.isfinite(t1) else 0.0,
        "engine_0_running": running_0, "engine_1_running": running_1,
        "engine_0_n1": n1_0, "engine_0_n2": n2_0,
        "elevator_pos_rad": elev, "left_aileron_pos_rad": lail,
        "right_aileron_pos_rad": rail, "rudder_pos_rad": rudd,
        "nonfinite": nonfinite, "crash": crashed,
        "death_reason": death,
    }


def _prop(sim, path: str) -> float:
    try:
        return float(sim.get_property_value(path))
    except Exception:
        return float("nan")


def _env_diagnostic(config_path: str, model: str, steps: int) -> list[dict]:
    """Run per-action env diagnostic using the given config and MAV model."""
    from uav_env import make_env

    rows = []
    cfg_name = Path(config_path).stem
    for action_name, action in FIXED_ACTIONS.items():
        env = make_env(config_path, env_type="jsbsim_hetero",
                       suppress_jsbsim_output=False)
        obs, info = env.reset(seed=0)
        action_dict = {}
        for rid in env.agent_ids:
            action_dict[rid] = np.zeros(3, dtype=np.float32)
        action_dict["red_0"] = action.astype(np.float32)

        rows.append(_env_step_record(env, action_name, 0, model, cfg_name))
        for s in range(1, steps + 1):
            obs, _rewards, _terminated, _truncated, _info = env.step(action_dict)
            rows.append(_env_step_record(env, action_name, s, model, cfg_name))
            sim = env.red_planes.get("red_0")
            if sim is not None and not sim.is_alive:
                # record final frame then stop
                break
        env.close()
    return rows


def _compute_summary(rows: list[dict]) -> dict:
    """Compute admissibility metrics from diagnostic rows."""
    models = sorted(set(r["model"] for r in rows))
    summary: dict = {}
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        nonfinite = sum(1 for r in model_rows if r["nonfinite"])
        crashes = sum(1 for r in model_rows if r["crash"])

        level_rows = [r for r in model_rows if r["action_name"] == "level"]
        speed_rows = [r for r in model_rows if r["action_name"] == "speed_up"]
        climb_rows = [r for r in model_rows if r["action_name"] == "climb"]

        def _speed_at(s_rows, sec: float):
            env_step = int(sec * 5.0)
            candidates = [r for r in s_rows if r["env_step"] >= env_step
                          and not r["crash"]]
            if not candidates:
                return float("nan")
            return candidates[0]["speed_mps"]

        def _thrust_at(s_rows, sec: float):
            env_step = int(sec * 5.0)
            candidates = [r for r in s_rows if r["env_step"] >= env_step
                          and not r["crash"]]
            if not candidates:
                return float("nan")
            return candidates[0]["combined_thrust_lbs"]

        sr60_level = _speed_at(level_rows, 60)
        sr60_speed = _speed_at(speed_rows, 60)
        sr120_level = _speed_at(level_rows, 120)
        sr120_speed = _speed_at(speed_rows, 120)

        thr_level_30 = _thrust_at(level_rows, 30)
        thr_speed_30 = _thrust_at(speed_rows, 30)

        level_final = level_rows[-1] if level_rows else {}
        speed_final = speed_rows[-1] if speed_rows else {}
        climb_final = climb_rows[-1] if climb_rows else {}

        # State differentiation: did climb produce different altitude vs level?
        alt_level_60 = level_final.get("altitude_m", 0.0) if not level_final.get("crash") else float("nan")
        alt_climb_60 = climb_final.get("altitude_m", 0.0) if not climb_final.get("crash") else float("nan")
        alt_sep = abs(alt_climb_60 - alt_level_60) if np.isfinite(alt_climb_60) and np.isfinite(alt_level_60) else 0.0

        admissible = (
            np.isfinite(sr60_level) and sr60_level >= 150.0
            and np.isfinite(thr_level_30) and np.isfinite(thr_speed_30)
            and thr_speed_30 >= thr_level_30 * 1.1  # 10% thrust increase
            and alt_sep >= 10.0  # climb produces 10m+ altitude difference
            and nonfinite == 0
        )

        summary[model] = {
            "model": model,
            "speed_retention_60s_level_mps": float(sr60_level),
            "speed_retention_60s_speed_up_mps": float(sr60_speed),
            "speed_retention_120s_level_mps": float(sr120_level),
            "speed_retention_120s_speed_up_mps": float(sr120_speed),
            "combined_thrust_lbs_level_30s": float(thr_level_30),
            "combined_thrust_lbs_speed_up_30s": float(thr_speed_30),
            "thrust_response_ratio": float(thr_speed_30 / max(thr_level_30, 1.0)),
            "altitude_separation_climb_vs_level_m": float(alt_sep),
            "nonfinite_rows": nonfinite,
            "crash_rows": crashes,
            "admissible": bool(admissible),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output-csv", default="outputs/environment_audit/aircraft_model_admissibility.csv")
    parser.add_argument("--output-json", default="outputs/environment_audit/aircraft_model_admissibility_summary.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/aircraft_model_admissibility.md")
    parser.add_argument("--models", nargs="*", default=["f16", "f22"])
    args = parser.parse_args()

    # Config ↔ model mapping from official configs
    CONFIG_MODEL_MAP = [
        ("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml", "f16", "3v2"),
        ("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml", "f22", "3v2"),
    ]

    out_csv = ROOT / args.output_csv
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for cfg, model, scenario in CONFIG_MODEL_MAP:
        if model not in args.models:
            continue
        rows = _env_diagnostic(cfg, model, args.steps)
        all_rows.extend(rows)

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # Compute summary
    summary = _compute_summary(all_rows)

    # Write JSON
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Write MD
    md_lines = ["# Aircraft Model Admissibility Audit", ""]
    md_lines.append("| Model | Speed@60s | Thrust@30s | Thrust Resp | Alt Sep | Nonfinite | Crash | Admissible |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for model in args.models:
        s = summary.get(model, {})
        if not s:
            continue
        md_lines.append(
            f"| {model} | {s['speed_retention_60s_level_mps']:.0f} m/s | "
            f"{s['combined_thrust_lbs_level_30s']:.0f} lbs | "
            f"{s['thrust_response_ratio']:.1f}x | "
            f"{s['altitude_separation_climb_vs_level_m']:.0f} m | "
            f"{s['nonfinite_rows']} | {s['crash_rows']} | "
            f"{'✅' if s['admissible'] else '❌'} |"
        )
    md_lines.append("")
    md_lines.append("## Conclusion")
    for model in args.models:
        s = summary.get(model, {})
        if not s:
            continue
        if s["admissible"]:
            md_lines.append(f"- **{model}**: ✅ ADMISSIBLE under parent init")
        else:
            md_lines.append(f"- **{model}**: ❌ NOT ADMISSIBLE under parent init")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Print summary
    for model in args.models:
        s = summary.get(model, {})
        if not s:
            continue
        print(f"{model}: speed_retention={s['speed_retention_60s_level_mps']:.0f} m/s "
              f"thrust_30s={s['combined_thrust_lbs_level_30s']:.0f} lbs "
              f"thrust_resp={s['thrust_response_ratio']:.1f}x "
              f"alt_sep={s['altitude_separation_climb_vs_level_m']:.0f}m "
              f"admissible={s['admissible']}")


if __name__ == "__main__":
    main()
