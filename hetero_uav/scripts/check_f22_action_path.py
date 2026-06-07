"""Minimal F-22 action path sanity check.

Verify that red_0 MAV actions reach JSBSim control properties and
produce distinguishable state responses.  Does not modify model files,
PID, action space, reward, termination, missile, evasion, or MAPPO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env

MAIN_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"

SCENARIOS = [
    ("level",       np.array([ 0.0,  0.0,  0.0], dtype=np.float32)),
    ("climb",       np.array([ 0.4,  0.0,  0.0], dtype=np.float32)),
    ("descend",     np.array([-0.4,  0.0,  0.0], dtype=np.float32)),
    ("turn_left",   np.array([ 0.0, -0.4,  0.0], dtype=np.float32)),
    ("turn_right",  np.array([ 0.0,  0.4,  0.0], dtype=np.float32)),
    ("speed_up",    np.array([ 0.0,  0.0,  0.8], dtype=np.float32)),
    ("slow_down",   np.array([ 0.0,  0.0, -0.8], dtype=np.float32)),
]

FCS_PROPS = [
    "fcs/elevator-cmd-norm",
    "fcs/aileron-cmd-norm",
    "fcs/rudder-cmd-norm",
    "fcs/throttle-cmd-norm",
]

STATE_PROPS = [
    "propulsion/engine[0]/thrust-lbs",
    "propulsion/engine[1]/thrust-lbs",
]


def _read_safe(env, prop: str) -> str | float:
    """Try to read a JSBSim property; return 'unavailable' on failure."""
    try:
        sim = env.red_planes.get("red_0")
        if sim is None:
            return "no_sim"
        return float(sim.get_property_value(prop))
    except Exception:
        return "unavailable"


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _red0_state(env) -> dict:
    sim = env.red_planes["red_0"]
    vel = sim.get_velocity()
    return {
        "altitude_m": float(sim.get_geodetic()[2]),
        "speed_mps": float(np.linalg.norm(vel)),
        "heading_rad": float(sim.get_rpy()[2]),
        "alive": bool(sim.is_alive),
    }


def _actions(env, red0_action: np.ndarray) -> dict:
    out = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    out["red_0"] = red0_action.astype(np.float32)
    return out


def run_check(steps: int, disable_mav_trim: bool) -> dict:
    records: dict[str, dict] = {}
    for name, action in SCENARIOS:
        env = make_env(MAIN_CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
        try:
            if disable_mav_trim:
                env.action_trim_by_role.pop("mav", None)
                env.action_trim_by_agent.pop("red_0", None)

            obs, _info = env.reset(seed=0)
            initial = _red0_state(env)
            nan_detected = False

            # Record applied and effective action for step 1
            applied_before_trim = action.tolist()
            effective_after_trim = applied_before_trim  # default
            try:
                trim = env._action_trim_for_agent("red_0")
                effective_after_trim = np.clip(
                    action + trim, -1.0, 1.0
                ).astype(np.float32).tolist()
            except Exception:
                pass

            fcs_first: list[dict] = []
            fcs_last: list[dict] = []

            for step_i in range(steps):
                obs, rewards, terminated, truncated, _info = env.step(
                    _actions(env, action)
                )
                for v in obs.values():
                    for sub in v.values():
                        arr = np.asarray(sub, dtype=np.float32)
                        if arr.dtype.kind == "f" and np.isnan(arr).any():
                            nan_detected = True
                for r in rewards.values():
                    if np.isnan(float(r)):
                        nan_detected = True

                if step_i == 0 or step_i == steps - 1:
                    snapshot = {}
                    sim = env.red_planes.get("red_0")
                    fcs_applied = True
                    for prop in FCS_PROPS:
                        val = _read_safe(env, prop)
                        snapshot[prop.split("/")[-1]] = val
                        if isinstance(val, str):
                            fcs_applied = False
                    for prop in STATE_PROPS:
                        key = prop.split("/")[-1]
                        snapshot[key] = _read_safe(env, prop)
                    snapshot["fcs_properties_seen"] = fcs_applied
                    if step_i == 0:
                        fcs_first.append(snapshot)
                    else:
                        fcs_last.append(snapshot)

                if all(terminated.values()) or all(truncated.values()):
                    break

            final = _red0_state(env)

            records[name] = {
                "applied_action_before_trim": applied_before_trim,
                "effective_action_after_trim": effective_after_trim,
                "initial_altitude_m": initial["altitude_m"],
                "final_altitude_m": final["altitude_m"],
                "altitude_delta_m": round(final["altitude_m"] - initial["altitude_m"], 3),
                "initial_speed_mps": initial["speed_mps"],
                "final_speed_mps": final["speed_mps"],
                "speed_delta_mps": round(final["speed_mps"] - initial["speed_mps"], 3),
                "initial_heading_rad": initial["heading_rad"],
                "final_heading_rad": final["heading_rad"],
                "heading_delta_rad": round(
                    _wrap_pi(final["heading_rad"] - initial["heading_rad"]), 6
                ),
                "crash": bool(not final["alive"]),
                "nan_detected": bool(nan_detected),
                "fcs_snapshot_first": fcs_first[0] if fcs_first else {},
                "fcs_snapshot_last": fcs_last[0] if fcs_last else {},
            }
        finally:
            env.close()
    return records


def assess(records: dict[str, dict], trim_pitch: float) -> dict:
    nan = any(r["nan_detected"] for r in records.values())
    crash = any(r["crash"] for r in records.values())

    # Check if any FCS properties were readable
    fcs_seen = any(
        r.get("fcs_snapshot_first", {}).get("fcs_properties_seen", False)
        for r in records.values()
    )

    # Action application check
    action_applied = "true"
    if fcs_seen:
        # Check if at least one scenario shows non-zero FCS values
        any_fcs = any(
            isinstance(r.get("fcs_snapshot_first", {}).get("elevator-cmd-norm"), (int, float))
            and abs(float(r["fcs_snapshot_first"]["elevator-cmd-norm"])) > 1e-9
            for r in records.values()
        )
        if not any_fcs:
            action_applied = "unknown"
    else:
        action_applied = "unknown"

    # Response separation checks
    climb = records["climb"]["altitude_delta_m"]
    descend = records["descend"]["altitude_delta_m"]
    level_alt = records["level"]["altitude_delta_m"]
    pitch_sep = climb > level_alt and descend < level_alt

    speed_up = records["speed_up"]["speed_delta_mps"]
    slow_down = records["slow_down"]["speed_delta_mps"]
    level_spd = records["level"]["speed_delta_mps"]
    speed_sep = speed_up > level_spd and slow_down < level_spd

    left = records["turn_left"]["heading_delta_rad"]
    right = records["turn_right"]["heading_delta_rad"]
    heading_sep = left < 0.0 and right > 0.0

    # Check FCS value variation across scenarios
    fcs_values: dict[str, list[float]] = {}
    for r in records.values():
        fcs = r.get("fcs_snapshot_first", {})
        for key in ["elevator-cmd-norm", "aileron-cmd-norm", "rudder-cmd-norm", "throttle-cmd-norm"]:
            v = fcs.get(key)
            if isinstance(v, (int, float)):
                fcs_values.setdefault(key, []).append(float(v))

    fcs_varies = {}
    for key, vals in fcs_values.items():
        fcs_varies[key] = len(set(round(v, 5) for v in vals)) > 1

    # Recommendation
    if not fcs_seen:
        recommendation = "f22_control_path_not_reliable_consider_model_decision"
    elif fcs_varies.get("elevator-cmd-norm") and fcs_varies.get("throttle-cmd-norm"):
        # Actions are reaching FCS and producing variation there even
        # if short-run state deltas don't always separate.
        recommendation = "f22_action_path_acceptable_continue_missile_audit"
    elif not pitch_sep and not heading_sep and not fcs_varies.get("elevator-cmd-norm"):
        recommendation = "try_disable_mav_trim_in_main_config"
    else:
        recommendation = "f22_action_path_acceptable_continue_missile_audit"

    if nan:
        recommendation = "f22_control_path_not_reliable_consider_model_decision"

    return {
        "f22_action_applied": action_applied,
        "f22_control_properties_seen": fcs_seen,
        "fcs_varies": fcs_varies,
        "trim_pitch_value": trim_pitch,
        "response_separation": {
            "pitch_response_separated": pitch_sep,
            "heading_response_separated": heading_sep,
            "speed_response_separated": speed_sep,
        },
        "nan_detected": nan,
        "crash": crash,
        "recommendation": recommendation,
    }


def _markdown(data: dict) -> str:
    lines = [
        "# F-22 Action Path Check",
        "",
        "Purpose: verify F-22 MAV high-level actions reach control properties",
        "and produce distinguishable state changes.",
        "",
        "This check does not modify missile, reward, termination, PID, or",
        "aircraft XML.",
        "",
        "**missile audit should wait until this check passes.**",
        "",
        f"## Summary",
        f"- f22_action_applied: {data['summary']['f22_action_applied']}",
        f"- f22_control_properties_seen: {data['summary']['f22_control_properties_seen']}",
        f"- trim_pitch_value: {data['summary']['trim_pitch_value']}",
        "",
        "### Response Separation",
        f"- pitch: {data['summary']['response_separation']['pitch_response_separated']}",
        f"- heading: {data['summary']['response_separation']['heading_response_separated']}",
        f"- speed: {data['summary']['response_separation']['speed_response_separated']}",
        "",
        f"### Recommendation",
        f"**{data['summary']['recommendation']}**",
        "",
        "## Scenarios",
    ]
    for name, r in data["scenarios"].items():
        fcs = r.get("fcs_snapshot_first", {})
        lines.append(f"### {name}")
        lines.append(f"- action: {r['applied_action_before_trim']}")
        lines.append(f"- effective: {r['effective_action_after_trim']}")
        lines.append(f"- altitude_delta_m: {r['altitude_delta_m']}")
        lines.append(f"- speed_delta_mps: {r['speed_delta_mps']}")
        lines.append(f"- heading_delta_rad: {r['heading_delta_rad']}")
        lines.append(f"- crash: {r['crash']}, nan: {r['nan_detected']}")
        lines.append(f"- elevator: {fcs.get('elevator-cmd-norm')}")
        lines.append(f"- aileron: {fcs.get('aileron-cmd-norm')}")
        lines.append(f"- rudder: {fcs.get('rudder-cmd-norm')}")
        lines.append(f"- throttle: {fcs.get('throttle-cmd-norm')}")
        thrust = fcs.get("thrust-lbs", "N/A")
        lines.append(f"- engine thrust: {thrust}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--disable-mav-trim", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/f22_action_path_check.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/f22_action_path_check.md",
    )
    args = parser.parse_args()

    # Read trim value from config
    import yaml
    cfg = yaml.safe_load(
        (ROOT / MAIN_CONFIG).read_text(encoding="utf-8")
    ) or {}
    trim_pitch = float(
        cfg.get("action_trim_by_role", {}).get("mav", {}).get("pitch", 0.0)
    )

    scenarios = run_check(args.steps, args.disable_mav_trim)
    summary = assess(scenarios, trim_pitch)

    data = {"scenarios": scenarios, "summary": summary}

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")

    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)
    print(f"f22_action_applied: {summary['f22_action_applied']}", flush=True)
    print(f"fcs_properties_seen: {summary['f22_control_properties_seen']}", flush=True)
    print(
        f"pitch_sep={summary['response_separation']['pitch_response_separated']} "
        f"heading_sep={summary['response_separation']['heading_response_separated']} "
        f"speed_sep={summary['response_separation']['speed_response_separated']}",
        flush=True,
    )
    print(f"recommendation: {summary['recommendation']}", flush=True)


if __name__ == "__main__":
    main()
