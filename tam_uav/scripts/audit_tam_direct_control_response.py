"""Run fixed-action response checks for the TAM direct-FCS interface."""
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


FIXED_ACTIONS = {
    "level": [0.65, 0.0, 0.0, 0.0],
    "throttle_high": [0.9, 0.0, 0.0, 0.0],
    "throttle_low": [0.4, 0.0, 0.0, 0.0],
    "climb_pos": [0.8, 0.0, 0.3, 0.0],
    "climb_neg": [0.8, 0.0, -0.3, 0.0],
    "roll_left": [0.75, -0.4, 0.0, 0.0],
    "roll_right": [0.75, 0.4, 0.0, 0.0],
    "rudder_left": [0.75, 0.0, 0.0, -0.4],
    "rudder_right": [0.75, 0.0, 0.0, 0.4],
}

FCS_COMMAND_PROPERTIES = {
    "throttle": "fcs/throttle-cmd-norm",
    "aileron": "fcs/aileron-cmd-norm",
    "elevator": "fcs/elevator-cmd-norm",
    "rudder": "fcs/rudder-cmd-norm",
}

FCS_SURFACE_PROPERTIES = {
    model: {
        "left_aileron": "fcs/left-aileron-pos-rad",
        "right_aileron": "fcs/right-aileron-pos-rad",
        "elevator": "fcs/elevator-pos-rad",
        "rudder": "fcs/rudder-pos-rad",
        "throttle": "fcs/throttle-pos-norm",
    }
    for model in ("f22", "f16")
}


def _property(sim, *names: str) -> float | None:
    for name in names:
        try:
            value = float(sim.get_property_value(name))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def _telemetry(env, agent_id: str, step: int, origin: np.ndarray) -> dict:
    sim = env._get_sim(agent_id)
    if sim is None:
        return {"step": step, "alive": False, "nonfinite": False}
    position = np.asarray(sim.get_position(), dtype=np.float64)
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, yaw = (float(value) for value in sim.get_rpy())
    altitude = float(sim.get_geodetic()[2])
    command = dict(env._last_tam_action_commands.get(agent_id, {}))
    model = env.agent_models[agent_id]
    surface_properties = FCS_SURFACE_PROPERTIES[model]
    command_values = {
        name: _property(sim, path) for name, path in FCS_COMMAND_PROPERTIES.items()
    }
    surface_values = {
        name: _property(sim, path) for name, path in surface_properties.items()
    }
    values = np.concatenate([position, velocity, [roll, pitch, yaw, altitude]])
    return {
        "step": step,
        "alive": bool(sim.is_alive),
        **command,
        "command_properties": {
            "paths": FCS_COMMAND_PROPERTIES,
            "values": command_values,
        },
        "surface_properties": {
            "paths": surface_properties,
            "values": surface_values,
        },
        "fcs_surface": surface_values,
        "thrust": _property(sim, "propulsion/engine[0]/thrust-lbs", "forces/fbx-prop-lbs"),
        "speed_mps": float(np.linalg.norm(velocity)),
        "altitude_m": altitude,
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "position_m": position.tolist(),
        "position_delta_m": (position - origin).tolist(),
        "nonfinite": bool(not np.isfinite(values).all()),
    }


def _run_case(config: str, target_id: str, name: str, raw_action: list[float], steps: int) -> dict:
    env = make_env(config)
    _obs, _info = env.reset(seed=0)
    sim = env._get_sim(target_id)
    origin = np.asarray(sim.get_position(), dtype=np.float64)
    records = [_telemetry(env, target_id, 0, origin)]
    last_info: dict = {}
    for step in range(1, steps + 1):
        actions = {
            aid: np.asarray(FIXED_ACTIONS["level"], dtype=np.float32)
            for aid in env.agent_ids
        }
        actions[target_id] = np.asarray(raw_action, dtype=np.float32)
        _obs, _rewards, terminated, truncated, last_info = env.step(actions)
        records.append(_telemetry(env, target_id, step, origin))
        if terminated.get(target_id, False) or truncated.get(target_id, False):
            break
    final = records[-1]
    initial = records[0]
    result = {
        "agent_id": target_id,
        "aircraft_model": env.agent_models[target_id],
        "case": name,
        "steps_completed": len(records) - 1,
        "duration_sec": (len(records) - 1) * env.env_dt,
        "raw_action": raw_action,
        "records": records,
        "delta": {
            key: float(final.get(key, 0.0) - initial.get(key, 0.0))
            for key in ("speed_mps", "altitude_m", "roll_rad", "pitch_rad", "yaw_rad")
        },
        "nonfinite": any(record.get("nonfinite", False) for record in records),
        "alive_final": bool(final.get("alive", False)),
        "crash_death_reason": (
            last_info.get(target_id, {}).get("death_reason", "")
            if isinstance(last_info.get(target_id, {}), dict) else ""
        ),
    }
    env.close()
    return result


def _checks(cases: list[dict]) -> dict:
    by_key = {(case["agent_id"], case["case"]): case for case in cases}
    checks: dict[str, bool] = {"action_dim_4": True, "no_nonfinite": True}
    for agent_id in ("red_0", "red_1"):
        prefix = f"{agent_id}_"
        high = by_key[(agent_id, "throttle_high")]["delta"]["speed_mps"]
        low = by_key[(agent_id, "throttle_low")]["delta"]["speed_mps"]
        left = by_key[(agent_id, "roll_left")]["delta"]
        right = by_key[(agent_id, "roll_right")]["delta"]
        climb_pos = by_key[(agent_id, "climb_pos")]["delta"]
        climb_neg = by_key[(agent_id, "climb_neg")]["delta"]
        rudder_left = by_key[(agent_id, "rudder_left")]
        rudder_right = by_key[(agent_id, "rudder_right")]
        roll_left_surface = by_key[(agent_id, "roll_left")]["records"][-1]["surface_properties"]["values"]
        roll_right_surface = by_key[(agent_id, "roll_right")]["records"][-1]["surface_properties"]["values"]
        climb_pos_surface = by_key[(agent_id, "climb_pos")]["records"][-1]["surface_properties"]["values"]
        climb_neg_surface = by_key[(agent_id, "climb_neg")]["records"][-1]["surface_properties"]["values"]
        rudder_left_surface = rudder_left["records"][-1]["surface_properties"]["values"]
        rudder_right_surface = rudder_right["records"][-1]["surface_properties"]["values"]
        roll_left_command = by_key[(agent_id, "roll_left")]["records"][-1]["command_properties"]["values"]
        roll_right_command = by_key[(agent_id, "roll_right")]["records"][-1]["command_properties"]["values"]
        climb_pos_command = by_key[(agent_id, "climb_pos")]["records"][-1]["command_properties"]["values"]
        climb_neg_command = by_key[(agent_id, "climb_neg")]["records"][-1]["command_properties"]["values"]
        rudder_left_command = rudder_left["records"][-1]["command_properties"]["values"]
        rudder_right_command = rudder_right["records"][-1]["command_properties"]["values"]
        checks[prefix + "throttle_high_gt_low"] = high > low
        checks[prefix + "aileron_command_distinct"] = abs(
            roll_left_command["aileron"] - roll_right_command["aileron"]
        ) > 1e-3
        checks[prefix + "elevator_command_distinct"] = abs(
            climb_pos_command["elevator"] - climb_neg_command["elevator"]
        ) > 1e-3
        checks[prefix + "rudder_command_distinct"] = abs(
            rudder_left_command["rudder"] - rudder_right_command["rudder"]
        ) > 1e-3
        checks[prefix + "aileron_surface_distinct"] = any(
            abs(roll_left_surface[name] - roll_right_surface[name]) > 1e-3
            for name in ("left_aileron", "right_aileron")
        )
        checks[prefix + "elevator_surface_distinct"] = abs(
            climb_pos_surface["elevator"] - climb_neg_surface["elevator"]
        ) > 1e-3
        checks[prefix + "rudder_surface_distinct"] = abs(
            rudder_left_surface["rudder"] - rudder_right_surface["rudder"]
        ) > 1e-3
        checks[prefix + "roll_trends_distinct"] = abs(left["roll_rad"] - right["roll_rad"]) > 0.02
        checks[prefix + "elevator_trends_distinct"] = (
            abs(climb_pos["pitch_rad"] - climb_neg["pitch_rad"]) > 0.01
            or abs(climb_pos["altitude_m"] - climb_neg["altitude_m"]) > 10.0
        )
        checks[prefix + "fcs_binding"] = all(
            checks[prefix + key]
            for key in (
                "aileron_command_distinct",
                "elevator_command_distinct",
                "rudder_command_distinct",
                "aileron_surface_distinct",
                "elevator_surface_distinct",
                "rudder_surface_distinct",
            )
        )
    checks["no_nonfinite"] = not any(case["nonfinite"] for case in cases)
    checks["no_crash_nonfinite_state"] = not any(
        "NonFinite" in str(case.get("crash_death_reason", ""))
        for case in cases
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output-dir", default="outputs/environment_audit/tam_direct_control_response")
    args = parser.parse_args()

    cases = [
        _run_case(args.config, agent_id, name, action, args.steps)
        for agent_id in ("red_0", "red_1")
        for name, action in FIXED_ACTIONS.items()
    ]
    payload = {"config": args.config, "steps": args.steps, "cases": cases, "checks": _checks(cases)}
    output_prefix = ROOT / args.output_dir
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# TAM Direct-Control Response Audit", "", "## Acceptance checks", ""]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in payload["checks"].items())
    lines.extend(["", "## Cases", ""])
    for case in cases:
        lines.append(
            f"- {case['agent_id']} ({case['aircraft_model']}) / {case['case']}: "
            f"steps={case['steps_completed']}, alive={case['alive_final']}, delta={case['delta']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
