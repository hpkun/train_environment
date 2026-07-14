"""Read-only response audit for the 3D high-level PID action adaptation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


AXES = {"pitch": 0, "heading": 1, "speed": 2}
PRIMARY_FIELDS = {"pitch": "pitch_rad", "heading": "yaw_rad", "speed": "speed_mps"}
SNAPSHOT_STEPS = (1, 5, 10, 25)


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _audit_action(axis: str, value: float) -> np.ndarray:
    action = np.zeros(3, dtype=np.float32)
    action[AXES[axis]] = np.float32(np.clip(value, -1.0, 1.0))
    return action


def _state(env, aid: str) -> dict[str, object]:
    sim = env._get_sim(aid)
    if sim is None:
        return {key: float("nan") for key in (
            "roll_rad", "pitch_rad", "yaw_rad", "speed_mps", "position_n_m",
            "position_e_m", "position_u_m", "altitude_m", "vertical_speed_mps",
            "overload_g")}
    position = np.asarray(sim.get_position(), dtype=np.float64)
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64)
    loads = np.asarray([
        sim.get_property_value("accelerations/n-pilot-x-norm"),
        sim.get_property_value("accelerations/n-pilot-y-norm"),
        sim.get_property_value("accelerations/n-pilot-z-norm"),
    ], dtype=np.float64)
    death_reason = str(getattr(env, "_death_reasons", {}).get(aid, ""))
    return {
        "roll_rad": float(roll), "pitch_rad": float(pitch), "yaw_rad": float(yaw),
        "speed_mps": float(np.linalg.norm(velocity)),
        "position_n_m": float(position[0]), "position_e_m": float(position[1]),
        "position_u_m": float(position[2]), "altitude_m": float(sim.get_geodetic()[2]),
        "vertical_speed_mps": float(velocity[2]), "overload_g": float(np.linalg.norm(loads)),
        "crash": int(not bool(sim.is_alive)),
        "out_of_bounds": int("out_of" in death_reason.lower() or "boundary" in death_reason.lower()),
        "death_reason": death_reason,
    }


def _difference(axis: str, value: float, neutral: float) -> float:
    if axis == "heading":
        return _wrap_pi(value - neutral)
    return value - neutral


def _summarize(rows: list[dict[str, object]], values: tuple[float, ...]) -> dict[str, object]:
    by_key = {(str(r["axis"]), float(r["action_value"]), int(r["step"])): r for r in rows}
    axes: dict[str, object] = {}
    for axis in AXES:
        primary = PRIMARY_FIELDS[axis]
        endpoint = SNAPSHOT_STEPS[-1]
        endpoint_values = [float(by_key[(axis, value, endpoint)][primary]) for value in values]
        diffs = np.diff(endpoint_values)
        monotonic = bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))
        response_delays: dict[str, int | None] = {}
        coupling: dict[str, float] = {}
        neutral_rows = {int(r["step"]): r for r in rows
                        if r["axis"] == axis and float(r["action_value"]) == 0.0}
        for value in values:
            if value == 0.0:
                continue
            key = f"{value:+.2f}"
            delay = None
            for step in SNAPSHOT_STEPS:
                row = by_key[(axis, value, step)]
                neutral = neutral_rows[step]
                threshold = 0.5 if axis == "speed" else math.radians(0.25)
                if abs(_difference(axis, float(row[primary]), float(neutral[primary]))) > threshold:
                    delay = step
                    break
            response_delays[key] = delay
            row = by_key[(axis, value, endpoint)]
            neutral = neutral_rows[endpoint]
            primary_delta = abs(_difference(axis, float(row[primary]), float(neutral[primary])))
            other_fields = [field for field in ("pitch_rad", "yaw_rad", "speed_mps") if field != primary]
            other_delta = sum(abs(_difference("heading" if field == "yaw_rad" else "linear",
                                             float(row[field]), float(neutral[field])))
                              for field in other_fields)
            coupling[key] = float(other_delta / max(primary_delta, 1e-6))
        axes[axis] = {
            "monotonic_response_at_step_25": monotonic,
            "response_delay_decision_steps": response_delays,
            "cross_axis_coupling_ratio_at_step_25": coupling,
            "saturation_rate": float(np.mean([abs(float(r["action_value"])) >= 0.999
                                               for r in rows if r["axis"] == axis])),
        }
    numeric_fields = (
        "roll_rad", "pitch_rad", "yaw_rad", "speed_mps", "position_n_m",
        "position_e_m", "position_u_m", "altitude_m", "vertical_speed_mps", "overload_g")
    return {
        "control_interface": "high_level_pid_3d",
        "action_semantics": ["target_pitch", "absolute_target_heading", "target_speed"],
        "paper_action_space_exact": False,
        "paper_action_space_reference": "multidiscrete_4d_40_bins",
        "snapshot_steps": list(SNAPSHOT_STEPS), "action_values": list(values),
        "rows": len(rows),
        "finite": bool(all(np.isfinite(float(row[field])) for row in rows for field in numeric_fields)),
        "any_crash": bool(any(int(row["crash"]) for row in rows)),
        "any_out_of_bounds": bool(any(int(row["out_of_bounds"]) for row in rows)),
        "axes": axes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--agent-id", default="red_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--values", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    parser.add_argument("--output-dir", default="outputs/audit_high_level_pid_action_response")
    args = parser.parse_args()
    values = tuple(sorted(set(float(np.clip(value, -1.0, 1.0)) for value in args.values) | {0.0}))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.config, suppress_jsbsim_output=True)
    rows: list[dict[str, object]] = []
    try:
        for axis in AXES:
            for value in values:
                env.reset(seed=args.seed)
                requested = _audit_action(axis, value)
                for step in range(1, SNAPSHOT_STEPS[-1] + 1):
                    actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
                    actions[args.agent_id] = requested.copy()
                    _obs, _rewards, terminated, truncated, info = env.step(actions)
                    if step in SNAPSHOT_STEPS:
                        row = {
                            "axis": axis, "action_value": value, "step": step,
                            "agent_id": args.agent_id,
                            "requested_action": json.dumps(requested.tolist()),
                            "executed_control_target": json.dumps(
                                info[args.agent_id].get("executed_control_target")),
                            "executed_control_mode": info[args.agent_id].get("executed_control_mode", ""),
                            "action_overridden": info[args.agent_id].get("action_overridden", False),
                            "action_override_reason": info[args.agent_id].get("action_override_reason", ""),
                            **_state(env, args.agent_id),
                        }
                        rows.append(row)
                    if terminated.get(args.agent_id, False) or truncated.get(args.agent_id, False):
                        for missing_step in SNAPSHOT_STEPS:
                            if missing_step > step:
                                rows.append({**rows[-1], "step": missing_step})
                        break
    finally:
        env.close()
    with (out / "action_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, values)
    (out / "action_response_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
