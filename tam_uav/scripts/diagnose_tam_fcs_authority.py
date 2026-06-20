"""Diagnose model-level TAM direct-FCS throttle authority with fixed actions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


def summarize_authority(scenarios):
    high = scenarios["throttle_high_neutral"]["red_0"]
    low = scenarios["throttle_low_neutral"]["red_0"]
    speed_delta = high["final_speed_mps"] - low["final_speed_mps"]
    thrust_delta = high["mean_thrust_lbs"] - low["mean_thrust_lbs"]
    return {
        "f22_final_speed_delta_mps": float(speed_delta),
        "f22_mean_thrust_delta_lbs": float(thrust_delta),
        "f22_throttle_authority_passed": bool(
            speed_delta >= 10.0 or thrust_delta >= 500.0
        ),
        "f22_high_throttle_avoids_46mps_failure": bool(
            high["min_speed_mps"] >= 100.0
        ),
        "f16_throttle_authority_passed": all(
            scenarios["throttle_high_neutral"][agent]["mean_thrust_lbs"]
            > scenarios["throttle_low_neutral"][agent]["mean_thrust_lbs"] + 500.0
            for agent in ("red_1", "red_2", "blue_0")
        ),
    }


def _disable_all_missiles(env):
    for sim in [*env.red_planes.values(), *env.blue_planes.values()]:
        sim.num_left_missiles = 0
        sim.num_missiles = 0


def _engine_values(sim):
    prefixes = ["propulsion/engine"]
    if sim.model == "f22":
        prefixes.append("propulsion/engine[1]")
    def values(suffix):
        result = []
        for prefix in prefixes:
            try:
                result.append(float(sim.jsbsim_exec.get_property_value(f"{prefix}/{suffix}")))
            except Exception:
                pass
        return result
    thrust = values("thrust-lbs")
    n1 = values("n1")
    n2 = values("n2")
    fuel = values("fuel-flow-rate-pps")
    return {
        "thrust_lbs": float(sum(thrust)),
        "n1": float(np.mean(n1)) if n1 else None,
        "n2": float(np.mean(n2)) if n2 else None,
        "fuel_flow_pps": float(sum(fuel)) if fuel else None,
    }


def _action_for(env, agent_id, throttle_idx, f22_elevator, exact_middle=False):
    if exact_middle:
        return np.asarray([throttle_idx, 20, 20, 20], dtype=np.int64)
    elevator = f22_elevator if env.agent_roles.get(agent_id) == "mav" else 4
    return np.asarray([throttle_idx, 20, elevator, 20], dtype=np.int64)


def _run_scenario(config, name, *, max_steps, throttle_idx=39,
                  f22_elevator=20, exact_middle=False, seed=0):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    env.reset(seed=seed)
    _disable_all_missiles(env)
    agent_ids = list(env.red_ids) + list(env.blue_ids)
    traces = {agent_id: [] for agent_id in agent_ids}
    for _ in range(max_steps):
        actions = {
            agent_id: _action_for(
                env, agent_id, throttle_idx, f22_elevator, exact_middle
            ) for agent_id in agent_ids
        }
        _obs, _rewards, terminated, truncated, _info = env.step(actions)
        for agent_id in agent_ids:
            sim = (env.red_planes if agent_id in env.red_planes else env.blue_planes)[agent_id]
            velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
            roll, pitch, yaw = (float(value) for value in sim.get_rpy())
            command = env._last_tam_action_commands.get(agent_id, {})
            engine = _engine_values(sim)
            traces[agent_id].append({
                "scenario": name, "step": int(env.current_step),
                "agent_id": agent_id, "model": sim.model,
                "speed_mps": float(np.linalg.norm(velocity)),
                "altitude_m": float(sim.get_geodetic()[2]),
                "pitch_rad": pitch, "roll_rad": roll, "yaw_rad": yaw,
                "vertical_speed_mps": float(velocity[2]),
                "throttle_cmd_norm": command.get("calibrated_throttle_cmd_norm"),
                "throttle_readback": command.get("readback_values", {}),
                "written_fcs_paths": command.get("written_fcs_paths", []),
                "missing_fcs_paths": command.get("missing_fcs_paths", []),
                **engine,
            })
        if not env.red_planes["red_0"].is_alive:
            break
    summaries = {}
    for agent_id, rows in traces.items():
        sim = (env.red_planes if agent_id in env.red_planes else env.blue_planes)[agent_id]
        summaries[agent_id] = {
            "model": sim.model,
            "steps": len(rows),
            "final_speed_mps": rows[-1]["speed_mps"],
            "min_speed_mps": min(row["speed_mps"] for row in rows),
            "final_altitude_m": rows[-1]["altitude_m"],
            "min_altitude_m": min(row["altitude_m"] for row in rows),
            "mean_thrust_lbs": float(np.mean([row["thrust_lbs"] for row in rows])),
            "mean_n1": float(np.mean([row["n1"] for row in rows if row["n1"] is not None])),
            "mean_n2": float(np.mean([row["n2"] for row in rows if row["n2"] is not None])),
            "mean_fuel_flow_pps": float(np.mean([
                row["fuel_flow_pps"] for row in rows if row["fuel_flow_pps"] is not None
            ])),
            "written_fcs_paths": rows[-1]["written_fcs_paths"],
            "missing_fcs_paths": rows[-1]["missing_fcs_paths"],
            "death_reason": env._death_reasons.get(agent_id) or "alive",
            "death_step": -1 if sim.is_alive else int(env.current_step),
        }
    env.close()
    return summaries, traces


def run_diagnosis(config, output_dir, max_steps=1000, seed=0):
    specs = [
        ("throttle_high_neutral", 39, 20, False),
        ("throttle_low_neutral", 0, 20, False),
        ("throttle_high_elevator_0", 39, 0, False),
        ("throttle_high_elevator_6", 39, 6, False),
        ("throttle_high_elevator_20", 39, 20, False),
        ("throttle_high_exact_middle", 39, 20, True),
        ("throttle_high_calibrated", 39, 20, False),
    ]
    scenarios, all_rows = {}, []
    for name, throttle, elevator, middle in specs:
        summary, traces = _run_scenario(
            config, name, max_steps=max_steps, throttle_idx=throttle,
            f22_elevator=elevator, exact_middle=middle, seed=seed,
        )
        scenarios[name] = summary
        all_rows.extend(row for rows in traces.values() for row in rows)
    authority = summarize_authority(scenarios)
    result = {
        "config": config, "max_steps": max_steps,
        "pre_fix_reference_f22_speed_at_60s_mps": 46.26841412186963,
        "scenarios": scenarios, "authority": authority,
        "conclusion": (
            "elevator_or_trim_dominates" if authority["f22_throttle_authority_passed"]
            else "throttle_authority_failed"
        ),
    }
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tam_fcs_authority.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    columns = [
        "scenario", "step", "agent_id", "model", "speed_mps", "altitude_m",
        "pitch_rad", "roll_rad", "yaw_rad", "vertical_speed_mps",
        "throttle_cmd_norm", "throttle_readback", "thrust_lbs", "n1", "n2",
        "fuel_flow_pps", "written_fcs_paths", "missing_fcs_paths",
    ]
    with (out_dir / "tam_fcs_authority_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            payload = dict(row)
            for field in ("throttle_readback", "written_fcs_paths", "missing_fcs_paths"):
                payload[field] = json.dumps(payload[field])
            writer.writerow(payload)
    lines = ["# TAM FCS Authority Audit", "", f"- Conclusion: **{result['conclusion']}**"]
    lines.extend(f"- {key}: `{value}`" for key, value in authority.items())
    (out_dir / "tam_fcs_authority.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_diagnosis(
        args.config, args.output_dir, max_steps=args.max_steps, seed=args.seed
    )
    print(json.dumps(result["authority"], indent=2))


if __name__ == "__main__":
    main()
