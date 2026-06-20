"""Audit TAM airborne reset and early-flight behavior without changing the environment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from scripts.diagnose_tam_mav_policy_drift import _run_episode
from scripts.validate_tam_categorical_initial_policy_flight import _disable_blue_missiles
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


def summarize_reset_reports(reports):
    failed = sorted(
        agent_id for agent_id, report in reports.items()
        if not bool(report.get("passed_reset_contract"))
    )
    return {
        "aircraft_count": len(reports),
        "passed_reset_contract": not failed and bool(reports),
        "failed_aircraft": failed,
        "reports": reports,
    }


def classify_flight_outcome(alive, death_reason, finite):
    if not finite:
        return "nonfinite"
    if alive:
        return "alive"
    reason = str(death_reason or "missing_death_reason")
    if reason == "Crash_LowAlt":
        return "long_horizon_trim_failure"
    if reason == "Missile_Kill":
        return "missile_kill"
    return f"policy_or_flight_failure:{reason}"


def classify_audit_failure(reset_ok, authority_ok, fixed_ok, formal_ok):
    if not reset_ok:
        return "reset_contract_failed"
    if not authority_ok:
        return "throttle_authority_failed"
    if not fixed_ok:
        return "static_trim_failed"
    if not formal_ok:
        return "policy_drift_or_missile_kill"
    return "passed"


def _aircraft_row(env, agent_id, sim):
    altitude = float(sim.get_geodetic()[2])
    roll, pitch, yaw = (float(value) for value in sim.get_rpy())
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    command = env._last_tam_action_commands.get(agent_id, {})
    return {
        "step": int(env.current_step),
        "time_sec": float(env.current_step * env.agent_interaction_steps / env.sim_freq),
        "agent_id": agent_id,
        "model": sim.model,
        "speed_mps": float(np.linalg.norm(velocity)),
        "altitude_m": altitude,
        "pitch_rad": pitch,
        "roll_rad": roll,
        "yaw_rad": yaw,
        "vertical_speed_mps": float(velocity[2]),
        "alive": bool(sim.is_alive),
        "written_fcs_paths": command.get("written_fcs_paths", []),
        "readback_values": command.get("readback_values", {}),
        "calibrated_throttle_cmd_norm": command.get(
            "calibrated_throttle_cmd_norm"
        ),
    }


def _fixed_action(env, agent_id):
    role = env.agent_roles.get(agent_id)
    elevator = 20 if role == "mav" else 4
    return np.asarray([39, 20, elevator, 20], dtype=np.int64)


def _run_fixed_neutral(config, duration_seconds=120, seed=0):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=1000)
    env.reset(seed=seed)
    _disable_blue_missiles(env)
    all_agents = list(env.red_ids) + list(env.blue_ids)
    max_steps = int(round(duration_seconds * env.sim_freq / env.agent_interaction_steps))
    traces = {agent_id: [] for agent_id in all_agents}
    finite = True
    for _ in range(max_steps):
        actions = {agent_id: _fixed_action(env, agent_id) for agent_id in all_agents}
        _obs, _reward, terminated, truncated, _info = env.step(actions)
        for agent_id in all_agents:
            sim = (env.red_planes if agent_id in env.red_planes else env.blue_planes)[agent_id]
            row = _aircraft_row(env, agent_id, sim)
            finite = finite and all(
                np.isfinite(row[key]) for key in (
                    "speed_mps", "altitude_m", "pitch_rad", "roll_rad",
                    "vertical_speed_mps",
                )
            )
            traces[agent_id].append(row)
        if not env.red_planes["red_0"].is_alive:
            break
    summaries = {}
    for agent_id, trace in traces.items():
        sim = (env.red_planes if agent_id in env.red_planes else env.blue_planes)[agent_id]
        death_reason = env._death_reasons.get(agent_id) or ("alive" if sim.is_alive else "missing_death_reason")
        speed_60 = next(
            (row["speed_mps"] for row in trace if row["time_sec"] >= 60.0),
            trace[-1]["speed_mps"] if trace else float("nan"),
        )
        speed_120 = next(
            (row["speed_mps"] for row in trace if row["time_sec"] >= 120.0),
            trace[-1]["speed_mps"] if trace else float("nan"),
        )
        summaries[agent_id] = {
            "model": sim.model,
            "neutral_action": _fixed_action(env, agent_id).tolist(),
            "steps": len(trace),
            "speed_at_60s_mps": float(speed_60),
            "speed_at_120s_mps": float(speed_120),
            "min_speed_mps": float(min(row["speed_mps"] for row in trace)),
            "final_speed_mps": float(trace[-1]["speed_mps"]),
            "final_altitude_m": float(trace[-1]["altitude_m"]),
            "min_altitude_m": float(min(row["altitude_m"] for row in trace)),
            "written_fcs_paths": trace[-1]["written_fcs_paths"],
            "readback_values": trace[-1]["readback_values"],
            "death_reason": death_reason,
            "outcome_classification": classify_flight_outcome(
                sim.is_alive, death_reason, finite
            ),
            "trace": trace,
        }
    env.close()
    return {
        "duration_seconds_requested": duration_seconds,
        "blue_missiles_disabled": True,
        "finite": bool(finite),
        "f22_speed_at_60s_passed": bool(
            summaries["red_0"]["speed_at_60s_mps"] >= 150.0
        ),
        "aircraft": summaries,
    }


def run_audit(config, output_dir, seed=0):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=1000)
    env.reset(seed=seed)
    planes = {**env.red_planes, **env.blue_planes}
    reset_reports = {
        agent_id: dict(sim._initial_stabilization_report)
        for agent_id, sim in planes.items()
    }
    reset = summarize_reset_reports(reset_reports)
    env.close()

    fixed = _run_fixed_neutral(config, duration_seconds=200, seed=seed)
    policy = TAMCategoricalRecurrentHAPPOPolicy().eval()
    adapter = HeteroObsAdapterV2()
    formal_episodes = [
        _run_episode(
            config, policy, adapter, deterministic=True,
            no_blue_missile=False, max_steps=1000, seed=seed + episode,
            device="cpu",
        )
        for episode in range(3)
    ]
    formal = []
    for episode in formal_episodes:
        first_5s = episode["trace"][:25]
        min_initial_speed = min(
            (row["speed_mps"] for row in first_5s), default=float("nan")
        )
        finite = all(
            np.isfinite(row["speed_mps"]) and np.isfinite(row["altitude_m"])
            for row in episode["trace"]
        )
        formal.append({
            "seed": episode["seed"],
            "death_time_sec": (
                -1.0 if episode["death_step"] < 0 else episode["death_step"] / 5.0
            ),
            "death_step": episode["death_step"],
            "death_reason": episode["death_reason"],
            "outcome_classification": classify_flight_outcome(
                episode["survived"], episode["death_reason"], finite
            ),
            "min_speed_first_5s_mps": float(min_initial_speed),
            "reset_low_speed_cold_start": bool(min_initial_speed < 150.0),
        })
    authority_path = Path(output_dir)
    if not authority_path.is_absolute():
        authority_path = ROOT / authority_path
    authority_path = authority_path / "tam_fcs_authority.json"
    authority_ok = False
    if authority_path.exists():
        authority_ok = bool(json.loads(
            authority_path.read_text(encoding="utf-8")
        ).get("authority", {}).get("f22_throttle_authority_passed"))
    fixed_mav = fixed["aircraft"]["red_0"]
    fixed_ok = bool(
        fixed["finite"] and fixed["f22_speed_at_60s_passed"]
        and fixed_mav["death_reason"] != "Crash_LowAlt"
        and fixed_mav["steps"] >= 1000
        and fixed_mav["min_altitude_m"] >= 4500.0
    )
    formal_ok = all(
        item["death_reason"] != "Crash_LowAlt" for item in formal
    )
    failure_classification = classify_audit_failure(
        reset["passed_reset_contract"], authority_ok, fixed_ok, formal_ok
    )
    result = {
        "config": config,
        "reset_contract": reset,
        "fixed_neutral_120s": fixed,
        "fixed_neutral_1000step": fixed,
        "formal_short_rollout": {
            "episodes": formal,
            "reset_low_speed_cold_start_detected": any(
                item["reset_low_speed_cold_start"] for item in formal
            ),
        },
        "throttle_authority_passed": authority_ok,
        "failure_classification": failure_classification,
        "passed": failure_classification == "passed",
    }
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tam_airborne_initialization.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# TAM Airborne Initialization Audit", "",
        f"- Reset contract passed: `{reset['passed_reset_contract']}`",
        f"- Fixed-neutral finite: `{fixed['finite']}`",
        f"- F22 speed at 60 s >= 150 m/s: `{fixed['f22_speed_at_60s_passed']}`",
        f"- Reset low-speed cold start detected: `{result['formal_short_rollout']['reset_low_speed_cold_start_detected']}`",
        f"- Overall audit passed: `{result['passed']}`", "",
        f"- Classification: `{failure_classification}`", "",
        "## Formal episodes", "",
    ]
    for item in formal:
        lines.append(
            f"- seed={item['seed']}: death={item['death_reason']} at "
            f"{item['death_time_sec']:.1f}s; class={item['outcome_classification']}"
        )
    (out_dir / "tam_airborne_initialization.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_audit(args.config, args.output_dir, seed=args.seed)
    print(json.dumps({
        "passed": result["passed"],
        "reset_contract": result["reset_contract"]["passed_reset_contract"],
        "f22_speed_at_60s": result["fixed_neutral_120s"]["aircraft"]["red_0"]["speed_at_60s_mps"],
    }, indent=2))


if __name__ == "__main__":
    main()
