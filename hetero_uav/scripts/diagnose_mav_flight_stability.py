"""Diagnose MAV/A-4 flight stability in hetero JSBSim configs.

This is a diagnostic-only script. It does not train, load MAPPO, or modify the
environment, PID controller, reward, termination, aircraft XML, or initial
states.
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

from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.export_hetero_tacview_acmi import _record_frame


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _case_specs() -> list[dict]:
    return [
        {"case": "zero_all", "mav_action": [0.0, 0.0, 0.0], "attack_action": [0.0, 0.0, 0.0]},
        {"case": "mav_zero_attack_zero", "mav_action": [0.0, 0.0, 0.0], "attack_action": [0.0, 0.0, 0.0]},
        {"case": "mav_pitch_bias_005", "mav_action": [0.05, 0.0, 0.0], "attack_action": [0.0, 0.0, 0.0]},
        {"case": "mav_pitch_bias_010", "mav_action": [0.10, 0.0, 0.0], "attack_action": [0.0, 0.0, 0.0]},
        {"case": "mav_level_speed_up", "mav_action": [0.05, 0.0, 0.5], "attack_action": [0.0, 0.0, 0.0]},
        {"case": "red_random_bounded", "random_bound": 0.3},
    ]


def _all_done(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(bool(v) for v in truncated.values())


def _mav_id(env) -> str:
    roles = getattr(env, "agent_roles", {})
    for aid in env.red_ids:
        if roles.get(aid) == "mav":
            return aid
    return env.red_ids[0]


def _red_actions(env, case: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if "random_bound" in case:
        bound = float(case["random_bound"])
        return {
            aid: rng.uniform(-bound, bound, size=3).astype(np.float32)
            for aid in env.red_ids
        }

    mav_id = _mav_id(env)
    actions = {}
    for aid in env.red_ids:
        values = case["mav_action"] if aid == mav_id else case["attack_action"]
        actions[aid] = np.asarray(values, dtype=np.float32)
    return actions


def _blue_actions(policy: OpponentPolicy, obs: dict, blue_ids: list[str]) -> dict[str, np.ndarray]:
    return policy.act(obs, blue_ids)


def _scan_mav(env, mav_id: str) -> dict:
    sim = env.red_planes[mav_id]
    geodetic = sim.get_geodetic().astype(np.float64)
    position = sim.get_position().astype(np.float64)
    velocity = sim.get_velocity().astype(np.float64)
    rpy = np.asarray(sim.get_rpy(), dtype=np.float64)
    values = np.concatenate([geodetic, position, velocity, rpy])
    speed = float(np.linalg.norm(velocity))
    return {
        "altitude_m": float(geodetic[2]),
        "speed_mps": speed,
        "pitch_deg": float(np.rad2deg(rpy[1])),
        "roll_deg": float(np.rad2deg(rpy[0])),
        "heading_deg": float(np.rad2deg(rpy[2])),
        "alive": bool(sim.is_alive),
        "crashed": bool(sim.is_crash or geodetic[2] <= 0.0),
        "nan_detected": bool(np.isnan(values).any() or np.isinf(values).any()),
    }


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _all_missiles(env) -> list:
    seen = set()
    missiles = []
    for sim in list(env.red_planes.values()) + list(env.blue_planes.values()):
        for missile in getattr(sim, "launch_missiles", []):
            uid = getattr(missile, "uid", str(id(missile)))
            if uid in seen:
                continue
            seen.add(uid)
            missiles.append(missile)
    return missiles


def _config_stem(config: str) -> str:
    return Path(config).stem


def run_case(
    *,
    config: str,
    case: dict,
    steps: int,
    seed: int,
    blue_policy_name: str,
    export_acmi: bool,
    output_acmi_dir: Path,
) -> dict:
    from uav_env import make_env
    from uav_env.JSBSim.render_tacview import TacviewLogger

    env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rng = np.random.default_rng(seed)
    blue_policy = OpponentPolicy(mode=blue_policy_name, seed=seed + 17)
    missile_id_map: dict[str, int] = {}
    logged_explosions: set[str] = set()
    logger = TacviewLogger() if export_acmi else None
    steps_executed = 0
    automatic_launches_seen = 0

    try:
        obs, _info = env.reset(seed=seed)
        mav_id = _mav_id(env)
        initial = _scan_mav(env, mav_id)
        min_altitude = initial["altitude_m"]
        max_altitude = initial["altitude_m"]
        min_speed = initial["speed_mps"]
        max_speed = initial["speed_mps"]
        nan_detected = initial["nan_detected"]

        if logger is not None:
            _record_frame(logger, env, 0.0, True, missile_id_map, logged_explosions)

        for step in range(1, int(steps) + 1):
            actions = _red_actions(env, case, rng)
            actions.update(_blue_actions(blue_policy, obs, env.blue_ids))
            obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed = step
            scan = _scan_mav(env, mav_id)
            min_altitude = min(min_altitude, scan["altitude_m"])
            max_altitude = max(max_altitude, scan["altitude_m"])
            min_speed = min(min_speed, scan["speed_mps"])
            max_speed = max(max_speed, scan["speed_mps"])
            nan_detected = nan_detected or scan["nan_detected"]
            automatic_launches_seen += len(info.get("__launch_quality_step__", []))

            if logger is not None:
                _record_frame(logger, env, step * float(env.env_dt), True, missile_id_map, logged_explosions)
            if _all_done(terminated, truncated):
                break

        final = _scan_mav(env, mav_id)
        red_alive, blue_alive = _alive_counts(env)
        missiles_seen = len(missile_id_map) if logger is not None else len(_all_missiles(env))
        altitude_delta = final["altitude_m"] - initial["altitude_m"]
        stable_level_like = (
            final["alive"]
            and not nan_detected
            and min_altitude > 1000.0
            and abs(altitude_delta) < 1000.0
        )
        needs_fix = (
            case["case"] in {"zero_all", "mav_zero_attack_zero"}
            and (final["crashed"] or min_altitude < 1000.0 or altitude_delta < -2000.0)
        )

        acmi_path = ""
        if logger is not None:
            output_acmi_dir.mkdir(parents=True, exist_ok=True)
            acmi_path = str(output_acmi_dir / f"{_config_stem(config)}_{case['case']}.acmi")
            logger.write(acmi_path)

        return {
            "config": config,
            "case": case["case"],
            "steps_executed": int(steps_executed),
            "crashed": bool(final["crashed"]),
            "mav_alive_final": bool(final["alive"]),
            "mav_crashed": bool(final["crashed"]),
            "mav_initial_altitude_m": float(initial["altitude_m"]),
            "mav_final_altitude_m": float(final["altitude_m"]),
            "mav_min_altitude_m": float(min_altitude),
            "mav_max_altitude_m": float(max_altitude),
            "mav_altitude_delta_m": float(altitude_delta),
            "mav_initial_speed_mps": float(initial["speed_mps"]),
            "mav_final_speed_mps": float(final["speed_mps"]),
            "mav_min_speed_mps": float(min_speed),
            "mav_max_speed_mps": float(max_speed),
            "mav_final_pitch_deg": float(final["pitch_deg"]),
            "mav_final_roll_deg": float(final["roll_deg"]),
            "mav_final_heading_deg": float(final["heading_deg"]),
            "nan_detected": bool(nan_detected),
            "red_alive_final": int(red_alive),
            "blue_alive_final": int(blue_alive),
            "missiles_seen": int(missiles_seen),
            "automatic_launches_seen": int(automatic_launches_seen),
            "stable_level_like": bool(stable_level_like),
            "needs_mav_stability_fix": bool(needs_fix),
            "output_acmi": acmi_path,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _summary(records: list[dict], configs: list[str]) -> dict:
    zero_records = [
        record for record in records
        if record["case"] in {"zero_all", "mav_zero_attack_zero"}
    ]
    bias_005 = [record for record in records if record["case"] == "mav_pitch_bias_005"]
    bias_010 = [record for record in records if record["case"] == "mav_pitch_bias_010"]
    zero_action_mav_stable = bool(zero_records and all(r["stable_level_like"] for r in zero_records))
    pitch_bias_improves_altitude = False
    for config in configs:
        z = next((r for r in records if r["config"] == config and r["case"] == "zero_all"), None)
        b05 = next((r for r in bias_005 if r["config"] == config), None)
        b10 = next((r for r in bias_010 if r["config"] == config), None)
        if z and ((b05 and b05["mav_final_altitude_m"] > z["mav_final_altitude_m"]) or
                  (b10 and b10["mav_final_altitude_m"] > z["mav_final_altitude_m"])):
            pitch_bias_improves_altitude = True

    recommended = []
    if pitch_bias_improves_altitude:
        recommended.append("If pitch bias improves altitude retention, discuss MAV trim or safe default only after this audit.")
    if any(record["needs_mav_stability_fix"] for record in records):
        recommended.append("Zero-policy MAV instability should be treated as environment/controller/model integration, not untrained RL.")
    recommended.append("Do not train or enter method modules before deciding the MAV stability policy.")

    return {
        "configs_checked": len(configs),
        "cases_checked": len(records),
        "zero_action_mav_stable": zero_action_mav_stable,
        "pitch_bias_improves_altitude": bool(pitch_bias_improves_altitude),
        "recommended_next_actions": recommended,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blue-policy",
        choices=["zero", "rule_nearest", "greedy_fsm"],
        default="zero",
    )
    parser.add_argument("--output-json", default="outputs/environment_audit/mav_flight_stability.json")
    parser.add_argument("--output-acmi-dir", default="outputs/tacview/mav_flight_stability")
    parser.add_argument("--export-acmi", action="store_true")
    args = parser.parse_args()

    output_acmi_dir = Path(args.output_acmi_dir)
    records = []
    for config in args.configs:
        for case in _case_specs():
            record = run_case(
                config=config,
                case=case,
                steps=args.steps,
                seed=args.seed,
                blue_policy_name=args.blue_policy,
                export_acmi=args.export_acmi,
                output_acmi_dir=output_acmi_dir,
            )
            records.append(record)
            print(
                f"{Path(config).stem} {record['case']}: "
                f"steps={record['steps_executed']} "
                f"alt_delta={record['mav_altitude_delta_m']:.3f} "
                f"min_alt={record['mav_min_altitude_m']:.3f} "
                f"alive={record['mav_alive_final']} "
                f"crashed={record['mav_crashed']} "
                f"stable={record['stable_level_like']}"
            )

    data = {"summary": _summary(records, list(args.configs)), "records": records}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"output_json: {output_json}")
    if args.export_acmi:
        print(f"output_acmi_dir: {output_acmi_dir}")


if __name__ == "__main__":
    main()
