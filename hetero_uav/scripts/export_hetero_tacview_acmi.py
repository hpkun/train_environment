"""Export a scripted hetero JSBSim rollout to Tacview ACMI.

This is a visualization/audit utility only. It does not train, load MAPPO
models, or modify environment mechanics.
"""
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

from algorithms.mappo.opponent_policy import OpponentPolicy


def _aircraft_acmi_id(agent_id: str) -> int:
    side, index_text = agent_id.split("_", 1)
    base = 100 if side == "red" else 200
    return base + int(index_text)


def _finite_float(value, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"non-finite Tacview value for {name}: {value!r}")
    return out


def _aircraft_name(env, agent_id: str, sim) -> str:
    role = getattr(env, "agent_roles", {}).get(agent_id, "")
    model = getattr(env, "agent_models", {}).get(agent_id, getattr(sim, "model", ""))
    parts = [agent_id]
    if role:
        parts.append(str(role))
    if model:
        parts.append(str(model))
    return "_".join(parts)


def _aircraft_entries(env) -> list[dict]:
    entries = []
    for agent_id in list(env.red_ids) + list(env.blue_ids):
        sim = env.red_planes.get(agent_id) or env.blue_planes.get(agent_id)
        if sim is None:
            continue
        lon, lat, alt = sim.get_geodetic()
        roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        entries.append({
            "acmi_id": _aircraft_acmi_id(agent_id),
            "type": "Air+FixedWing",
            "lon": _finite_float(lon, f"{agent_id}.lon"),
            "lat": _finite_float(lat, f"{agent_id}.lat"),
            "alt": _finite_float(alt, f"{agent_id}.alt"),
            "roll": _finite_float(roll, f"{agent_id}.roll"),
            "pitch": _finite_float(pitch, f"{agent_id}.pitch"),
            "yaw": _finite_float(yaw, f"{agent_id}.yaw"),
            "name": _aircraft_name(env, agent_id, sim),
            "color": "Red" if agent_id.startswith("red_") else "Blue",
            "alive": bool(sim.is_alive),
        })
    return entries


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


def _missile_entries(env, missile_id_map: dict[str, int]) -> list[dict]:
    entries = []
    for missile in _all_missiles(env):
        uid = getattr(missile, "uid", str(id(missile)))
        if uid not in missile_id_map:
            missile_id_map[uid] = 1000 + len(missile_id_map)
        if not bool(getattr(missile, "is_alive", False)):
            continue
        lon, lat, alt = missile.get_geodetic()
        roll, pitch, yaw = np.asarray(missile.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        entries.append({
            "acmi_id": missile_id_map[uid],
            "type": "Weapon+Missile",
            "lon": _finite_float(lon, f"{uid}.lon"),
            "lat": _finite_float(lat, f"{uid}.lat"),
            "alt": _finite_float(alt, f"{uid}.alt"),
            "roll": _finite_float(roll, f"{uid}.roll"),
            "pitch": _finite_float(pitch, f"{uid}.pitch"),
            "yaw": _finite_float(yaw, f"{uid}.yaw"),
            "name": str(getattr(missile, "model", "AIM-9L")).upper(),
            "color": str(getattr(missile, "color", "White")),
            "alive": True,
        })
    return entries


def _missile_explosions(
    env, missile_id_map: dict[str, int], logged_explosions: set[str]
) -> list[dict]:
    explosions = []
    for missile in _all_missiles(env):
        uid = getattr(missile, "uid", str(id(missile)))
        if uid in logged_explosions:
            continue
        if uid not in missile_id_map:
            missile_id_map[uid] = 1000 + len(missile_id_map)
        if bool(getattr(missile, "is_done", False)) and bool(
            getattr(missile, "is_success", False)
        ):
            lon, lat, alt = missile.get_geodetic()
            explosions.append({
                "acmi_id": missile_id_map[uid],
                "lon": _finite_float(lon, f"{uid}.explosion.lon"),
                "lat": _finite_float(lat, f"{uid}.explosion.lat"),
                "alt": _finite_float(alt, f"{uid}.explosion.alt"),
                "color": "Yellow",
                "radius": _finite_float(getattr(missile, "_Rc", 300.0), f"{uid}.radius"),
            })
            logged_explosions.add(uid)
    return explosions


def _alive_counts(env) -> tuple[int, int, bool]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    mav_alive = False
    for aid in env.red_ids:
        if getattr(env, "agent_roles", {}).get(aid) == "mav":
            sim = env.red_planes.get(aid)
            mav_alive = bool(sim is not None and sim.is_alive)
            break
    return red_alive, blue_alive, mav_alive


def _launch_record_key(record: dict) -> tuple:
    return (
        record.get("missile_id"),
        record.get("shooter_id"),
        record.get("target_id"),
        record.get("launch_step"),
        record.get("physics_frame"),
    )


def _merge_launch_records(records: list[dict], new_records: list[dict]) -> None:
    seen = {_launch_record_key(record) for record in records}
    for record in new_records:
        key = _launch_record_key(record)
        if key not in seen:
            records.append(dict(record))
            seen.add(key)


def _launch_metadata(env, records: list[dict]) -> dict:

    by_target_role: dict[str, int] = {}
    by_shooter: dict[str, int] = {}
    mav_targeted = 0
    roles = getattr(env, "agent_roles", {})
    for record in records:
        shooter = str(record.get("shooter_id", ""))
        target = str(record.get("target_id", ""))
        target_role = str(record.get("target_role") or roles.get(target, ""))
        by_shooter[shooter] = by_shooter.get(shooter, 0) + 1
        if target_role:
            by_target_role[target_role] = by_target_role.get(target_role, 0) + 1
        if target_role == "mav":
            mav_targeted += 1

    return {
        "missile_launch_counts": by_shooter,
        "launch_records_count": len(records),
        "launch_records_by_target_role": by_target_role,
        "launch_records_by_shooter": by_shooter,
        "mav_targeted_by_missile_count": mav_targeted,
    }


def _record_frame(
    logger,
    env,
    sim_time: float,
    record_missiles: bool,
    missile_id_map: dict[str, int],
    logged_explosions: set[str],
) -> None:
    entries = _aircraft_entries(env)
    explosions = []
    if record_missiles:
        entries.extend(_missile_entries(env, missile_id_map))
        explosions = _missile_explosions(env, missile_id_map, logged_explosions)
    logger.record_frame(sim_time, entries, explosions)


def _red_actions(env, policy_name: str, rng: np.random.Generator) -> dict:
    if policy_name == "zero":
        return {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
    if policy_name == "random":
        return {
            rid: rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            for rid in env.red_ids
        }
    raise ValueError(f"unsupported red policy: {policy_name}")


def _all_done(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(
        bool(v) for v in truncated.values()
    )


def _safe_position(sim) -> np.ndarray:
    try:
        return np.asarray(sim.get_position(), dtype=np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _nearest_range_to_team(env, sim, team_ids: list[str], team_planes: dict) -> float:
    pos = _safe_position(sim)
    best = float("nan")
    for aid in team_ids:
        other = team_planes.get(aid)
        if other is None or not bool(getattr(other, "is_alive", False)):
            continue
        distance = float(np.linalg.norm(pos - _safe_position(other)))
        if not math.isfinite(best) or distance < best:
            best = distance
    return best


def _safe_debug_float(debug: dict, key: str) -> float:
    try:
        return float(debug.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")


def _agent_diag_row(
    env,
    step: int,
    sim_time: float,
    aid: str,
    sim,
    action: np.ndarray,
    nearest_enemy_range_m: float,
    debug: dict | None = None,
) -> dict:
    debug = debug or {}
    pos = _safe_position(sim)
    vel = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    return {
        "step": int(step),
        "sim_time_sec": float(sim_time),
        "agent_id": aid,
        "alive": int(bool(getattr(sim, "is_alive", False))),
        "north_m": float(pos[0]) if pos.size > 0 else 0.0,
        "east_m": float(pos[1]) if pos.size > 1 else 0.0,
        "up_m": float(pos[2]) if pos.size > 2 else 0.0,
        "alt_m": float(sim.get_altitude()) if hasattr(sim, "get_altitude") else float(pos[2] if pos.size > 2 else 0.0),
        "speed_mps": speed,
        "roll_deg": float(np.rad2deg(roll)),
        "pitch_deg": float(np.rad2deg(pitch)),
        "yaw_deg": float(np.rad2deg(yaw)),
        "action_pitch": float(action[0]) if len(action) > 0 else 0.0,
        "action_heading": float(action[1]) if len(action) > 1 else 0.0,
        "action_speed": float(action[2]) if len(action) > 2 else 0.0,
        "nearest_enemy_range_m": float(nearest_enemy_range_m),
        "death_reason": str(getattr(env, "_death_reasons", {}).get(aid, "")),
        "desired_heading_source": str(debug.get("desired_heading_source", "")),
        "roll_recovery_active": int(debug.get("roll_recovery_active", 0) or 0),
        "extreme_roll_recovery_active": int(debug.get("extreme_roll_recovery_active", 0) or 0),
        "simple_lost_steps": int(debug.get("simple_lost_steps", 0) or 0),
        "selected_range_m": _safe_debug_float(debug, "selected_range_m"),
        "selected_AO_rad": _safe_debug_float(debug, "selected_AO_rad"),
        "selected_TA_rad": _safe_debug_float(debug, "selected_TA_rad"),
        "selected_target_quality": str(debug.get("selected_target_quality", "")),
        "selected_rel_body_x_norm": _safe_debug_float(debug, "selected_rel_body_x_norm"),
        "selected_rel_body_y_norm": _safe_debug_float(debug, "selected_rel_body_y_norm"),
        "selected_rel_body_up_norm": _safe_debug_float(debug, "selected_rel_body_up_norm"),
        "action_heading_abs_rad": _safe_debug_float(debug, "action_heading_abs_rad"),
    }


def _append_diag_rows(
    env,
    step: int,
    sim_time: float,
    red_actions: dict[str, np.ndarray],
    blue_actions: dict[str, np.ndarray],
    red_rows: list[dict],
    blue_rows: list[dict],
) -> None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from rule_based_agent import _simple_debug_state
    except Exception:
        _simple_debug_state = {}

    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None:
            continue
        nearest = _nearest_range_to_team(env, sim, env.blue_ids, env.blue_planes)
        red_rows.append(_agent_diag_row(
            env, step, sim_time, rid, sim,
            np.asarray(red_actions.get(rid, np.zeros(3)), dtype=np.float32),
            nearest,
        ))
    for index, bid in enumerate(env.blue_ids):
        sim = env.blue_planes.get(bid)
        if sim is None:
            continue
        nearest = _nearest_range_to_team(env, sim, env.red_ids, env.red_planes)
        blue_rows.append(_agent_diag_row(
            env, step, sim_time, bid, sim,
            np.asarray(blue_actions.get(bid, np.zeros(3)), dtype=np.float32),
            nearest,
            dict(_simple_debug_state.get(index, {})),
        ))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--red-policy", choices=["zero", "random"], default="zero")
    parser.add_argument(
        "--blue-policy",
        choices=["zero", "rule_nearest", "greedy_fsm", "random", "brma_rule", "brma_rule_safe_pursuit", "tam_greedy_easy", "brma_rule_safe_pursuit_easy"],
        default="greedy_fsm",
    )
    parser.add_argument(
        "--output-acmi",
        default="outputs/tacview/hetero_rollout.acmi",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tacview/hetero_rollout_meta.json",
    )
    parser.add_argument("--record-missiles", action="store_true")
    parser.add_argument("--reference-time", default="2026-01-01T00:00:00Z")
    parser.add_argument("--disable-config-trim", action="store_true")
    args = parser.parse_args()

    output_acmi = Path(args.output_acmi)
    output_json = Path(args.output_json)
    output_acmi.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    from uav_env import make_env
    from uav_env.JSBSim.render_tacview import TacviewLogger

    env = make_env(args.config, env_type="jsbsim_hetero")
    if args.disable_config_trim and hasattr(env, "set_action_trim_enabled"):
        env.set_action_trim_enabled(False)
    logger = TacviewLogger(reference_time=args.reference_time)
    rng = np.random.default_rng(args.seed)
    blue_policy = OpponentPolicy(mode=args.blue_policy, seed=args.seed + 17)
    missile_id_map: dict[str, int] = {}
    logged_explosions: set[str] = set()
    launch_records: list[dict] = []
    red_diag_rows: list[dict] = []
    blue_diag_rows: list[dict] = []
    steps_executed = 0
    final_terminated = False
    final_truncated = False

    try:
        obs, _info = env.reset(seed=args.seed)
        _append_diag_rows(
            env, 0, 0.0,
            {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids},
            {bid: np.zeros(3, dtype=np.float32) for bid in env.blue_ids},
            red_diag_rows, blue_diag_rows,
        )
        _record_frame(
            logger, env, 0.0, args.record_missiles,
            missile_id_map, logged_explosions)

        for step in range(1, args.steps + 1):
            red_actions = _red_actions(env, args.red_policy, rng)
            blue_actions = blue_policy.act(obs, env.blue_ids, env=env)
            actions = dict(red_actions)
            actions.update(blue_actions)
            obs, _rewards, terminated, truncated, _info = env.step(actions)
            sim_time = step * float(env.env_dt)
            _append_diag_rows(
                env, step, sim_time, red_actions, blue_actions,
                red_diag_rows, blue_diag_rows,
            )
            _merge_launch_records(launch_records, _info.get("__launch_quality_step__", []))
            _merge_launch_records(launch_records, _info.get("__launch_quality_done__", []))
            steps_executed = step
            _record_frame(
                logger, env, sim_time, args.record_missiles,
                missile_id_map, logged_explosions)
            final_terminated = all(bool(v) for v in terminated.values())
            final_truncated = all(bool(v) for v in truncated.values())
            if _all_done(terminated, truncated):
                break

        logger.write(str(output_acmi))
        diagnostics_dir = output_acmi.parent / "diagnostics"
        blue_diag_path = diagnostics_dir / "blue_behavior_timeseries.csv"
        red_diag_path = diagnostics_dir / "red_behavior_timeseries.csv"
        _write_csv(blue_diag_path, blue_diag_rows)
        _write_csv(red_diag_path, red_diag_rows)
        red_alive, blue_alive, mav_alive = _alive_counts(env)
        metadata = {
            "acmi_entity_type_fix": True,
            "config": args.config,
            "output_acmi": str(output_acmi),
            "output_json": str(output_json),
            "blue_behavior_timeseries": str(blue_diag_path),
            "red_behavior_timeseries": str(red_diag_path),
            "steps_requested": int(args.steps),
            "steps_executed": int(steps_executed),
            "frames_recorded": int(logger.frame_count),
            "terminated": bool(final_terminated),
            "truncated": bool(final_truncated),
            "final_red_alive": int(red_alive),
            "final_blue_alive": int(blue_alive),
            "final_mav_alive": bool(mav_alive),
            "red_policy": args.red_policy,
            "blue_policy": args.blue_policy,
            "record_missiles": bool(args.record_missiles),
            "env_max_steps": int(getattr(env, "max_steps", 0)),
            "sim_freq": int(getattr(env, "sim_freq", 0)),
            "agent_interaction_steps": int(getattr(env, "agent_interaction_steps", 0)),
            "decision_dt": float(getattr(env, "env_dt", 0.0)),
            "reference_time": args.reference_time,
            "missiles_seen": len(missile_id_map),
            "action_trim_by_role": {
                key: [round(float(v), 6) for v in value]
                for key, value in getattr(env, "action_trim_by_role", {}).items()
            },
            "trim_enabled": bool(getattr(env, "action_trim_enabled", False)),
            "note": "ACMI red-policy zero may still include config-level MAV trim if enabled.",
        }
        metadata.update(_launch_metadata(env, launch_records))
        output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"output_acmi: {output_acmi}")
    print(f"output_json: {output_json}")
    print(f"frames_recorded: {logger.frame_count}")
    print(f"steps_executed: {steps_executed}")
    print(f"final_red_alive: {red_alive}")
    print(f"final_blue_alive: {blue_alive}")
    print(f"final_mav_alive: {mav_alive}")


if __name__ == "__main__":
    main()
