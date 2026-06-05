"""Export a scripted hetero JSBSim rollout to Tacview ACMI.

This is a visualization/audit utility only. It does not train, load MAPPO
models, or modify environment mechanics.
"""
from __future__ import annotations

import argparse
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
        choices=["zero", "rule_nearest", "greedy_fsm", "random"],
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
    steps_executed = 0
    final_terminated = False
    final_truncated = False

    try:
        obs, _info = env.reset(seed=args.seed)
        _record_frame(
            logger, env, 0.0, args.record_missiles,
            missile_id_map, logged_explosions)

        for step in range(1, args.steps + 1):
            actions = _red_actions(env, args.red_policy, rng)
            actions.update(blue_policy.act(obs, env.blue_ids))
            obs, _rewards, terminated, truncated, _info = env.step(actions)
            _merge_launch_records(launch_records, _info.get("__launch_quality_step__", []))
            _merge_launch_records(launch_records, _info.get("__launch_quality_done__", []))
            steps_executed = step
            _record_frame(
                logger, env, step * float(env.env_dt), args.record_missiles,
                missile_id_map, logged_explosions)
            final_terminated = all(bool(v) for v in terminated.values())
            final_truncated = all(bool(v) for v in truncated.values())
            if _all_done(terminated, truncated):
                break

        logger.write(str(output_acmi))
        red_alive, blue_alive, mav_alive = _alive_counts(env)
        metadata = {
            "acmi_entity_type_fix": True,
            "config": args.config,
            "output_acmi": str(output_acmi),
            "output_json": str(output_json),
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
