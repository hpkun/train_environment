"""Environment rationality audit for JSBSim heterogeneous air combat.

This script performs static checks plus no-training scripted rollouts. It is
intentionally read-only with respect to environment, reward, missile, PID,
aircraft XML, blue rule, action space, observation space and trainer logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env
from uav_env.JSBSim.alignment.los_geometry import compute_3d_range


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v7_role_aligned.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_brma_paper_homogeneous_v1.yaml",
    "uav_env/JSBSim/configs/hetero_3v2_all_attack_uav_brma_paper_homogeneous_v1.yaml",
]

STATIC_CONFIGS = [
    *DEFAULT_CONFIGS,
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]

POLICIES = [
    "zero_action_red_vs_blue_rule",
    "level_flight_red_vs_blue_rule",
    "straight_chase_red_vs_blue_rule",
    "oracle_geometry_red_vs_blue_rule",
    "oracle_launch_window_red_vs_blue_rule",
    "oracle_launch_window_red_vs_blue_zero",
    "obs_limited_chase_red_vs_blue_rule",
    "obs_limited_chase_red_vs_blue_zero",
    "obs_limited_chase_red_vs_blue_rule_with_mav_shared",
    "red_rule_vs_blue_zero",
    "red_rule_vs_blue_rule_symmetric_all_attack",
    "blue_rule_only_strength_probe",
]

LAUNCH_DIAG_FIELDS = [
    "range_ok_pairs",
    "ao_ok_pairs",
    "ta_ok_pairs",
    "boresight_ok_pairs",
    "geometry_ok_pairs",
    "direct_track_candidates",
    "mav_shared_track_candidates",
    "track_unobserved_blocked",
    "role_blocked_mav",
    "ammo_empty_blocked",
    "cooldown_blocked",
    "kill_cooldown_blocked",
    "lock_delay_blocked",
    "lock_started",
    "lock_continued",
    "lock_lost",
    "lock_mature_pairs",
    "engaged_blocked",
    "launches",
    "alive_enemy_pairs",
    "unengaged_enemy_pairs",
]

BLOCKED_FIELDS = [
    "track_unobserved_blocked",
    "role_blocked_mav",
    "ammo_empty_blocked",
    "cooldown_blocked",
    "kill_cooldown_blocked",
    "lock_delay_blocked",
    "engaged_blocked",
]


def _read(path: str) -> str:
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _side(agent_id: str) -> str:
    return "red" if str(agent_id).startswith("red_") else "blue"


def _alive_count(env, side: str) -> int:
    planes = env.red_planes if side == "red" else env.blue_planes
    return sum(1 for sim in planes.values() if sim.is_alive)


def _role(env, aid: str) -> str:
    return str(getattr(env, "agent_roles", {}).get(aid, ""))


def _agent_state(env, aid: str) -> dict[str, float]:
    sim = env._get_sim(aid)
    if sim is None:
        return {
            "alive": 0,
            "altitude_m": math.nan,
            "speed_mps": math.nan,
            "roll_rad": math.nan,
            "pitch_rad": math.nan,
            "yaw_rad": math.nan,
            "vertical_speed_mps": math.nan,
        }
    vel = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64)
    return {
        "alive": int(bool(sim.is_alive)),
        "altitude_m": float(sim.get_geodetic()[2]),
        "speed_mps": float(np.linalg.norm(vel)),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "yaw_rad": float(yaw),
        "vertical_speed_mps": float(vel[2]) if vel.size >= 3 else math.nan,
    }


def _bearing_norm(src: np.ndarray, dst: np.ndarray) -> float:
    delta = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    heading = math.atan2(float(delta[1]), float(delta[0]))
    return float(np.clip(heading / math.pi, -1.0, 1.0))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _nearest_alive(pos: np.ndarray, planes: dict[str, Any]) -> tuple[str | None, Any | None]:
    best_id = None
    best_sim = None
    best_dist = float("inf")
    for aid, sim in planes.items():
        if not sim.is_alive:
            continue
        dist = compute_3d_range(pos, sim.get_position())
        if dist < best_dist:
            best_id, best_sim, best_dist = aid, sim, dist
    return best_id, best_sim


def _safe_action_for_altitude(altitude_m: float, heading_norm: float, speed: float = 0.55) -> np.ndarray:
    if altitude_m < 3500.0:
        pitch = 0.25
    elif altitude_m < 7000.0:
        pitch = 0.05
    elif altitude_m > 8500.0:
        pitch = -0.08
    else:
        pitch = 0.0
    return np.asarray([pitch, float(np.clip(heading_norm, -1.0, 1.0)), speed], dtype=np.float32)


def _level_action(sim, speed: float = 0.2) -> np.ndarray:
    return np.asarray([0.0, float(sim.get_rpy()[2]) / math.pi, speed], dtype=np.float32)


def _obs_array(obs: dict, key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(obs.get(key, []), dtype=np.float32)
    if shape is not None and arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return arr


def _obs_limited_enemy_choice(obs: dict, allow_mav_shared: bool) -> dict[str, Any]:
    """Select a target using only fields present in one actor observation."""

    enemy_states = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
    enemy_geo = np.asarray(obs.get("enemy_geo_states", []), dtype=np.float32)
    alive = np.asarray(obs.get("enemy_alive_mask", []), dtype=np.float32).reshape(-1)
    observed = np.asarray(obs.get("enemy_observed_mask", []), dtype=np.float32).reshape(-1)
    source = np.asarray(obs.get("enemy_track_source", []), dtype=np.float32)
    if source.ndim != 2:
        source = np.zeros((alive.size, 2), dtype=np.float32)
    if enemy_states.ndim != 2:
        enemy_states = np.zeros((alive.size, 11), dtype=np.float32)
    if enemy_geo.ndim != 2:
        enemy_geo = np.zeros((alive.size, 5), dtype=np.float32)

    candidates = []
    for idx in range(int(alive.size)):
        is_alive = bool(alive[idx] > 0.5)
        is_observed = bool(idx < observed.size and observed[idx] > 0.5)
        src = source[idx] if idx < source.shape[0] else np.zeros(2, dtype=np.float32)
        has_direct = bool(src.size >= 1 and src[0] > 0.5)
        has_shared = bool(src.size >= 2 and src[1] > 0.5)
        if not is_alive or not is_observed:
            continue
        if not has_direct and not (allow_mav_shared and has_shared):
            continue
        dist_norm = float(enemy_geo[idx, 2]) if idx < enemy_geo.shape[0] and enemy_geo.shape[1] >= 3 else float("inf")
        candidates.append((dist_norm, idx, has_direct, has_shared))

    if not candidates:
        return {"index": "", "reason": "no_observed_target", "heading_norm": 0.0}

    _dist, idx, has_direct, has_shared = min(candidates, key=lambda item: item[0])
    ego_geo = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32).reshape(-1)
    ego_yaw = float(ego_geo[5] * math.pi) if ego_geo.size >= 6 else 0.0
    if idx < enemy_states.shape[0] and enemy_states.shape[1] >= 2:
        body_x = float(enemy_states[idx, 0])
        body_y = float(enemy_states[idx, 1])
        rel_heading = math.atan2(body_y, body_x) if abs(body_x) + abs(body_y) > 1e-6 else 0.0
        heading = _wrap_pi(ego_yaw + rel_heading)
        reason = "enemy_states_body_xy"
    else:
        heading = ego_yaw
        reason = "fallback_keep_heading_no_body_xy"
    return {
        "index": idx,
        "reason": reason,
        "heading_norm": float(np.clip(heading / math.pi, -1.0, 1.0)),
        "target_source": "direct" if has_direct else "mav_shared",
        "has_direct_track": int(has_direct),
        "has_mav_shared_track": int(has_shared),
        "enemy_alive": 1,
        "distance_norm_or_raw": float(_dist),
        "ata_norm_or_raw": float(enemy_geo[idx, 3]) if idx < enemy_geo.shape[0] and enemy_geo.shape[1] >= 4 else "",
        "aa_norm_or_raw": float(enemy_geo[idx, 4]) if idx < enemy_geo.shape[0] and enemy_geo.shape[1] >= 5 else "",
    }


def _obs_limited_red_actions(env, obs: dict, policy: str, decision_rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, np.ndarray]:
    allow_shared = policy.endswith("_with_mav_shared") or "mav_shared_geo" in str(getattr(env, "observation_mode", ""))
    actions: dict[str, np.ndarray] = {}
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None or not sim.is_alive:
            actions[rid] = np.zeros(3, dtype=np.float32)
            continue
        role = _role(env, rid)
        if role == "mav":
            act = _level_action(sim, 0.2)
            choice = {"index": "", "target_source": "", "reason": "mav_support_loiter"}
        else:
            choice = _obs_limited_enemy_choice(obs.get(rid, {}), allow_mav_shared=allow_shared)
            heading = _finite_float(choice.get("heading_norm"), float(sim.get_rpy()[2]) / math.pi)
            act = _safe_action_for_altitude(_agent_state(env, rid)["altitude_m"], heading, speed=0.55)
        decision_rows.append({
            **context,
            "agent_id": rid,
            "role": role,
            "selected_target_index": choice.get("index", ""),
            "target_source": choice.get("target_source", ""),
            "has_direct_track": choice.get("has_direct_track", ""),
            "has_mav_shared_track": choice.get("has_mav_shared_track", ""),
            "enemy_alive": choice.get("enemy_alive", ""),
            "distance_norm_or_raw": choice.get("distance_norm_or_raw", ""),
            "ata_norm_or_raw": choice.get("ata_norm_or_raw", ""),
            "aa_norm_or_raw": choice.get("aa_norm_or_raw", ""),
            "heading_action": float(act[1]),
            "pitch_action": float(act[0]),
            "speed_action": float(act[2]),
            "decision_reason": choice.get("reason", ""),
        })
        actions[rid] = np.asarray(np.clip(act, -1.0, 1.0), dtype=np.float32)
    return actions


def _oracle_target(env, aid: str, sim) -> tuple[Any | None, dict[str, Any], str]:
    enemies = env.blue_planes if aid.startswith("red_") else env.red_planes
    candidates = []
    for tid, target in enemies.items():
        if not target.is_alive:
            continue
        metrics = env._missile_candidate_metrics(sim, target)
        geom_count = int(metrics.get("range_ok", False)) + int(metrics.get("ao_ok", False)) + int(metrics.get("ta_ok", False)) + int(metrics.get("boresight_ok_3d", False))
        distance = float(metrics.get("range_m", 1e9))
        desired_range_score = -abs(distance - 8000.0)
        score = (
            1000.0 * int(metrics.get("launch_geometry_ok_3d", False))
            + 100.0 * geom_count
            + 10.0 * int(metrics.get("range_ok", False))
            - float(metrics.get("AO_rad", math.pi))
            + float(metrics.get("TA_rad", 0.0))
            + desired_range_score / 10000.0
        )
        candidates.append((score, -distance, tid, target, metrics))
    if not candidates:
        return None, {}, "no_alive_target"
    _score, _neg_d, tid, target, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    return target, metrics, f"oracle_target={tid}"


def _oracle_red_actions(env, policy: str, decision_rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None or not sim.is_alive:
            actions[rid] = np.zeros(3, dtype=np.float32)
            continue
        role = _role(env, rid)
        if role == "mav":
            # MAV support behavior: keep altitude and avoid chasing into the merge.
            heading = float(sim.get_rpy()[2]) / math.pi
            act = _safe_action_for_altitude(_agent_state(env, rid)["altitude_m"], heading, speed=0.25)
            metrics = {}
            target_id = ""
            reason = "mav_rear_support_loiter"
        else:
            target, metrics, reason = _oracle_target(env, rid, sim)
            target_id = getattr(target, "uid", "") if target is not None else ""
            heading = _bearing_norm(sim.get_position(), target.get_position()) if target is not None else float(sim.get_rpy()[2]) / math.pi
            act = _safe_action_for_altitude(_agent_state(env, rid)["altitude_m"], heading, speed=0.55)
        decision_rows.append({
            **context,
            "agent_id": rid,
            "selected_target_id": target_id,
            "range_m": metrics.get("range_m", ""),
            "AO_rad": metrics.get("AO_rad", ""),
            "TA_rad": metrics.get("TA_rad", ""),
            "boresight_rad": metrics.get("boresight_rad", ""),
            "range_ok": metrics.get("range_ok", ""),
            "ao_ok": metrics.get("ao_ok", ""),
            "ta_ok": metrics.get("ta_ok", ""),
            "boresight_ok": metrics.get("boresight_ok_3d", ""),
            "geometry_ok": metrics.get("launch_geometry_ok_3d", ""),
            "track_ok": "",
            "desired_heading_rad": float(act[1] * math.pi),
            "heading_action": float(act[1]),
            "pitch_action": float(act[0]),
            "speed_action": float(act[2]),
            "altitude_m": _agent_state(env, rid)["altitude_m"],
            "decision_reason": reason,
        })
        actions[rid] = np.asarray(np.clip(act, -1.0, 1.0), dtype=np.float32)
    return actions


def _full_state_red_actions(env, policy: str) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None or not sim.is_alive:
            actions[rid] = np.zeros(3, dtype=np.float32)
            continue
        role = _role(env, rid)
        if policy == "zero_action_red_vs_blue_rule":
            act = np.zeros(3, dtype=np.float32)
        elif policy in {"level_flight_red_vs_blue_rule", "blue_rule_only_strength_probe"}:
            act = _level_action(sim, 0.2)
        elif role == "mav":
            act = _level_action(sim, 0.2)
        else:
            _tid, target = _nearest_alive(sim.get_position(), env.blue_planes)
            heading = _bearing_norm(sim.get_position(), target.get_position()) if target is not None else 0.0
            act = _safe_action_for_altitude(_agent_state(env, rid)["altitude_m"], heading, speed=0.6)
        actions[rid] = np.asarray(np.clip(act, -1.0, 1.0), dtype=np.float32)
    return actions


def _blue_actions(env, obs: dict, policy: str, opponent: OpponentPolicy) -> dict[str, np.ndarray]:
    if policy in {"red_rule_vs_blue_zero", "obs_limited_chase_red_vs_blue_zero", "oracle_launch_window_red_vs_blue_zero"}:
        return {bid: np.zeros(3, dtype=np.float32) for bid in env.blue_ids}
    if policy == "red_rule_vs_blue_rule_symmetric_all_attack":
        acts = {}
        for bid in env.blue_ids:
            sim = env.blue_planes.get(bid)
            if sim is None or not sim.is_alive:
                acts[bid] = np.zeros(3, dtype=np.float32)
                continue
            _tid, target = _nearest_alive(sim.get_position(), env.red_planes)
            heading = _bearing_norm(sim.get_position(), target.get_position()) if target is not None else 0.0
            acts[bid] = _safe_action_for_altitude(_agent_state(env, bid)["altitude_m"], heading, speed=0.6)
        return acts
    return opponent.act(obs, env.blue_ids, env=env)


def _red_actions(env, obs: dict, policy: str, decision_rows: dict[str, list[dict[str, Any]]], context: dict[str, Any]) -> dict[str, np.ndarray]:
    if policy.startswith("obs_limited_chase"):
        return _obs_limited_red_actions(env, obs, policy, decision_rows["obs"], context)
    if policy.startswith("oracle_launch_window"):
        return _oracle_red_actions(env, policy, decision_rows["oracle"], context)
    return _full_state_red_actions(env, policy)


def _diag_summary(info: dict, side: str) -> dict[str, Any]:
    diag = (info.get("__launch_diag__", {}) or {}).get(side, {}) or {}
    row: dict[str, Any] = {}
    for key in LAUNCH_DIAG_FIELDS:
        # Missing fields stay blank; existing zero values remain 0.
        row[f"{side}_{key}"] = diag[key] if key in diag else ""
    return row


def _info_schema_probe(configs: list[str], out_dir: Path) -> None:
    probes = []
    for config in configs:
        env = make_env(config, env_type="jsbsim_hetero", max_steps=3)
        try:
            obs, info = env.reset(seed=0)
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            _obs2, _rewards, _terminated, _truncated, step_info = env.step(actions)
            sample_obs = obs.get(env.red_ids[0], {}) if env.red_ids else {}
            probes.append({
                "config": config,
                "reset_info_keys": sorted(info.keys()),
                "step_info_keys": sorted(step_info.keys()),
                "red_obs_keys": sorted(sample_obs.keys()) if isinstance(sample_obs, dict) else [],
                "launch_diag_keys": {
                    side: sorted(((step_info.get("__launch_diag__", {}) or {}).get(side, {}) or {}).keys())
                    for side in ("red", "blue")
                },
                "launch_quality_step_keys": sorted((step_info.get("__launch_quality_step__", [{}]) or [{}])[0].keys()) if step_info.get("__launch_quality_step__") else [],
                "per_agent_info_keys": sorted(step_info.get(env.red_ids[0], {}).keys()) if env.red_ids else [],
            })
        finally:
            env.close()
    (out_dir / "info_schema_probe.json").write_text(json.dumps(probes, indent=2, default=str), encoding="utf-8")
    lines = ["# Info Schema Probe", ""]
    for probe in probes:
        lines.append(f"## {probe['config']}")
        lines.append(f"- step info keys: `{', '.join(probe['step_info_keys'])}`")
        lines.append(f"- red launch diag keys: `{', '.join(probe['launch_diag_keys'].get('red', []))}`")
        lines.append(f"- blue launch diag keys: `{', '.join(probe['launch_diag_keys'].get('blue', []))}`")
        lines.append(f"- red obs keys: `{', '.join(probe['red_obs_keys'])}`")
        lines.append("")
    _write(out_dir / "info_schema_probe.md", "\n".join(lines))


def _make_jittered_configs(configs: list[str], args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    generated = []
    gen_dir = out_dir / "generated_configs"
    gen_dir.mkdir(parents=True, exist_ok=True)
    if not args.jitter:
        for cfg in configs:
            generated.append({
                "base_config": cfg,
                "config": cfg,
                "jitter_seed": "",
                "jitter_applied": False,
                "generated_config_path": "",
            })
        return generated

    for cfg in configs:
        data = _load_yaml(cfg)
        states = data.get("initial_states", {})
        stem = Path(cfg).stem
        for seed in range(int(args.jitter_seeds)):
            rng = np.random.default_rng(seed)
            patched = json.loads(json.dumps(data))
            for _aid, st in (patched.get("initial_states", {}) or {}).items():
                lat0 = float(st.get("lat", 0.0))
                lon0 = float(st.get("lon", 0.0))
                d_north = float(rng.uniform(-args.jitter_lat_m, args.jitter_lat_m))
                d_east = float(rng.uniform(-args.jitter_lon_m, args.jitter_lon_m))
                st["lat"] = lat0 + d_north / 111320.0
                st["lon"] = lon0 + d_east / max(111320.0 * math.cos(math.radians(lat0)), 1.0)
                if "altitude_m" in st:
                    st["altitude_m"] = max(1000.0, float(st["altitude_m"]) + float(rng.uniform(-args.jitter_alt_m, args.jitter_alt_m)))
                if "speed_mps" in st:
                    st["speed_mps"] = max(120.0, float(st["speed_mps"]) + float(rng.uniform(-args.jitter_speed_mps, args.jitter_speed_mps)))
                if "yaw_deg" in st:
                    st["yaw_deg"] = float(st["yaw_deg"]) + float(rng.uniform(-args.jitter_yaw_deg, args.jitter_yaw_deg))
            path = gen_dir / f"{stem}_jitter_seed_{seed}.yaml"
            _dump_yaml(path, patched)
            generated.append({
                "base_config": cfg,
                "config": _rel(path),
                "jitter_seed": seed,
                "jitter_applied": True,
                "generated_config_path": _rel(path),
            })
    return generated


def _episode_outcome(red_alive: int, blue_alive: int) -> str:
    if blue_alive == 0 and red_alive > 0:
        return "red_win"
    if red_alive == 0 and blue_alive > 0:
        return "blue_win"
    return "draw_or_timeout"


def _record_blue_launch(rows: list[dict[str, Any]], context: dict[str, Any], record: dict[str, Any]) -> None:
    if str(record.get("team", "")) != "blue":
        return
    tid = str(record.get("target_id", ""))
    rows.append({
        **context,
        "shooter_id": record.get("shooter_id", ""),
        "target_id": tid,
        "target_role": record.get("target_role", ""),
        "target_is_mav": int(tid == "red_0" or record.get("target_role", "") == "mav"),
        "target_range_m": record.get("range_m", record.get("range_3d_m", "")),
        "target_AO_rad": record.get("AO_rad", record.get("AO_3d_rad", "")),
        "target_TA_rad": record.get("TA_rad", record.get("TA_3d_rad", "")),
        "target_boresight_rad": record.get("boresight_rad", ""),
        "target_track_source": record.get("track_source", ""),
    })


def _crash_window_rows(window: deque[dict[str, Any]], context: dict[str, Any], death_reason: str) -> list[dict[str, Any]]:
    rows = []
    for item in window:
        rows.append({**context, **item, "death_reason": death_reason})
    return rows


def run_scripted_rollouts(config_entries: list[dict[str, Any]], episodes: int, max_steps: int, out_dir: Path) -> dict[str, Any]:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    obs_decisions: list[dict[str, Any]] = []
    oracle_decisions: list[dict[str, Any]] = []
    crash_window_rows: list[dict[str, Any]] = []
    blue_target_rows: list[dict[str, Any]] = []
    blue_first_rows: list[dict[str, Any]] = []
    raw_diag_samples: list[dict[str, Any]] = []
    policy_configs: list[tuple[dict[str, Any], str]] = []
    for entry in config_entries:
        for policy in POLICIES:
            if policy == "red_rule_vs_blue_rule_symmetric_all_attack" and "all_attack" not in Path(entry["base_config"]).name:
                continue
            policy_configs.append((entry, policy))

    for entry, policy in policy_configs:
        config = entry["config"]
        for ep in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
            opponent = OpponentPolicy("brma_rule", seed=1000 + ep + int(entry.get("jitter_seed") or 0) * 100)
            obs, info = env.reset(seed=ep)
            first_launch_side = ""
            first_hit_side = ""
            first_blue_launch_recorded = False
            death_reasons = Counter()
            red_fire = blue_fire = red_hit = blue_hit = 0
            min_alt = {"red": float("inf"), "blue": float("inf")}
            min_speed = {"red": float("inf"), "blue": float("inf")}
            max_speed = {"red": 0.0, "blue": 0.0}
            final_info = info
            episode_step_rows: list[dict[str, Any]] = []
            pre_windows: dict[str, deque[dict[str, Any]]] = {aid: deque(maxlen=50) for aid in getattr(env, "agent_ids", [])}
            context_base = {
                "base_config": entry["base_config"],
                "config": config,
                "policy": policy,
                "episode_id": ep,
                "jitter_seed": entry.get("jitter_seed", ""),
                "jitter_applied": int(bool(entry.get("jitter_applied", False))),
                "generated_config_path": entry.get("generated_config_path", ""),
            }
            steps_executed = 0
            try:
                for step in range(max_steps):
                    context = {**context_base, "step": step}
                    actions = _red_actions(env, obs, policy, {"obs": obs_decisions, "oracle": oracle_decisions}, context)
                    actions.update(_blue_actions(env, obs, policy, opponent))
                    obs, _rewards, terminated, truncated, info = env.step(actions)
                    final_info = info
                    steps_executed = step + 1
                    if len(raw_diag_samples) < 200:
                        raw_diag_samples.append({
                            **context,
                            "launch_diag_json": json.dumps(info.get("__launch_diag__", {}), default=str),
                            "launch_quality_step_json": json.dumps(info.get("__launch_quality_step__", []), default=str),
                            "launch_quality_done_json": json.dumps(info.get("__launch_quality_done__", []), default=str),
                        })
                    step_red_fire = sum(int(info.get(aid, {}).get("missiles_fired_this_step", 0)) for aid in env.red_ids)
                    step_blue_fire = sum(int(info.get(aid, {}).get("missiles_fired_this_step", 0)) for aid in env.blue_ids)
                    red_fire += step_red_fire
                    blue_fire += step_blue_fire
                    if step_red_fire and not first_launch_side:
                        first_launch_side = "red"
                    if step_blue_fire and not first_launch_side:
                        first_launch_side = "blue"
                    for rec in info.get("__launch_quality_step__", []) or []:
                        _record_blue_launch(blue_target_rows, context, rec)
                        if str(rec.get("team", "")) == "blue" and not first_blue_launch_recorded:
                            _record_blue_launch(blue_first_rows, context, rec)
                            first_blue_launch_recorded = True
                    for rec in info.get("__launch_quality_done__", []) or []:
                        side = str(rec.get("team", ""))
                        reason = str(rec.get("raw_termination_reason", ""))
                        if reason == "hit":
                            if side == "red":
                                red_hit += 1
                            elif side == "blue":
                                blue_hit += 1
                            if not first_hit_side:
                                first_hit_side = side
                    for side, ids in [("red", env.red_ids), ("blue", env.blue_ids)]:
                        for aid in ids:
                            st = _agent_state(env, aid)
                            act = actions.get(aid, np.zeros(3, dtype=np.float32))
                            if math.isfinite(st["altitude_m"]):
                                min_alt[side] = min(min_alt[side], st["altitude_m"])
                            if math.isfinite(st["speed_mps"]):
                                min_speed[side] = min(min_speed[side], st["speed_mps"])
                                max_speed[side] = max(max_speed[side], st["speed_mps"])
                            pre_windows[aid].append({
                                "agent_id": aid,
                                "step": step,
                                "altitude_m": st["altitude_m"],
                                "speed_mps": st["speed_mps"],
                                "roll_rad": st["roll_rad"],
                                "pitch_rad": st["pitch_rad"],
                                "yaw_rad": st["yaw_rad"],
                                "vertical_speed_mps": st["vertical_speed_mps"],
                                "action_pitch": float(act[0]) if len(act) else "",
                                "action_heading": float(act[1]) if len(act) > 1 else "",
                                "action_speed": float(act[2]) if len(act) > 2 else "",
                                "nearest_enemy_distance_m": "",
                                "missile_warning": int(bool(info.get(aid, {}).get("missile_warning", False))),
                            })
                    for ev in info.get("death_events", []) or []:
                        side = str(ev.get("side", ""))
                        reason = str(ev.get("death_reason", ""))
                        aid = str(ev.get("agent_id") or ev.get("uid") or ev.get("id") or "")
                        death_reasons[(side, reason)] += 1
                        if side == "red" and reason == "Crash_LowAlt" and aid in pre_windows:
                            crash_window_rows.extend(_crash_window_rows(pre_windows[aid], context_base, reason))
                    row = {
                        **context,
                        "red_alive": _alive_count(env, "red"),
                        "blue_alive": _alive_count(env, "blue"),
                        "mav_alive": int(bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)),
                        "red_fire_count": step_red_fire,
                        "blue_fire_count": step_blue_fire,
                        "red_hit_count_cumulative": red_hit,
                        "blue_hit_count_cumulative": blue_hit,
                        "first_launch_side": first_launch_side,
                        "first_hit_side": first_hit_side,
                        "death_reasons": ";".join(f"{s}:{r}:{n}" for (s, r), n in death_reasons.items()),
                    }
                    row.update(_diag_summary(info, "red"))
                    row.update(_diag_summary(info, "blue"))
                    step_rows.append(row)
                    episode_step_rows.append(row)
                    if all(terminated.values()) or all(truncated.values()):
                        break
            finally:
                env.close()
            last_row = episode_step_rows[-1] if episode_step_rows else {}
            red_alive = int(last_row.get("red_alive", 0) or 0)
            blue_alive = int(last_row.get("blue_alive", 0) or 0)
            episode_rows.append({
                **context_base,
                "episode_length": steps_executed,
                "outcome": _episode_outcome(red_alive, blue_alive),
                "red_alive_final": red_alive,
                "blue_alive_final": blue_alive,
                "mav_alive_final": int(bool(final_info.get("red_0", {}).get("alive", False))),
                "red_fire_count": red_fire,
                "blue_fire_count": blue_fire,
                "red_hit_count": red_hit,
                "blue_hit_count": blue_hit,
                "first_launch_side": first_launch_side,
                "first_hit_side": first_hit_side,
                "red_min_altitude_m": min_alt["red"] if math.isfinite(min_alt["red"]) else "",
                "blue_min_altitude_m": min_alt["blue"] if math.isfinite(min_alt["blue"]) else "",
                "red_speed_min_mps": min_speed["red"] if math.isfinite(min_speed["red"]) else "",
                "red_speed_max_mps": max_speed["red"],
                "blue_speed_min_mps": min_speed["blue"] if math.isfinite(min_speed["blue"]) else "",
                "blue_speed_max_mps": max_speed["blue"],
                "death_reasons": ";".join(f"{s}:{r}:{n}" for (s, r), n in death_reasons.items()),
            })

    roll_dir = out_dir / "scripted_rollouts"
    step_fields = [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "red_alive", "blue_alive", "mav_alive",
        "red_fire_count", "blue_fire_count", "red_hit_count_cumulative", "blue_hit_count_cumulative",
        *[f"red_{k}" for k in LAUNCH_DIAG_FIELDS],
        *[f"blue_{k}" for k in LAUNCH_DIAG_FIELDS],
        "death_reasons", "first_launch_side", "first_hit_side",
    ]
    _write_csv(roll_dir / "scripted_rollout_step_summary_v2.csv", step_rows, step_fields)
    episode_fields = [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "episode_length", "outcome", "red_alive_final",
        "blue_alive_final", "mav_alive_final", "red_fire_count", "blue_fire_count",
        "red_hit_count", "blue_hit_count", "first_launch_side", "first_hit_side",
        "red_min_altitude_m", "blue_min_altitude_m", "red_speed_min_mps",
        "red_speed_max_mps", "blue_speed_min_mps", "blue_speed_max_mps", "death_reasons",
    ]
    _write_csv(roll_dir / "scripted_rollout_episode_summary_v2.csv", episode_rows, episode_fields)
    _write_csv(roll_dir / "launch_diag_raw_sample.csv", raw_diag_samples, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "launch_diag_json", "launch_quality_step_json",
        "launch_quality_done_json",
    ])
    _write_csv(roll_dir / "obs_limited_action_decisions.csv", obs_decisions, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "agent_id", "role", "selected_target_index",
        "target_source", "has_direct_track", "has_mav_shared_track", "enemy_alive",
        "distance_norm_or_raw", "ata_norm_or_raw", "aa_norm_or_raw", "heading_action",
        "pitch_action", "speed_action", "decision_reason",
    ])
    _write_csv(roll_dir / "oracle_launch_window_decisions.csv", oracle_decisions, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "agent_id", "selected_target_id", "range_m",
        "AO_rad", "TA_rad", "boresight_rad", "range_ok", "ao_ok", "ta_ok",
        "boresight_ok", "geometry_ok", "track_ok", "desired_heading_rad",
        "heading_action", "pitch_action", "speed_action", "altitude_m", "decision_reason",
    ])
    _write_csv(roll_dir / "crash_preceding_window.csv", crash_window_rows, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "agent_id", "step", "altitude_m", "speed_mps",
        "roll_rad", "pitch_rad", "yaw_rad", "vertical_speed_mps", "action_pitch",
        "action_heading", "action_speed", "nearest_enemy_distance_m", "missile_warning",
        "death_reason",
    ])
    _write_csv(roll_dir / "blue_target_distribution_v2.csv", blue_target_rows, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "shooter_id", "target_id", "target_role",
        "target_is_mav", "target_range_m", "target_AO_rad", "target_TA_rad",
        "target_boresight_rad", "target_track_source",
    ])
    _write_csv(roll_dir / "blue_first_launch_geometry_v2.csv", blue_first_rows, [
        "base_config", "config", "policy", "episode_id", "jitter_seed", "jitter_applied",
        "generated_config_path", "step", "shooter_id", "target_id", "target_role",
        "target_is_mav", "target_range_m", "target_AO_rad", "target_TA_rad",
        "target_boresight_rad", "target_track_source",
    ])
    write_aggregate_tables_v2(step_rows, episode_rows, roll_dir)
    write_crash_coupling_report(crash_window_rows, roll_dir)
    write_blue_rule_strength_report(blue_target_rows, blue_first_rows, episode_rows, out_dir)
    return {
        "episode_rows": episode_rows,
        "step_rows": step_rows,
        "obs_decisions": obs_decisions,
        "oracle_decisions": oracle_decisions,
    }


def write_aggregate_tables_v2(step_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], out: Path) -> None:
    launch_rows = []
    for side in ("red", "blue"):
        for metric in LAUNCH_DIAG_FIELDS:
            values = [r.get(f"{side}_{metric}", "") for r in step_rows]
            present = [v for v in values if v != ""]
            launch_rows.append({
                "side": side,
                "metric": metric,
                "sum": sum(float(v) for v in present) if present else "",
                "observed_steps": len(present),
                "missing_steps": len(values) - len(present),
            })
    _write_csv(out / "launch_gate_by_side_v2.csv", launch_rows, ["side", "metric", "sum", "observed_steps", "missing_steps"])
    blocked_rows = []
    for side in ("red", "blue"):
        for metric in BLOCKED_FIELDS:
            values = [r.get(f"{side}_{metric}", "") for r in step_rows]
            present = [v for v in values if v != ""]
            blocked_rows.append({
                "side": side,
                "blocked_reason": metric,
                "sum": sum(float(v) for v in present) if present else "",
                "observed_steps": len(present),
                "missing_steps": len(values) - len(present),
            })
    _write_csv(out / "blocked_reason_by_side_v2.csv", blocked_rows, ["side", "blocked_reason", "sum", "observed_steps", "missing_steps"])
    c = Counter()
    for row in episode_rows:
        for item in str(row.get("death_reasons", "")).split(";"):
            if item:
                c[item] += 1
    _write_csv(out / "death_reason_by_side.csv", [{"reason": k, "count": v} for k, v in c.items()], ["reason", "count"])
    _write_csv(out / "first_launch_hit_summary.csv", [
        {
            "policy": p,
            "first_launch_red": sum(1 for r in rows if r["first_launch_side"] == "red"),
            "first_launch_blue": sum(1 for r in rows if r["first_launch_side"] == "blue"),
            "first_hit_red": sum(1 for r in rows if r["first_hit_side"] == "red"),
            "first_hit_blue": sum(1 for r in rows if r["first_hit_side"] == "blue"),
            "episodes": len(rows),
        }
        for p, rows in _group(episode_rows, "policy").items()
    ], ["policy", "first_launch_red", "first_launch_blue", "first_hit_red", "first_hit_blue", "episodes"])
    target_rows = []
    for p, rows in _group(episode_rows, "policy").items():
        target_rows.append({
            "policy": p,
            "episodes": len(rows),
            "red_fire_total": sum(int(float(r.get("red_fire_count") or 0)) for r in rows),
            "red_hit_total": sum(int(float(r.get("red_hit_count") or 0)) for r in rows),
            "blue_fire_total": sum(int(float(r.get("blue_fire_count") or 0)) for r in rows),
            "blue_hit_total": sum(int(float(r.get("blue_hit_count") or 0)) for r in rows),
            "outcomes": json.dumps(dict(Counter(r.get("outcome", "") for r in rows)), sort_keys=True),
        })
    _write_csv(out / "target_distribution_summary.csv", target_rows, [
        "policy", "episodes", "red_fire_total", "red_hit_total", "blue_fire_total",
        "blue_hit_total", "outcomes",
    ])
    _write_csv(out / "crash_altitude_speed_summary.csv", [
        {
            "policy": p,
            "episodes": len(rows),
            "red_min_altitude_m": _mean([r.get("red_min_altitude_m") for r in rows]),
            "blue_min_altitude_m": _mean([r.get("blue_min_altitude_m") for r in rows]),
            "red_speed_min_mps": _mean([r.get("red_speed_min_mps") for r in rows]),
            "blue_speed_min_mps": _mean([r.get("blue_speed_min_mps") for r in rows]),
        }
        for p, rows in _group(episode_rows, "policy").items()
    ], ["policy", "episodes", "red_min_altitude_m", "blue_min_altitude_m", "red_speed_min_mps", "blue_speed_min_mps"])


def write_crash_coupling_report(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Crash Action Coupling Summary", ""]
    if not rows:
        lines.append("- No red Crash_LowAlt preceding windows were recorded in this run.")
    else:
        alt = [_finite_float(r.get("altitude_m")) for r in rows]
        speed = [_finite_float(r.get("speed_mps")) for r in rows]
        pitch = [_finite_float(r.get("action_pitch")) for r in rows]
        lines.extend([
            f"- preceding window rows: {len(rows)}",
            f"- mean altitude m: {_mean(alt)}",
            f"- mean speed mps: {_mean(speed)}",
            f"- mean action_pitch: {_mean(pitch)}",
            "- Interpretation: this is diagnostic only; no action/PID/GCAS changes were made.",
        ])
    _write(out / "crash_action_coupling_summary.md", "\n".join(lines))


def write_blue_rule_strength_report(blue_rows: list[dict[str, Any]], first_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Blue Rule Strength V2", ""]
    if not blue_rows:
        lines.append("- No blue missile launch records were observed in this run.")
    else:
        by_policy = _group(blue_rows, "policy")
        for policy, rows in by_policy.items():
            mav_targets = sum(1 for r in rows if str(r.get("target_is_mav", "")) in {"1", "True", "true"})
            lines.append(f"## {policy}")
            lines.append(f"- blue launches recorded: {len(rows)}")
            lines.append(f"- MAV target launches: {mav_targets}")
            lines.append(f"- first launch steps: {[r.get('step') for r in first_rows if r.get('policy') == policy]}")
            ep = [r for r in episode_rows if r.get("policy") == policy]
            lines.append(f"- outcomes: {dict(Counter(r.get('outcome', '') for r in ep))}")
            lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("- Blue rule strength is conditional on red trajectory and launch geometry; do not label it simply as too strong without the per-policy rows above.")
    _write(out / "blue_rule_strength_v2.md", "\n".join(lines))


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(out)


def _mean(values) -> float | str:
    nums = []
    for value in values:
        v = _finite_float(value)
        if math.isfinite(v):
            nums.append(v)
    return sum(nums) / len(nums) if nums else ""


@dataclass
class MockAircraft:
    uid: str
    pos: np.ndarray
    yaw: float
    alive: bool = True
    num_left_missiles: int = 2

    @property
    def is_alive(self):
        return self.alive

    def get_position(self):
        return self.pos

    def get_velocity(self):
        return np.array([250.0 * math.cos(self.yaw), 250.0 * math.sin(self.yaw), 0.0])

    def get_rpy(self):
        return np.array([0.0, 0.0, self.yaw])

    def get_geodetic(self):
        return np.array([0.0, 0.0, self.pos[2]])


def golden_geometry(out_dir: Path) -> None:
    env = make_env(DEFAULT_CONFIGS[0], env_type="jsbsim_hetero", max_steps=5)
    rows = []
    scenarios = [
        ("red_chases_blue", 0.0, 0.0),
        ("head_on", 0.0, math.pi),
        ("side_aspect", 0.0, math.pi / 2),
        ("blue_chases_red", math.pi, math.pi),
    ]
    try:
        for distance in [5000.0, 10000.0, 15000.0, 20000.0]:
            for name, red_yaw, blue_yaw in scenarios:
                red = MockAircraft("red_mock", np.array([0.0, 0.0, 6000.0]), red_yaw)
                blue = MockAircraft("blue_mock", np.array([distance, 0.0, 6000.0]), blue_yaw)
                for shooter, target, side in [(red, blue, "red_to_blue"), (blue, red, "blue_to_red")]:
                    geom = env._build_launch_geometry_3d(shooter, target)
                    range_ok = env.MISSILE_LAUNCH_MIN_RANGE < geom["range_m"] < env._missile_launch_range_m_effective
                    ao_ok = geom["AO_rad"] < env.MISSILE_LAUNCH_AO_THRESH
                    ta_ok = geom["TA_rad"] > env.MISSILE_LAUNCH_TA_THRESH
                    boresight_ok = (not getattr(env, "use_boresight_launch_gate", False)) or geom.get("boresight_rad", 0.0) < env.MISSILE_LAUNCH_AO_THRESH
                    rows.append({
                        "scenario": name,
                        "side": side,
                        "distance_m": distance,
                        "range_m": geom["range_m"],
                        "AO_rad": geom["AO_rad"],
                        "TA_rad": geom["TA_rad"],
                        "boresight_rad": geom.get("boresight_rad", ""),
                        "range_ok": int(range_ok),
                        "ao_ok": int(ao_ok),
                        "ta_ok": int(ta_ok),
                        "boresight_ok": int(boresight_ok),
                        "track_ok": "not_checked_function_level",
                        "geometry_ok": int(range_ok and ao_ok and ta_ok and boresight_ok),
                    })
    finally:
        env.close()
    _write_csv(out_dir / "golden_geometry_launch_gate.csv", rows, [
        "scenario", "side", "distance_m", "range_m", "AO_rad", "TA_rad",
        "boresight_rad", "range_ok", "ao_ok", "ta_ok", "boresight_ok",
        "track_ok", "geometry_ok",
    ])
    asym = []
    for scenario in scenarios:
        name = scenario[0]
        for distance in [5000.0, 10000.0, 15000.0, 20000.0]:
            sub = [r for r in rows if r["scenario"] == name and r["distance_m"] == distance]
            if len(sub) == 2 and sub[0]["geometry_ok"] != sub[1]["geometry_ok"]:
                asym.append(f"- {name} {distance:.0f}m geometry differs: {sub[0]['side']}={sub[0]['geometry_ok']} {sub[1]['side']}={sub[1]['geometry_ok']}")
    _write(out_dir / "golden_geometry_launch_gate.md", "# Golden Geometry Launch Gate\n\n" + ("\n".join(asym) if asym else "- Function-level geometry is symmetric under mirrored geometry; asymmetry depends on orientation, not team code.\n"))


def static_audits(out_dir: Path, configs: list[str]) -> None:
    geometry_rows = []
    for config in configs:
        cfg = _load_yaml(config)
        states = cfg.get("initial_states", {})
        for aid, st in sorted(states.items()):
            side = "red" if aid.startswith("red") else "blue"
            lat = float(st.get("lat", 0.0))
            lon = float(st.get("lon", 0.0))
            alt = float(st.get("altitude_m", 0.0))
            nearest = ""
            nearest_d: float | str = ""
            for bid, bst in states.items():
                if bid.startswith(side):
                    continue
                dlat = (float(bst.get("lat", 0.0)) - lat) * 111320.0
                dlon = (float(bst.get("lon", 0.0)) - lon) * 111320.0 * max(math.cos(math.radians(lat)), 1e-6)
                dalt = float(bst.get("altitude_m", 0.0)) - alt
                d = math.sqrt(dlat * dlat + dlon * dlon + dalt * dalt)
                if nearest_d == "" or d < nearest_d:
                    nearest, nearest_d = bid, d
            geometry_rows.append({
                "config": config,
                "agent": aid,
                "side": side,
                "lat": st.get("lat", ""),
                "lon": st.get("lon", ""),
                "altitude_m": st.get("altitude_m", ""),
                "speed_mps": st.get("speed_mps", ""),
                "yaw_deg": st.get("yaw_deg", ""),
                "nearest_enemy": nearest,
                "nearest_enemy_distance_m": nearest_d,
                "red_agent_types": cfg.get("red_agent_types", ""),
                "blue_agent_types": cfg.get("blue_agent_types", ""),
                "observation_mode": cfg.get("observation_mode", ""),
                "reward_mode": cfg.get("hetero_reward_mode", ""),
            })
    _write_csv(out_dir / "initial_geometry_audit.csv", geometry_rows, [
        "config", "agent", "side", "lat", "lon", "altitude_m", "speed_mps",
        "yaw_deg", "nearest_enemy", "nearest_enemy_distance_m", "red_agent_types",
        "blue_agent_types", "observation_mode", "reward_mode",
    ])
    _write(out_dir / "initial_geometry_audit.md", "# Initial Geometry Audit\n\n- See CSV for initial geometry and nearest enemy distance.\n- V2 dynamic rollout can generate jittered temporary configs under `generated_configs/`.\n")
    _write(out_dir / "action_mapping_audit.md", "\n".join([
        "# Action Mapping Audit",
        "- Action space is Box [-1,1]^3 for every agent.",
        "- PID path maps act[0] to absolute target pitch, act[1] to absolute heading, act[2] to target velocity.",
        "- This audit does not change action dimensions or semantics.",
    ]))
    _write(out_dir / "pid_gcas_evasion_audit.md", "\n".join([
        "# PID / GCAS / Evasion Audit",
        "- Blue has GCAS safety net when enabled by config; red does not use the same blue GCAS path.",
        "- Red uses scripted missile evasion; blue rule opponent does not use red scripted evasion.",
        "- This audit reports crash coupling only; it does not enable red GCAS or change PID.",
    ]))
    _write(out_dir / "observation_track_gate_audit.md", "\n".join([
        "# Observation / Track Gate Audit",
        "- `mav_shared_geo` exposes `enemy_observed_mask` and `enemy_track_source`.",
        "- V2 adds obs-limited scripted policies that read only actor observation fields.",
        "- If full-state chase succeeds while obs-limited chase fails, observation or representation is more suspicious than missile dynamics.",
    ]))
    _write(out_dir / "launch_gate_static_audit.md", "\n".join([
        "# Launch Gate Static Audit",
        "- Launch gate includes alive, ammo, cooldown, lock delay, track, range, AO, TA, optional boresight and engaged-target deconfliction.",
        "- V2 reads real `__launch_diag__` fields such as `range_ok_pairs`, `ao_ok_pairs`, `ta_ok_pairs`, `geometry_ok_pairs`, track candidates and block counters.",
    ]))
    _write(out_dir / "missile_dynamics_audit.md", "\n".join([
        "# Missile Dynamics Audit",
        "- Missile dynamics are not modified by this audit.",
        "- Current scripted AAM termination reasons should be inspected through `__launch_quality_done__` and rich logs.",
    ]))
    _write(out_dir / "termination_death_crash_audit.md", "\n".join([
        "# Termination / Death / Crash Audit",
        "- V2 records red Crash_LowAlt preceding windows for action/crash coupling.",
        "- Episode final state is computed from episode-local last row, not the global rollout table.",
    ]))
    _write(out_dir / "reward_env_interface_audit.md", "\n".join([
        "# Reward / Environment Interface Audit",
        "- Reward code is not changed by this audit.",
        "- This audit uses launch/death/info diagnostics to decide whether failures are likely environment reachability, observation, reward or algorithm related.",
    ]))
    _write(out_dir / "blue_rule_audit.md", "\n".join([
        "# Blue Rule Audit",
        "- V2 records blue target distribution and first-launch geometry by red scripted policy.",
        "- Blue rule strength must be interpreted conditionally by red trajectory and geometry.",
    ]))


def write_scripted_report(out_dir: Path, results: dict[str, Any], episodes: int) -> None:
    rows = results["episode_rows"]
    by_policy = _group(rows, "policy")
    lines = ["# Scripted Rollout Audit V2", "", f"- Episodes per policy/generated-config requested: {episodes}", ""]
    for policy, items in sorted(by_policy.items()):
        lines.append(f"## {policy}")
        lines.append(f"- episodes: {len(items)}")
        lines.append(f"- red fire total: {sum(int(float(r.get('red_fire_count') or 0)) for r in items)}")
        lines.append(f"- red hit total: {sum(int(float(r.get('red_hit_count') or 0)) for r in items)}")
        lines.append(f"- blue fire total: {sum(int(float(r.get('blue_fire_count') or 0)) for r in items)}")
        lines.append(f"- blue hit total: {sum(int(float(r.get('blue_hit_count') or 0)) for r in items)}")
        lines.append(f"- outcomes: {dict(Counter(r['outcome'] for r in items))}")
        lines.append("")
    _write(out_dir / "scripted_rollouts" / "scripted_rollout_audit.md", "\n".join(lines))


def _metric_sum(step_rows: list[dict[str, Any]], side: str, metric: str) -> float:
    vals = [r.get(f"{side}_{metric}", "") for r in step_rows]
    return sum(float(v) for v in vals if v != "")


def final_report_v2(out_dir: Path, results: dict[str, Any], args: argparse.Namespace) -> None:
    rows = results["episode_rows"]
    step_rows = results["step_rows"]

    def totals(policy: str) -> dict[str, int]:
        sub = [r for r in rows if r["policy"] == policy]
        return {
            "episodes": len(sub),
            "red_fire": sum(int(float(r.get("red_fire_count") or 0)) for r in sub),
            "red_hit": sum(int(float(r.get("red_hit_count") or 0)) for r in sub),
            "blue_fire": sum(int(float(r.get("blue_fire_count") or 0)) for r in sub),
            "blue_hit": sum(int(float(r.get("blue_hit_count") or 0)) for r in sub),
        }

    full = totals("straight_chase_red_vs_blue_rule")
    obs_limited = totals("obs_limited_chase_red_vs_blue_rule_with_mav_shared")
    obs_limited_zero = totals("obs_limited_chase_red_vs_blue_zero")
    oracle = totals("oracle_launch_window_red_vs_blue_rule")
    blue_zero = totals("oracle_launch_window_red_vs_blue_zero")
    red_block = {k: _metric_sum(step_rows, "red", k) for k in BLOCKED_FIELDS}
    blue_block = {k: _metric_sum(step_rows, "blue", k) for k in BLOCKED_FIELDS}
    red_max_block = max(red_block.items(), key=lambda kv: kv[1]) if red_block else ("", 0)
    blue_max_block = max(blue_block.items(), key=lambda kv: kv[1]) if blue_block else ("", 0)
    red_track = _metric_sum(step_rows, "red", "mav_shared_track_candidates")
    direct_track = _metric_sum(step_rows, "red", "direct_track_candidates")
    obs_sources = Counter(str(r.get("target_source", "")) for r in results.get("obs_decisions", []))
    report = [
        "# Environment Rationality Audit V2 Report",
        "",
        "## Run Scope",
        f"- requested episodes: {args.episodes}",
        f"- max_steps: {args.max_steps}",
        f"- jitter: {bool(args.jitter)}",
        f"- jitter_seeds: {args.jitter_seeds if args.jitter else 0}",
        "- No RL training was run.",
        "- No reward, missile dynamics, launch gate, PID, blue rule, action/observation dimension, aircraft XML, engine XML or trainer logic was modified.",
        "",
        "## 1. Full-state chase vs obs-limited chase",
        f"- full-state straight chase: {full}",
        f"- obs-limited chase with MAV shared track: {obs_limited}",
        f"- obs-limited chase against zero-action blue: {obs_limited_zero}",
        f"- obs-limited target-source decisions: {dict(obs_sources)}",
        "- If full-state succeeds while obs-limited fails, prioritize observation schema, MAV shared track usability or policy representation before missile dynamics.",
        "",
        "## 2. True oracle launch-window vs straight chase",
        f"- true oracle launch-window vs blue rule: {oracle}",
        f"- true oracle launch-window vs blue zero: {blue_zero}",
        "- Oracle policy explicitly scores launch-window metrics and uses altitude-safe pitch logic; compare its crash/fire/hit rows with straight chase in CSV.",
        "",
        "## 3. Where RL non-firing is more likely to be blocked",
        f"- red largest blocked counter: {red_max_block[0]}={red_max_block[1]}",
        f"- blue largest blocked counter: {blue_max_block[0]}={blue_max_block[1]}",
        f"- red MAV-shared track candidate sum: {red_track}",
        f"- red direct track candidate sum: {direct_track}",
        "- Use `launch_gate_by_side_v2.csv` and `blocked_reason_by_side_v2.csv`; missing fields are blank, not silently zero-filled.",
        "",
        "## 4. Red/blue launch-gate symmetry",
        "- The gate code is shared, but actual counts differ because red has MAV role blocking and MAV-shared track paths while blue has different observations and rule policy behavior.",
        "- V2 records real `range_ok_pairs`, `ao_ok_pairs`, `ta_ok_pairs`, `boresight_ok_pairs`, `geometry_ok_pairs`, track candidates and block counters.",
        "",
        "## 5. MAV shared track",
        "- MAV shared track participation is measured by `red_mav_shared_track_candidates` and obs-limited action decisions with `target_source=mav_shared`.",
        f"- In this run, red launch diag saw MAV-shared candidates={red_track}, direct candidates={direct_track}.",
        "- If these remain zero while full-state chase succeeds, the shared-track observation path is suspicious for RL learnability.",
        "",
        "## 6. Blue rule target preference",
        "- See `blue_rule_strength_v2.md`, `blue_target_distribution_v2.csv` and `blue_first_launch_geometry_v2.csv`.",
        "- The report separates passive red, level-flight red, obs-limited red, full-state chase red and oracle red policies.",
        "",
        "## 7. Crash and action coupling",
        "- See `crash_preceding_window.csv` and `crash_action_coupling_summary.md`.",
        "- This audit does not enable red GCAS or change action range; it only reports whether low-altitude crash risk persists.",
        "",
        "## 8. All-attack homogeneous interpretation",
        "- The all-attack config is included to isolate homogeneous reward/control sanity. If it still crashes under true oracle, suspect action/PID/GCAS/flight-envelope. If only scripted chase crashes, suspect the scripted chase policy and initial geometry.",
        "- Check `death_reason_by_side.csv` and per-policy outcomes before blaming homogeneous reward alone.",
        "",
        "## 9. Conditional conclusion",
        "- Current environment is not bottom-layer unreachable if full-state or true-oracle rows show missile-launch-hit reachability.",
        "- Whether it is friendly to RL depends on obs-limited chase, launch blocked reasons and crash coupling.",
        "- If obs-limited fails while full-state succeeds, prioritize observation/policy representation.",
        "- If obs-limited succeeds while RL fails, prioritize reward/algorithm.",
        "- If true oracle still crashes frequently, prioritize action/PID/GCAS/flight-envelope diagnostics.",
        "- If blue-zero succeeds while blue-rule fails, prioritize blue-rule pressure diagnostics.",
        "",
        "## 10. Forbidden overstatements",
        "- Do not claim the environment has no problem.",
        "- Do not claim the algorithm is definitely wrong.",
        "- Do not claim reward or blue rule is definitely wrong without the conditional evidence above.",
    ]
    _write(out_dir / "environment_rationality_audit_v2_report.md", "\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--jitter-seeds", type=int, default=5)
    parser.add_argument("--jitter-lat-m", type=float, default=500.0)
    parser.add_argument("--jitter-lon-m", type=float, default=500.0)
    parser.add_argument("--jitter-alt-m", type=float, default=300.0)
    parser.add_argument("--jitter-speed-mps", type=float, default=20.0)
    parser.add_argument("--jitter-yaw-deg", type=float, default=10.0)
    args = parser.parse_args()

    del args.device
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = ROOT / "outputs" / f"environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if out_dir.exists():
        # Refuse to mix v2 data with an existing directory unless it is empty.
        if any(out_dir.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    static_configs = list(dict.fromkeys([*args.configs, *STATIC_CONFIGS]))
    static_audits(out_dir, static_configs)
    _info_schema_probe(args.configs, out_dir)
    golden_geometry(out_dir)
    config_entries = _make_jittered_configs(args.configs, args, out_dir)
    results = run_scripted_rollouts(config_entries, args.episodes, args.max_steps, out_dir)
    write_scripted_report(out_dir, results, args.episodes)
    final_report_v2(out_dir, results, args)
    (out_dir / "manifest.json").write_text(json.dumps({
        "output_dir": str(out_dir),
        "configs": args.configs,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "jitter": bool(args.jitter),
        "jitter_seeds": args.jitter_seeds if args.jitter else 0,
        "policy_count": len(POLICIES),
    }, indent=2), encoding="utf-8")
    try:
        print(out_dir)
    except OSError:
        # Windows/Conda shells can occasionally close stdout after a long run.
        # All audit artifacts are written before this point, so avoid turning a
        # completed read-only audit into a false failure because of final echo.
        pass


if __name__ == "__main__":
    main()
