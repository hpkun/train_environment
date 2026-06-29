"""Environment rationality audit for JSBSim heterogeneous air combat.

This script performs read-only static checks plus short scripted rollouts.  It
does not train models and does not modify environment/reward/missile logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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
    "red_rule_vs_blue_zero",
    "red_rule_vs_blue_rule_symmetric_all_attack",
    "blue_rule_only_strength_probe",
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


def _load_yaml(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _side(agent_id: str) -> str:
    return "red" if agent_id.startswith("red_") else "blue"


def _alive_count(env, side: str) -> int:
    planes = env.red_planes if side == "red" else env.blue_planes
    return sum(1 for sim in planes.values() if sim.is_alive)


def _agent_state(env, aid: str) -> dict[str, float]:
    sim = env._get_sim(aid)
    if sim is None:
        return {"alive": 0, "altitude_m": math.nan, "speed_mps": math.nan}
    vel = np.asarray(sim.get_velocity(), dtype=np.float64)
    return {
        "alive": int(bool(sim.is_alive)),
        "altitude_m": float(sim.get_geodetic()[2]),
        "speed_mps": float(np.linalg.norm(vel)),
    }


def _bearing_norm(src: np.ndarray, dst: np.ndarray) -> float:
    delta = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    heading = math.atan2(float(delta[1]), float(delta[0]))
    return float(np.clip(heading / math.pi, -1.0, 1.0))


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


def _red_actions(env, policy: str) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None or not sim.is_alive:
            actions[rid] = np.zeros(3, dtype=np.float32)
            continue
        role = getattr(env, "agent_roles", {}).get(rid, "")
        if policy in {"zero_action_red_vs_blue_rule"}:
            act = [0.0, 0.0, 0.0]
        elif policy in {"level_flight_red_vs_blue_rule", "blue_rule_only_strength_probe"}:
            act = [0.0, float(sim.get_rpy()[2]) / math.pi, 0.2]
        else:
            if role == "mav":
                act = [0.0, float(sim.get_rpy()[2]) / math.pi, 0.2]
            else:
                _tid, target = _nearest_alive(sim.get_position(), env.blue_planes)
                heading = _bearing_norm(sim.get_position(), target.get_position()) if target is not None else 0.0
                act = [0.0, heading, 0.7]
        actions[rid] = np.asarray(np.clip(act, -1.0, 1.0), dtype=np.float32)
    return actions


def _blue_actions(env, obs: dict, policy: str, opponent: OpponentPolicy) -> dict[str, np.ndarray]:
    if policy == "red_rule_vs_blue_zero":
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
            acts[bid] = np.asarray([0.0, heading, 0.7], dtype=np.float32)
        return acts
    return opponent.act(obs, env.blue_ids, env=env)


def _sum_launch_quality(records: list[dict[str, Any]]) -> Counter:
    c = Counter()
    for rec in records or []:
        side = str(rec.get("team", ""))
        reason = str(rec.get("raw_termination_reason") or rec.get("launch_block_reason") or "")
        if side and reason:
            c[(side, reason)] += 1
    return c


def _diag_summary(info: dict, side: str) -> dict[str, Any]:
    diag = (info.get("__launch_diag__", {}) or {}).get(side, {}) or {}
    blocked = []
    for key, value in diag.items():
        if key.startswith("blocked_") and value:
            blocked.append(f"{key}:{value}")
    return {
        f"{side}_range_ok": diag.get("range_ok", ""),
        f"{side}_ao_ok": diag.get("ao_ok", ""),
        f"{side}_ta_ok": diag.get("ta_ok", ""),
        f"{side}_boresight_ok": diag.get("boresight_ok", ""),
        f"{side}_track_ok": diag.get("track_ok", ""),
        f"{side}_geometry_ok": diag.get("geometry_ok", ""),
        f"{side}_blocked_reasons": ";".join(blocked),
    }


def run_scripted_rollouts(configs: list[str], episodes: int, max_steps: int, out_dir: Path) -> dict[str, Any]:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    policy_configs: list[tuple[str, str]] = []
    for config in configs:
        for policy in POLICIES:
            if policy == "red_rule_vs_blue_rule_symmetric_all_attack" and "all_attack" not in Path(config).name:
                continue
            policy_configs.append((config, policy))

    for config, policy in policy_configs:
        for ep in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
            opponent = OpponentPolicy("brma_rule", seed=1000 + ep)
            obs, info = env.reset(seed=ep)
            first_launch_side = ""
            first_hit_side = ""
            death_reasons = Counter()
            red_fire = blue_fire = red_hit = blue_hit = 0
            min_alt = {"red": float("inf"), "blue": float("inf")}
            min_speed = {"red": float("inf"), "blue": float("inf")}
            max_speed = {"red": 0.0, "blue": 0.0}
            final_info = info
            steps_executed = 0
            try:
                for step in range(max_steps):
                    actions = _red_actions(env, policy)
                    actions.update(_blue_actions(env, obs, policy, opponent))
                    obs, _rewards, terminated, truncated, info = env.step(actions)
                    final_info = info
                    steps_executed = step + 1
                    step_red_fire = sum(int(info.get(aid, {}).get("missiles_fired_this_step", 0)) for aid in env.red_ids)
                    step_blue_fire = sum(int(info.get(aid, {}).get("missiles_fired_this_step", 0)) for aid in env.blue_ids)
                    red_fire += step_red_fire
                    blue_fire += step_blue_fire
                    if step_red_fire and not first_launch_side:
                        first_launch_side = "red"
                    if step_blue_fire and not first_launch_side:
                        first_launch_side = "blue"
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
                    for ev in info.get("death_events", []) or []:
                        death_reasons[(ev.get("side", ""), ev.get("death_reason", ""))] += 1
                    for side, ids in [("red", env.red_ids), ("blue", env.blue_ids)]:
                        for aid in ids:
                            st = _agent_state(env, aid)
                            if math.isfinite(st["altitude_m"]):
                                min_alt[side] = min(min_alt[side], st["altitude_m"])
                            if math.isfinite(st["speed_mps"]):
                                min_speed[side] = min(min_speed[side], st["speed_mps"])
                                max_speed[side] = max(max_speed[side], st["speed_mps"])
                    row = {
                        "config": config,
                        "policy": policy,
                        "episode_id": ep,
                        "step": step,
                        "red_alive": _alive_count(env, "red"),
                        "blue_alive": _alive_count(env, "blue"),
                        "mav_alive": int(bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)),
                        "red_fire_count": step_red_fire,
                        "blue_fire_count": step_blue_fire,
                        "red_hit_count": red_hit,
                        "blue_hit_count": blue_hit,
                        "first_launch_side": first_launch_side,
                        "first_hit_side": first_hit_side,
                        "death_reasons": ";".join(f"{s}:{r}:{n}" for (s, r), n in death_reasons.items()),
                    }
                    row.update(_diag_summary(info, "red"))
                    row.update(_diag_summary(info, "blue"))
                    step_rows.append(row)
                    if all(terminated.values()) or all(truncated.values()):
                        break
            finally:
                env.close()
            red_alive = int(step_rows[-1]["red_alive"]) if step_rows else 0
            blue_alive = int(step_rows[-1]["blue_alive"]) if step_rows else 0
            if blue_alive == 0 and red_alive > 0:
                outcome = "red_win"
            elif red_alive == 0 and blue_alive > 0:
                outcome = "blue_win"
            else:
                outcome = "draw_or_timeout"
            episode_rows.append({
                "config": config,
                "policy": policy,
                "episode_id": ep,
                "episode_length": steps_executed,
                "outcome": outcome,
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
    _write_csv(roll_dir / "scripted_rollout_step_summary.csv", step_rows, [
        "config", "policy", "episode_id", "step", "red_alive", "blue_alive", "mav_alive",
        "red_fire_count", "blue_fire_count", "red_hit_count", "blue_hit_count",
        "red_range_ok", "red_ao_ok", "red_ta_ok", "red_boresight_ok", "red_track_ok",
        "red_geometry_ok", "red_blocked_reasons", "blue_range_ok", "blue_ao_ok",
        "blue_ta_ok", "blue_boresight_ok", "blue_track_ok", "blue_geometry_ok",
        "blue_blocked_reasons", "death_reasons", "first_launch_side", "first_hit_side",
    ])
    _write_csv(roll_dir / "scripted_rollout_episode_summary.csv", episode_rows, [
        "config", "policy", "episode_id", "episode_length", "outcome", "red_alive_final",
        "blue_alive_final", "mav_alive_final", "red_fire_count", "blue_fire_count",
        "red_hit_count", "blue_hit_count", "first_launch_side", "first_hit_side",
        "red_min_altitude_m", "blue_min_altitude_m", "red_speed_min_mps",
        "red_speed_max_mps", "blue_speed_min_mps", "blue_speed_max_mps", "death_reasons",
    ])
    write_aggregate_tables(step_rows, episode_rows, roll_dir)
    return {"episode_rows": episode_rows, "step_rows": step_rows}


def write_aggregate_tables(step_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], out: Path) -> None:
    def side_count_rows(field: str, out_name: str):
        c = Counter()
        for row in step_rows:
            for side in ("red", "blue"):
                raw = str(row.get(f"{side}_{field}", ""))
                for item in raw.split(";"):
                    if item:
                        c[(side, item)] += 1
        _write_csv(out / out_name, [{"side": s, "item": i, "count": n} for (s, i), n in c.items()], ["side", "item", "count"])

    _write_csv(out / "launch_gate_by_side.csv", [
        {"side": side, "metric": metric, "sum": sum(float(r.get(f"{side}_{metric}", 0) or 0) for r in step_rows)}
        for side in ("red", "blue")
        for metric in ("range_ok", "ao_ok", "ta_ok", "boresight_ok", "track_ok", "geometry_ok")
    ], ["side", "metric", "sum"])
    side_count_rows("blocked_reasons", "blocked_reason_by_side.csv")
    c = Counter()
    for row in episode_rows:
        for item in str(row.get("death_reasons", "")).split(";"):
            if item:
                c[item] += 1
    _write_csv(out / "death_reason_by_side.csv", [{"reason": k, "count": v} for k, v in c.items()], ["reason", "count"])
    _write_csv(out / "first_launch_hit_summary.csv", [
        {"policy": p, "first_launch_red": sum(1 for r in rows if r["first_launch_side"] == "red"),
         "first_launch_blue": sum(1 for r in rows if r["first_launch_side"] == "blue"),
         "first_hit_red": sum(1 for r in rows if r["first_hit_side"] == "red"),
         "first_hit_blue": sum(1 for r in rows if r["first_hit_side"] == "blue"),
         "episodes": len(rows)}
        for p, rows in _group(episode_rows, "policy").items()
    ], ["policy", "first_launch_red", "first_launch_blue", "first_hit_red", "first_hit_blue", "episodes"])
    _write_csv(out / "target_distribution_summary.csv", [{"note": "target ids are not consistently exposed in info for all policies; not verified"}], ["note"])
    _write_csv(out / "crash_altitude_speed_summary.csv", [
        {"policy": p, "episodes": len(rows),
         "red_min_altitude_m": _mean([r.get("red_min_altitude_m") for r in rows]),
         "blue_min_altitude_m": _mean([r.get("blue_min_altitude_m") for r in rows]),
         "red_speed_min_mps": _mean([r.get("red_speed_min_mps") for r in rows]),
         "blue_speed_min_mps": _mean([r.get("blue_speed_min_mps") for r in rows])}
        for p, rows in _group(episode_rows, "policy").items()
    ], ["policy", "episodes", "red_min_altitude_m", "blue_min_altitude_m", "red_speed_min_mps", "blue_speed_min_mps"])


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(out)


def _mean(values) -> float | str:
    nums = []
    for value in values:
        try:
            v = float(value)
            if math.isfinite(v):
                nums.append(v)
        except Exception:
            pass
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
                        "scenario": name, "side": side, "distance_m": distance,
                        "range_m": geom["range_m"], "AO_rad": geom["AO_rad"],
                        "TA_rad": geom["TA_rad"], "boresight_rad": geom.get("boresight_rad", ""),
                        "range_ok": int(range_ok), "ao_ok": int(ao_ok), "ta_ok": int(ta_ok),
                        "boresight_ok": int(boresight_ok), "track_ok": "not_checked_function_level",
                        "geometry_ok": int(range_ok and ao_ok and ta_ok and boresight_ok),
                    })
    finally:
        env.close()
    _write_csv(out_dir / "golden_geometry_launch_gate.csv", rows, [
        "scenario", "side", "distance_m", "range_m", "AO_rad", "TA_rad", "boresight_rad",
        "range_ok", "ao_ok", "ta_ok", "boresight_ok", "track_ok", "geometry_ok",
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
    env_py = _read("uav_env/JSBSim/env.py")
    hetero_py = _read("uav_env/JSBSim/envs/hetero_uav_combat_env.py")
    sim_py = _read("uav_env/JSBSim/simulator.py")
    opp_py = _read("algorithms/mappo/opponent_policy.py")
    rich_py = _read("scripts/rich_logging.py")

    _write(out_dir / "action_mapping_audit.md", "\n".join([
        "# Action Mapping Audit",
        "- Action space is Box [-1,1]^3 for every agent.",
        "- PID path maps act[0] to absolute target pitch +/-90 deg, act[1] to absolute target heading +/-180 deg, act[2] to velocity 102-408 m/s.",
        "- Direct-FCS path maps act[0]/act[1] to elevator/aileron command and act[2] to throttle 0.4-0.9.",
        "- Red and blue share Layer 3 action mapping; team asymmetries occur before action mapping: red missile evasion, blue GCAS.",
        "- Risk: +/-90 deg target pitch is aggressive and can expose red agents without GCAS to low-altitude or over-control crashes.",
    ]))
    _write(out_dir / "pid_gcas_evasion_audit.md", "\n".join([
        "# PID / GCAS / Evasion Audit",
        "- Blue has GCAS safety net when `enable_gcas_for_blue=true`; red does not.",
        "- Red has scripted missile evasion; blue does not.",
        "- These asymmetries are explicit in `_parse_actions()` and can affect crash rates and survivability.",
        "- Current main F16-dynamics/F22-visual configs use F16 dynamics for MAV and F22 only as ACMI visual label.",
        "- F16-dynamics MAV surrogate means this audit cannot be interpreted as true F22 flight-dynamics rationality.",
    ]))

    geometry_rows = []
    for config in configs:
        cfg = _load_yaml(config)
        states = cfg.get("initial_states", {})
        ids = sorted(states)
        for aid in ids:
            st = states.get(aid, {})
            side = "red" if aid.startswith("red") else "blue"
            nearest = ""
            nearest_d = ""
            lon = float(st.get("lon", 0.0)); lat = float(st.get("lat", 0.0)); alt = float(st.get("altitude_m", 0.0))
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
                "config": config, "agent": aid, "side": side, "lat": st.get("lat", ""),
                "lon": st.get("lon", ""), "altitude_m": st.get("altitude_m", ""),
                "speed_mps": st.get("speed_mps", ""), "yaw_deg": st.get("yaw_deg", ""),
                "nearest_enemy": nearest, "nearest_enemy_distance_m": nearest_d,
                "red_agent_types": cfg.get("red_agent_types", ""), "blue_agent_types": cfg.get("blue_agent_types", ""),
                "observation_mode": cfg.get("observation_mode", ""), "reward_mode": cfg.get("hetero_reward_mode", ""),
            })
    _write_csv(out_dir / "initial_geometry_audit.csv", geometry_rows, [
        "config", "agent", "side", "lat", "lon", "altitude_m", "speed_mps", "yaw_deg",
        "nearest_enemy", "nearest_enemy_distance_m", "red_agent_types", "blue_agent_types",
        "observation_mode", "reward_mode",
    ])
    _write(out_dir / "initial_geometry_audit.md", "# Initial Geometry Audit\n\n- See CSV for per-agent initial lat/lon/alt/speed/yaw and nearest enemy distances.\n- Risk to inspect: red_0 MAV in MAV configs starts behind/offset, but blue target selection may still choose it if nearest/visible geometry makes it attractive.\n- 3v2 is numerically asymmetric by design: red has 1 MAV + 2 attack UAV vs blue 2 attack UAV; terminal formulas using alive counts can encode team-size bias depending on reward mode.\n")
    _write(out_dir / "observation_track_gate_audit.md", "\n".join([
        "# Observation / Track Gate Audit",
        "- `brma_sensor` provides classic entity states; `mav_shared_geo` adds geo states, observed masks and `enemy_track_source`.",
        "- Red MAV role is blocked from launch by `_has_launch_track()` returning `role_blocked_mav`.",
        "- Red UAV can use direct track or `mav_shared` track if obs cache is populated.",
        "- Blue launch track checks `enemy_track_source` / `enemy_observed_mask`, then falls back to legacy visible enemy states.",
        "- Potential asymmetry: blue fallback/direct track can be broader than red MAV-shared track path; this must be interpreted with rollout blocked-reason tables.",
    ]))
    _write(out_dir / "launch_gate_static_audit.md", "\n".join([
        "# Launch Gate Static Audit",
        "- `_check_missile_launch()` checks alive, ammo, cooldown, lock delay, track, range, AO, TA, optional boresight, engaged target and target selection.",
        "- BRMA-style launch gate remains unchanged by this audit.",
        "- Red-specific restrictions include MAV role launch block and MAV/shared track dependency.",
        "- Blue uses same geometry gate but may differ in observation fallback and scripted policy behavior.",
        "- Red target selection can be `closest` or `mav_threat_rank`; blue rule target selection is in parent `rule_based_agent.py` via `OpponentPolicy(brma_rule)`.",
    ]))
    _write(out_dir / "missile_dynamics_audit.md", "\n".join([
        "# Missile Dynamics Audit",
        "- Current MissileSimulator is scripted close-range AAM with fixed `missile_speed_mps=600`, `t_max=60`, `K=3`, hit radius `Rc=300`, arm time 0.15s.",
        "- Legal termination reasons in current code are hit, p_hit_fail, timeout, target_dead, unknown fallback.",
        "- Low-speed and overshoot missile terminations are not expected in the current scripted AAM path.",
        "- Red and blue missiles share the same simulator class; asymmetry is more likely from launch geometry, track, target selection or shooter flight state.",
    ]))
    _write(out_dir / "termination_death_crash_audit.md", "\n".join([
        "# Termination / Death / Crash Audit",
        "- Episode terminates when all blue or all red aircraft are dead; truncates at max_steps.",
        "- Crash/death reasons include missile hit, low altitude, over-G/extreme/non-finite states depending on env checks.",
        "- Red lacks blue GCAS, so red low-altitude crash risk can be higher under aggressive actions.",
        "- Death reasons are exposed through `info[aid]['death_reason']` and `info['death_events']`.",
    ]))
    _write(out_dir / "reward_env_interface_audit.md", "\n".join([
        "# Reward / Environment Interface Audit",
        "- `_step_kill_count` is step-local kill accounting used by reward overlays.",
        "- `missiles_fired_this_step` in info is step-local and reset after info generation.",
        "- `__missile_term__` is accumulated termination counters by side.",
        "- `reward_components` are merged into per-agent info for diagnostics.",
        "- Rich logs are sufficient only when enabled; missing reward/missile/aircraft timeseries prevents post-hoc causality.",
        "- `brma_paper_homogeneous_v1` is a diagnostic homogeneous baseline, not a claim of full original-paper reproduction.",
        "- 3v2 `30*(Nred-Nblue)` terminal can encode initial team-size bias unless applied only at episode end and interpreted carefully.",
    ]))
    _write(out_dir / "blue_rule_audit.md", "\n".join([
        "# Blue Rule Audit",
        "- `OpponentPolicy(brma_rule)` delegates to parent `rule_based_agent.blue_coordinated_actions`.",
        "- Local wrapper passes blue observations, num_blue/num_red, engaged targets, own positions and own headings.",
        "- The local wrapper does not pass red roles directly, but observations may encode geometry that makes red_0 a target.",
        "- Blue has GCAS in the environment while red does not; blue trajectory stability can therefore be higher.",
        "- Whether blue is too strong must be judged from scripted rollout first-launch/hit and blocked-reason summaries, not assumed statically.",
    ]))


def write_scripted_report(out_dir: Path, results: dict[str, Any], episodes: int) -> None:
    rows = results["episode_rows"]
    by_policy = _group(rows, "policy")
    lines = ["# Scripted Rollout Audit", "", f"- Episodes per policy/config requested: {episodes}", ""]
    for policy, items in by_policy.items():
        lines.append(f"## {policy}")
        lines.append(f"- episodes: {len(items)}")
        lines.append(f"- red fire mean: {_mean([r['red_fire_count'] for r in items])}")
        lines.append(f"- red hit mean: {_mean([r['red_hit_count'] for r in items])}")
        lines.append(f"- blue fire mean: {_mean([r['blue_fire_count'] for r in items])}")
        lines.append(f"- blue hit mean: {_mean([r['blue_hit_count'] for r in items])}")
        lines.append(f"- outcomes: {dict(Counter(r['outcome'] for r in items))}")
        lines.append("")
    _write(out_dir / "scripted_rollouts" / "scripted_rollout_audit.md", "\n".join(lines))


def final_report(out_dir: Path, results: dict[str, Any], configs: list[str]) -> None:
    rows = results["episode_rows"]
    oracle = [r for r in rows if r["policy"] == "oracle_geometry_red_vs_blue_rule"]
    blue_zero = [r for r in rows if r["policy"] == "red_rule_vs_blue_zero"]
    zero = [r for r in rows if r["policy"] == "zero_action_red_vs_blue_rule"]
    oracle_fire = sum(int(r["red_fire_count"]) for r in oracle)
    oracle_hit = sum(int(r["red_hit_count"]) for r in oracle)
    blue_zero_fire = sum(int(r["red_fire_count"]) for r in blue_zero)
    zero_blue_hit = sum(int(r["blue_hit_count"]) for r in zero)
    decision = []
    if oracle_fire > 0 or oracle_hit > 0:
        decision.append("Oracle/chase-style red can reach at least some launch/hit events; if RL still fails, reward/algorithm or policy representation remains suspicious.")
    else:
        decision.append("Oracle red did not reliably launch/hit in this audit; environment/blue rule/launch gate remain suspicious.")
    if blue_zero_fire > oracle_fire:
        decision.append("Red improves when blue is disabled/weak; blue rule strength or initial geometry may suppress early learning.")
    if zero_blue_hit > 0:
        decision.append("Blue can hit passive/weak red in scripted rollouts; blue rule strength is non-trivial.")
    if not decision:
        decision.append("Current evidence is insufficient for attribution; run the next minimum experiment.")
    text = "\n".join([
        "# Environment Rationality Audit Report",
        "",
        "## 1. Executive Summary",
        "- This audit performed static code/config review plus no-training scripted rollouts.",
        "- Current environment has plausible learning signal paths, but important asymmetries exist: blue GCAS, red-only missile evasion, MAV launch block, track-gate differences, and blue scripted rule strength.",
        "- Recommendation: do not continue long training until scripted/oracle reachability and blue-rule strength are inspected in the generated CSVs.",
        "",
        "## 2. Environment Mechanism Findings",
        "- Action mapping: `[-1,1]^3` to pitch/heading/speed; pitch target range is aggressive at +/-90 deg.",
        "- PID/GCAS/evasion: blue has GCAS, red has missile evasion, red lacks GCAS.",
        "- Dynamics: main F16-dynamics/F22-visual configs do not audit true F22 dynamics.",
        "- Initial geometry, observation, launch gate, missile and termination details are in companion audit files.",
        "",
        "## 3. Red/Blue Symmetry Findings",
        "- Flight action mapping is mostly symmetric at layer 3.",
        "- Asymmetries: blue GCAS, red-only evasion, MAV role launch block, red target selection option, blue rule policy implementation.",
        "- Some asymmetries are intentional experimental design; others can affect early learning and should be ablated diagnostically.",
        "",
        "## 4. Blue Rule Strength Findings",
        "- See `scripted_rollouts/first_launch_hit_summary.csv` and `blue_rule_audit.md`.",
        f"- In passive-red rollout, blue hit count total observed: {zero_blue_hit}.",
        "",
        "## 5. Scripted/Oracle Reachability Findings",
        f"- Oracle red total fire={oracle_fire}, hit={oracle_hit}.",
        f"- Red-vs-blue-zero total fire={blue_zero_fire}.",
        "- See scripted rollout CSVs for per-policy outcomes.",
        "",
        "## 6. Crash and Flight-Envelope Findings",
        "- See crash_altitude_speed_summary.csv and termination_death_crash_audit.md.",
        "- If red low-altitude crashes dominate while blue does not, action/PID/GCAS asymmetry is a primary suspect.",
        "",
        "## 7. Reward Interface Findings",
        "- Reward signals depend on step-local kill/fire counts and info diagnostics. Rich logging must be enabled for causal post-hoc analysis.",
        "- brma_paper_homogeneous_v1 should be treated only as a diagnostic homogeneous baseline.",
        "",
        "## 8. Decision Tree Conclusion",
        *[f"- {d}" for d in decision],
        "",
        "## 9. Concrete Next Steps",
        "1. If oracle red cannot launch/hit, inspect launch/track gate and blue pressure before reward changes.",
        "2. If blue-zero enables red launch but blue-rule suppresses it, run a blue-strength curriculum/ablation diagnostic only.",
        "3. If scripted red works but RL fails, then inspect reward/algorithm with rich logs before long training.",
    ])
    _write(out_dir / "environment_rationality_audit_report.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    del args.device
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = ROOT / "outputs" / f"environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    static_configs = list(dict.fromkeys([*args.configs, *STATIC_CONFIGS]))
    static_audits(out_dir, static_configs)
    golden_geometry(out_dir)
    results = run_scripted_rollouts(args.configs, args.episodes, args.max_steps, out_dir)
    write_scripted_report(out_dir, results, args.episodes)
    final_report(out_dir, results, args.configs)
    (out_dir / "manifest.json").write_text(json.dumps({
        "output_dir": str(out_dir),
        "configs": args.configs,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
    }, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
