"""Utilities for red attack-chain diagnostics.

These helpers intentionally do not modify environment mechanics.  They only
construct short rollouts, read public diagnostics from ``info`` and inspect
runtime aircraft state.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"
)


def write_json(path: str | Path, data: Any) -> Path:
    out = _rel(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def write_md(path: str | Path, lines: list[str]) -> Path:
    out = _rel(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_env(config: str = DEFAULT_CONFIG, **overrides):
    from uav_env import make_env as _make_env

    return _make_env(config, env_type="jsbsim_hetero", **overrides)


def zero_actions(env) -> dict[str, np.ndarray]:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def safe_mav_action() -> np.ndarray:
    return np.array([0.0, 0.0, 0.35], dtype=np.float32)


def nearest_enemy(env, aid: str):
    sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
    if sim is None or not sim.is_alive:
        return None, float("inf")
    enemies = env.blue_planes if aid.startswith("red_") else env.red_planes
    best = None
    best_dist = float("inf")
    pos = np.asarray(sim.get_position(), dtype=np.float64)
    for enemy in enemies.values():
        if not enemy.is_alive:
            continue
        dist = float(np.linalg.norm(np.asarray(enemy.get_position(), dtype=np.float64) - pos))
        if dist < best_dist:
            best = enemy
            best_dist = dist
    return best, best_dist


def direct_chase_action(env, aid: str, speed: float = 0.8) -> np.ndarray:
    sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
    target, _dist = nearest_enemy(env, aid)
    if sim is None or target is None:
        return np.array([0.0, 0.0, 0.3], dtype=np.float32)
    spos = np.asarray(sim.get_position(), dtype=np.float64)
    tpos = np.asarray(target.get_position(), dtype=np.float64)
    dn = float(tpos[0] - spos[0])
    de = float(tpos[1] - spos[1])
    du = float(tpos[2] - spos[2])
    heading = math.atan2(de, dn) / math.pi
    pitch = np.clip(du / 5000.0, -0.25, 0.25)
    return np.array([pitch, np.clip(heading, -1.0, 1.0), speed], dtype=np.float32)


def obs_rule_attack_action(obs: dict, speed: float = 0.8) -> np.ndarray:
    enemies = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
    if enemies.ndim != 2 or enemies.shape[0] == 0:
        return np.array([0.0, 0.0, 0.3], dtype=np.float32)
    best = None
    best_dist = float("inf")
    for state in enemies:
        if state.size < 6 or np.allclose(state, 0.0):
            continue
        dist = abs(float(state[5]))
        if dist <= 0:
            dist = float(np.linalg.norm(state[:3]))
        if dist < best_dist:
            best = state
            best_dist = dist
    if best is None:
        return np.array([0.0, 0.0, 0.3], dtype=np.float32)
    pitch = float(best[2]) * 2.0
    heading = float(best[1]) * 2.0
    return np.clip(np.array([pitch, heading, speed], dtype=np.float32), -1.0, 1.0)


def red_oracle_actions(env, obs: dict, mode: str) -> dict[str, np.ndarray]:
    actions = {}
    for rid in env.red_ids:
        if env.agent_roles.get(rid) == "mav":
            actions[rid] = safe_mav_action()
        elif mode == "direct_chase":
            actions[rid] = direct_chase_action(env, rid)
        else:
            actions[rid] = obs_rule_attack_action(obs.get(rid, {}))
    return actions


def blue_actions(env, obs: dict, mode: str) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {bid: np.zeros(3, dtype=np.float32) for bid in env.blue_ids}
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy = OpponentPolicy(mode="brma_rule", seed=0)
    return policy.act(obs, env.blue_ids, env=env)


def team_done(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(bool(v) for v in truncated.values())


def alive_counts(env) -> dict[str, float]:
    return {
        "red_alive": float(sum(s.is_alive for s in env.red_planes.values())),
        "blue_alive": float(sum(s.is_alive for s in env.blue_planes.values())),
        "red_dead": float(sum(not s.is_alive for s in env.red_planes.values())),
        "blue_dead": float(sum(not s.is_alive for s in env.blue_planes.values())),
    }


def collect_step_counts(info: dict) -> dict[str, int]:
    red_fired = blue_fired = 0
    for aid, agent_info in info.items():
        if not isinstance(agent_info, dict):
            continue
        fired = int(agent_info.get("missiles_fired_this_step", 0) or 0)
        if aid.startswith("red_"):
            red_fired += fired
        elif aid.startswith("blue_"):
            blue_fired += fired
    mt = info.get("__missile_term__", {})
    red_hits = int(mt.get("red", {}).get("hit", 0) or 0) if isinstance(mt, dict) else 0
    blue_hits = int(mt.get("blue", {}).get("hit", 0) or 0) if isinstance(mt, dict) else 0
    return {
        "red_fired": red_fired,
        "blue_fired": blue_fired,
        "red_hits_total": red_hits,
        "blue_hits_total": blue_hits,
    }


def geometry(env, aid: str, target_id: str | None = None) -> dict[str, float | bool | str | None]:
    sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
    if sim is None or not sim.is_alive:
        return {"agent_id": aid, "alive": False}
    if target_id:
        target = env.red_planes.get(target_id) or env.blue_planes.get(target_id)
        dist = float("inf")
    else:
        target, dist = nearest_enemy(env, aid)
    if target is None or not target.is_alive:
        return {"agent_id": aid, "alive": True, "target_id": None}
    from uav_env.JSBSim.utils import get2d_AO_TA_R

    spos = np.asarray(sim.get_position(), dtype=np.float64)
    svel = np.asarray(sim.get_velocity(), dtype=np.float64)
    tpos = np.asarray(target.get_position(), dtype=np.float64)
    tvel = np.asarray(target.get_velocity(), dtype=np.float64)
    sfeat = np.array([spos[0], spos[1], -spos[2], svel[0], svel[1], -svel[2]], dtype=np.float64)
    tfeat = np.array([tpos[0], tpos[1], -tpos[2], tvel[0], tvel[1], -tvel[2]], dtype=np.float64)
    ao, ta, rng = get2d_AO_TA_R(sfeat, tfeat)
    if not np.isfinite(dist):
        dist = float(rng)
    return {
        "agent_id": aid,
        "alive": True,
        "target_id": target.uid,
        "distance_m": float(rng),
        "ata_rad": float(ao),
        "aspect_rad": float(ta),
        "altitude_diff_m": float(tpos[2] - spos[2]),
        "speed_mps": float(np.linalg.norm(svel)),
        "missile_count": int(getattr(sim, "num_left_missiles", 0)),
        "cooldown": int(getattr(env, "_missile_cooldown", {}).get(aid, 0)),
        "launch_condition_distance": bool(
            env.MISSILE_LAUNCH_MIN_RANGE < rng < env.MISSILE_LAUNCH_RANGE_THRESH
        ),
        "launch_condition_angle": bool(
            ao < env.MISSILE_LAUNCH_AO_THRESH and ta > env.MISSILE_LAUNCH_TA_THRESH
        ),
        "launch_condition_all": bool(
            env.MISSILE_LAUNCH_MIN_RANGE < rng < env.MISSILE_LAUNCH_RANGE_THRESH
            and ao < env.MISSILE_LAUNCH_AO_THRESH
            and ta > env.MISSILE_LAUNCH_TA_THRESH
            and getattr(sim, "num_left_missiles", 0) > 0
        ),
    }


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }

