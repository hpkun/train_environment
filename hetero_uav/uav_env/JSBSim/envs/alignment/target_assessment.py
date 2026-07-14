"""Shared TAM-HAPPO target-assessment helpers.

The weights and one-step assessment terms are copied from the existing v5
reward implementation so reward target selection and launch target selection
can use one contract.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


TARGET_WEIGHTS = {"angle": 0.35, "distance": 0.25, "height": 0.20, "speed": 0.20}


def safe_state_vector(sim, getter: str) -> np.ndarray:
    try:
        value = np.asarray(getattr(sim, getter)(), dtype=np.float64).reshape(-1)
    except Exception:
        value = np.zeros(3, dtype=np.float64)
    if value.size < 3:
        value = np.pad(value, (0, 3 - value.size))
    value = value[:3]
    return value if np.isfinite(value).all() else np.zeros(3, dtype=np.float64)


def paper_target_score(env, attacker, target, cfg: dict) -> dict[str, float]:
    """TAM-HAPPO Eq. 11-12 target-assessment score used by v5."""
    geom = env._brma_tam_3d_geometry(attacker, target)
    ata = float(geom["tam_ata_rad"])
    aa = float(geom["tam_aa_rad"])
    distance = float(geom["target_distance_m"])
    own_pos = safe_state_vector(attacker, "get_position")
    target_pos = safe_state_vector(target, "get_position")
    own_vel = safe_state_vector(attacker, "get_velocity")
    target_vel = safe_state_vector(target, "get_velocity")
    assessment = cfg.get("target_assessment", {})
    hmax = float(assessment.get("relative_altitude_norm_m", 10000.0))
    vmax = float(assessment.get("relative_speed_norm_mps", 408.0))
    dmax = float(assessment.get("engagement_range_m", 14000.0))
    e_angle = 1.0 - (ata + aa) / (2.0 * math.pi)
    e_distance = 1.0 if distance <= dmax else 0.0
    e_height = float((own_pos[2] - target_pos[2]) / max(hmax, 1e-8))
    e_speed = float(np.linalg.norm(own_vel - target_vel) / max(vmax, 1e-8))
    score = (TARGET_WEIGHTS["angle"] * e_angle
             + TARGET_WEIGHTS["distance"] * e_distance
             + TARGET_WEIGHTS["height"] * e_height
             + TARGET_WEIGHTS["speed"] * e_speed)
    return {
        "score": float(score), "ata": ata, "aa": aa, "distance_m": distance,
        "target_angle_assessment": float(e_angle),
        "target_distance_assessment": float(e_distance),
        "target_height_assessment": float(e_height),
        "target_speed_assessment": float(e_speed),
    }


def select_paper_assessment_target(
    env,
    attacker,
    cfg: dict,
    *,
    hold_steps: int = 0,
    require_observed: bool = False,
) -> tuple[str | None, Any | None, dict[str, Any]]:
    """Select or hold one paper-assessment target for an attack UAV."""
    aid = str(getattr(attacker, "uid", ""))
    state = getattr(env, "_engagement_target_state", {})
    previous = state.get(aid, {}) if isinstance(state, dict) else {}
    previous_id = str(previous.get("target_id", "") or "")
    previous_step = int(previous.get("selected_step", -10**9))
    previous_switches = int(previous.get("switch_count", 0))

    def valid_target(bid: str):
        target = getattr(env, "blue_planes", {}).get(bid)
        if target is None or not getattr(target, "is_alive", False):
            return None
        if require_observed and not env._has_launch_track(aid, bid)[0]:
            return None
        return target

    held = valid_target(previous_id)
    if held is not None and int(getattr(env, "current_step", 0)) - previous_step < int(hold_steps):
        values = paper_target_score(env, attacker, held, cfg)
        values.update({"target_id": previous_id, "target_rank": 1.0,
                       "target_held": 1.0, "target_switched": 0.0})
        step_diag = getattr(env, "_engagement_target_step_diag", None)
        if isinstance(step_diag, dict):
            step_diag.setdefault(aid, {"held": False, "switched": False})["held"] = True
        return previous_id, held, values

    candidates = []
    own_pos = safe_state_vector(attacker, "get_position")
    for bid in getattr(env, "blue_ids", []):
        blue = valid_target(bid)
        if blue is None:
            continue
        values = paper_target_score(env, attacker, blue, cfg)
        values["target_id"] = bid
        values["nearest_distance"] = float(
            np.linalg.norm(safe_state_vector(blue, "get_position") - own_pos))
        candidates.append(values)
    if not candidates:
        if isinstance(state, dict):
            state.pop(aid, None)
        return None, None, {}
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["target_id"])))
    selected = dict(candidates[0])
    bid = str(selected["target_id"])
    selected.update({
        "target_rank": 1.0,
        "closest_target_id": min(candidates, key=lambda row: row["nearest_distance"])["target_id"],
        "target_held": float(bid == previous_id),
        "target_switched": float(bool(previous_id and bid != previous_id)),
    })
    if isinstance(state, dict):
        state[aid] = {
            "target_id": bid,
            "selected_step": int(getattr(env, "current_step", 0)),
            "switch_count": previous_switches + int(bool(previous_id and bid != previous_id)),
        }
    step_diag = getattr(env, "_engagement_target_step_diag", None)
    if isinstance(step_diag, dict):
        diag = step_diag.setdefault(aid, {"held": False, "switched": False})
        diag["held"] = bool(diag["held"] or bid == previous_id)
        diag["switched"] = bool(diag["switched"] or (previous_id and bid != previous_id))
    return bid, getattr(env, "blue_planes", {})[bid], selected
