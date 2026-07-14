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


def target_reserved_by_other(env, shooter_id: str, target_id: str) -> bool:
    """Return whether an engaged target is unavailable to this shooter."""
    if target_id not in getattr(env, "_engaged_targets", set()):
        return False
    own_missile_targets_target = False
    for missile in getattr(env, "_missiles_in_flight", {}).values():
        missile_target = str(getattr(missile, "_target_id", "") or getattr(
            getattr(missile, "target_aircraft", None), "uid", "") or "")
        missile_shooter = str(getattr(getattr(missile, "parent_aircraft", None), "uid", "")
                              or getattr(missile, "_parent_id", "") or "")
        if missile_target != target_id:
            continue
        if missile_shooter != shooter_id:
            return True
        own_missile_targets_target = True
    if str(getattr(env, "_lock_target", {}).get(shooter_id, "") or "") == target_id:
        return False
    if own_missile_targets_target:
        return False
    return True


def safe_state_vector(sim, getter: str) -> np.ndarray:
    try:
        value = np.asarray(getattr(sim, getter)(), dtype=np.float64).reshape(-1)
    except Exception:
        value = np.zeros(3, dtype=np.float64)
    if value.size < 3:
        value = np.pad(value, (0, 3 - value.size))
    value = value[:3]
    return value if np.isfinite(value).all() else np.zeros(3, dtype=np.float64)


def target_hold_sequence_stats(target_ids) -> tuple[int, int, float]:
    """Return valid decisions, contiguous target segments, and mean segment length."""
    valid = 0
    segments = 0
    previous = ""
    for target_id in target_ids:
        current = str(target_id or "")
        if not current:
            previous = ""
            continue
        valid += 1
        if current != previous:
            segments += 1
        previous = current
    return valid, segments, float(valid / segments) if segments else 0.0


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
    candidate_target_ids: set[str] | None = None,
) -> tuple[str | None, Any | None, dict[str, Any]]:
    """Select or hold one paper-assessment target for an attack UAV."""
    aid = str(getattr(attacker, "uid", ""))
    state = getattr(env, "_engagement_target_state", {})
    previous = state.get(aid, {}) if isinstance(state, dict) else {}
    previous_id = str(previous.get("target_id", "") or "")
    previous_step = int(previous.get("selected_step", -10**9))
    previous_switches = int(previous.get("switch_count", 0))
    reallocation_counts = getattr(env, "_engagement_reallocation_counts", None)
    previous_reallocations = int(
        reallocation_counts.get(aid, 0) if isinstance(reallocation_counts, dict)
        else previous.get("reallocation_count", 0))
    allowed_ids = None if candidate_target_ids is None else {
        str(target_id) for target_id in candidate_target_ids}

    def engaged_by_other(bid: str) -> bool:
        return target_reserved_by_other(env, aid, bid)

    def invalid_reason(bid: str) -> str:
        target = getattr(env, "blue_planes", {}).get(bid)
        if target is None:
            return "missing"
        if not getattr(target, "is_alive", False):
            return "dead"
        if require_observed and not env._has_launch_track(aid, bid)[0]:
            return "unobserved"
        if engaged_by_other(bid):
            return "engaged_by_other"
        if allowed_ids is not None and bid not in allowed_ids:
            return "not_fire_control_candidate"
        return ""

    def valid_target(bid: str):
        if invalid_reason(bid):
            return None
        return getattr(env, "blue_planes", {}).get(bid)

    held = valid_target(previous_id)
    current_step = int(getattr(env, "current_step", 0))
    if held is not None and (
            current_step == previous_step
            or current_step - previous_step < int(hold_steps)):
        values = paper_target_score(env, attacker, held, cfg)
        own_pos = safe_state_vector(attacker, "get_position")
        valid_ids = [
            bid for bid in getattr(env, "blue_ids", []) if valid_target(bid) is not None]
        closest_id = min(
            valid_ids,
            key=lambda bid: float(np.linalg.norm(
                safe_state_vector(getattr(env, "blue_planes", {})[bid], "get_position")
                - own_pos)),
        ) if valid_ids else previous_id
        values.update({"target_id": previous_id, "target_rank": 1.0,
                       "closest_target_id": closest_id,
                       "target_held": 1.0, "target_switched": 0.0})
        step_diag = getattr(env, "_engagement_target_step_diag", None)
        if isinstance(step_diag, dict):
            step_diag.setdefault(
                aid, {"held": False, "switched": False, "reallocated": False,
                      "reallocation_reason": ""})["held"] = True
        return previous_id, held, values

    reallocation_reason = invalid_reason(previous_id) if previous_id else ""
    reallocated = bool(previous_id and reallocation_reason)
    if reallocated and isinstance(reallocation_counts, dict):
        reallocation_counts[aid] = previous_reallocations + 1

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
        step_diag = getattr(env, "_engagement_target_step_diag", None)
        if isinstance(step_diag, dict) and reallocated:
            diag = step_diag.setdefault(
                aid, {"held": False, "switched": False, "reallocated": False,
                      "reallocation_reason": ""})
            diag["reallocated"] = True
            diag["reallocation_reason"] = reallocation_reason
        return None, None, {}
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["target_id"])))
    selected = dict(candidates[0])
    bid = str(selected["target_id"])
    selected.update({
        "target_rank": 1.0,
        "closest_target_id": min(candidates, key=lambda row: row["nearest_distance"])["target_id"],
        "target_held": float(bid == previous_id),
        "target_switched": float(bool(previous_id and bid != previous_id)),
        "target_reallocated": float(reallocated),
        "target_reallocation_reason": reallocation_reason,
    })
    if isinstance(state, dict):
        state[aid] = {
            "target_id": bid,
            "selected_step": int(getattr(env, "current_step", 0)),
            "switch_count": previous_switches + int(bool(previous_id and bid != previous_id)),
            "reallocation_count": previous_reallocations + int(reallocated),
        }
    step_diag = getattr(env, "_engagement_target_step_diag", None)
    if isinstance(step_diag, dict):
        diag = step_diag.setdefault(
            aid, {"held": False, "switched": False, "reallocated": False,
                  "reallocation_reason": ""})
        diag["held"] = bool(diag.get("held", False) or bid == previous_id)
        diag["switched"] = bool(
            diag.get("switched", False) or (previous_id and bid != previous_id))
        diag["reallocated"] = bool(diag.get("reallocated", False) or reallocated)
        if reallocated:
            diag["reallocation_reason"] = reallocation_reason
    return bid, getattr(env, "blue_planes", {})[bid], selected
