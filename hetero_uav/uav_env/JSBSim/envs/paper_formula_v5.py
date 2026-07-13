"""TAM-HAPPO paper-formula reward with documented JSBSim adaptations.

This is not an exact reproduction of unpublished paper constants.  The TAM
reward path and BRMA Eq. 20-23 reference path are intentionally separate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..utils import get2d_AO_TA_R


CLAIM = (
    "TAM-HAPPO paper-formula implementation with documented JSBSim "
    "adaptations; not an exact reproduction of unpublished constants"
)
GLOBAL_REWARD_SCALE = 1.0 / 200.0
UAV_WEIGHTS = {"height": 10.0, "speed": 10.0, "angle": 15.0, "distance": 10.0, "dodge": 30.0}
TARGET_WEIGHTS = {"angle": 0.35, "distance": 0.25, "height": 0.20, "speed": 0.20}

UAV_FIELDS = (
    "height_pv_raw", "height_ph_raw", "height_raw", "height_adaptation_active",
    "speed_own", "speed_target", "speed_ratio", "speed_invalid", "speed_raw",
    "target_id", "target_score", "target_rank", "closest_target_id",
    "lock_target_id", "launch_target_id", "reward_target_matches_lock",
    "reward_target_matches_launch", "ata", "aa", "angle_raw", "distance_km",
    "distance_raw", "dodge_lambda", "dodge_missile_speed_current",
    "dodge_missile_speed_previous", "dodge_angle_raw", "dodge_speed_raw",
    "dodge_raw", "kill_event_raw", "death_event_raw", "oob_event_raw",
    "uav_raw_total", "uav_scaled_total",
)
MAV_FIELDS = (
    "dist_safe_raw", "nearest_threat_id", "nearest_threat_distance_m",
    "missile_safe_raw", "aspect_raw_sum", "aspect_raw_mean", "safety_raw",
    "support_rear_projection", "support_lateral_offset", "support_distance",
    "support_position_raw", "awareness_shared_pair_count",
    "awareness_valid_pair_count", "awareness_ratio", "awareness_raw",
    "support_raw", "death_event_raw", "team_kill_alive_raw",
    "team_kill_after_mav_death_raw", "shared_kill_raw", "direct_kill_raw",
    "event_credit_delta", "event_credit_used", "event_credit_cap",
    "mav_event_raw", "mav_raw_total", "mav_scaled_total",
)
COMMON_FIELDS = (
    "alive_before", "alive_after", "death_transition", "dead_before", "total",
    "reconstructed_sum", "identity_error", "brma_adv_reference_raw",
    "brma_adv_pair_count", "brma_adv_per_target_mean", "brma_key_entity_id",
    "brma_key_pair_contribution", "brma_end_reference_raw", "brma_overlay_enabled",
    "true_final_j", "environment_timeout", "censored", "terminal_observed",
    "red_alive_final", "blue_alive_final", "mav_alive_final",
    "unique_red_launch", "unique_red_hit", "unique_blue_launch", "unique_blue_hit",
)
V5_COMPONENT_FIELDS = tuple(dict.fromkeys(UAV_FIELDS + MAV_FIELDS + COMMON_FIELDS))
V5_TRAIN_FIELDS = (
    "v5_uav_height_mean", "v5_uav_speed_mean", "v5_uav_angle_mean",
    "v5_uav_distance_mean", "v5_uav_dodge_mean", "v5_uav_event_mean",
    "v5_mav_safety_mean", "v5_mav_support_mean", "v5_mav_event_mean",
    "v5_identity_max_abs",
)


def collect_v5_effective_samples(components: dict, roles: dict) -> tuple[dict[str, float], dict[str, float]]:
    sums = {key: 0.0 for key in V5_TRAIN_FIELDS}
    counts = {key: 0.0 for key in V5_TRAIN_FIELDS}
    for aid, comp in (components or {}).items():
        if not isinstance(comp, dict) or float(comp.get("alive_before", 0.0)) <= 0.5:
            continue
        if roles.get(aid) == "mav":
            values = {
                "v5_mav_safety_mean": comp.get("safety_raw", 0.0),
                "v5_mav_support_mean": comp.get("support_raw", 0.0),
                "v5_mav_event_mean": comp.get("mav_event_raw", 0.0),
            }
        else:
            values = {
                "v5_uav_height_mean": comp.get("height_raw", 0.0),
                "v5_uav_speed_mean": comp.get("speed_raw", 0.0),
                "v5_uav_angle_mean": comp.get("angle_raw", 0.0),
                "v5_uav_distance_mean": comp.get("distance_raw", 0.0),
                "v5_uav_dodge_mean": comp.get("dodge_raw", 0.0),
                "v5_uav_event_mean": (float(comp.get("kill_event_raw", 0.0))
                                       + float(comp.get("death_event_raw", 0.0))
                                       + float(comp.get("oob_event_raw", 0.0))),
            }
        for key, value in values.items():
            sums[key] += float(value); counts[key] += 1.0
        sums["v5_identity_max_abs"] = max(sums["v5_identity_max_abs"], abs(float(comp.get("identity_error", 0.0))))
        counts["v5_identity_max_abs"] = 1.0
    return sums, counts


def reset_v5_state(env) -> None:
    env._tam_v5_state = {
        "uav_death_seen": set(),
        "uav_oob_seen": set(),
        "mav_death_seen": False,
        "missile_speed": {},
        "mav_event_credit_used": 0.0,
        "launches": {},
        "terminals": set(),
        "red_launch": set(),
        "red_hit": set(),
        "blue_launch": set(),
        "blue_hit": set(),
        "new_red_hit": set(),
    }


def tam_speed_reward(own_speed: float, target_speed: float, eps: float = 1e-8) -> dict[str, float]:
    own = float(own_speed)
    target = float(target_speed)
    if not np.isfinite(own) or not np.isfinite(target) or own <= eps:
        return {"speed_own": own if np.isfinite(own) else 0.0,
                "speed_target": target if np.isfinite(target) else 0.0,
                "speed_ratio": 0.0, "speed_invalid": 1.0, "speed_raw": -1.0}
    ratio = target / own
    if target < 0.5 * own:
        raw = 1.0
    elif target <= 1.5 * own:
        raw = 2.0 - 2.0 * ratio
    else:
        raw = -1.0
    return {"speed_own": own, "speed_target": target, "speed_ratio": ratio,
            "speed_invalid": 0.0, "speed_raw": float(raw)}


def tam_distance_reward(distance_m: float) -> dict[str, float]:
    km = float(distance_m) / 1000.0
    if km <= 5.0:
        raw = 1.0
    elif km < 10.0:
        raw = math.exp(-0.921 * (km - 5.0))
    else:
        raw = -1.0
    return {"distance_km": km, "distance_raw": float(raw)}


def tam_angle_reward(ata: float, aa: float) -> float:
    value = 1.0 - (float(ata) + float(aa)) / math.pi
    if not np.isfinite(value):
        raise ValueError(f"non-finite TAM angle reward: ATA={ata}, AA={aa}")
    return float(value)


def brma_angle_situation(q_los_rad: float) -> float:
    """BRMA-MAPPO Eq. 20, including its published discontinuity at 4 deg."""
    q = float(q_los_rad)
    a1, a2, a3 = map(math.radians, (4.0, 15.0, 35.0))
    if q <= a1:
        return 10.0
    if q <= a2:
        return 1.0 + 2.0 * (a2 - q) / (a2 - a1)
    if q <= a3:
        return 1.0 - (q - a2) / (a3 - a2)
    return 0.0


def brma_distance_situation(distance_m: float) -> float:
    km = float(distance_m) / 1000.0
    return 1.0 if km <= 15.0 else float(math.exp(1.0 - km / 15.0))


def brma_end_reference(red_alive: int, blue_alive: int) -> float:
    return 0.0 if int(red_alive) == int(blue_alive) else 30.0 * (int(red_alive) - int(blue_alive))


def reconstruct_total(parts: list[float]) -> float:
    return float(sum(float(x) for x in parts))


def identity_error(total: float, parts: list[float]) -> float:
    return float(total) - reconstruct_total(parts)


def _safe_vec(sim, getter: str) -> np.ndarray:
    try:
        value = np.asarray(getattr(sim, getter)(), dtype=np.float64).reshape(-1)
    except Exception:
        value = np.zeros(3, dtype=np.float64)
    if value.size < 3:
        value = np.pad(value, (0, 3 - value.size))
    value = value[:3]
    return value if np.isfinite(value).all() else np.zeros(3, dtype=np.float64)


def _pair_geometry(env, attacker, target) -> dict[str, float]:
    return env._brma_tam_3d_geometry(attacker, target)


def paper_target_score(env, attacker, target, cfg: dict) -> dict[str, float]:
    """TAM-HAPPO Eq. 11-12 target-assessment score, separate from reward."""
    geom = _pair_geometry(env, attacker, target)
    ata = float(geom["tam_ata_rad"])
    aa = float(geom["tam_aa_rad"])
    distance = float(geom["target_distance_m"])
    own_pos = _safe_vec(attacker, "get_position")
    target_pos = _safe_vec(target, "get_position")
    own_vel = _safe_vec(attacker, "get_velocity")
    target_vel = _safe_vec(target, "get_velocity")
    assessment = cfg.get("target_assessment", {})
    hmax = float(assessment.get("relative_altitude_norm_m", 10000.0))
    vmax = float(assessment.get("relative_speed_norm_mps", 408.0))
    dmax = float(assessment.get("engagement_range_m", 14000.0))
    e_angle = 1.0 - (ata + aa) / (2.0 * math.pi)
    e_distance = 1.0 if distance <= dmax else 0.0
    e_height = float((own_pos[2] - target_pos[2]) / max(hmax, 1e-8))
    e_speed = float(np.linalg.norm(own_vel - target_vel) / max(vmax, 1e-8))
    score = (TARGET_WEIGHTS["angle"] * e_angle + TARGET_WEIGHTS["distance"] * e_distance
             + TARGET_WEIGHTS["height"] * e_height + TARGET_WEIGHTS["speed"] * e_speed)
    return {"score": float(score), "ata": ata, "aa": aa, "distance_m": distance,
            "target_angle_assessment": float(e_angle),
            "target_distance_assessment": float(e_distance),
            "target_height_assessment": float(e_height),
            "target_speed_assessment": float(e_speed)}


def _select_target(env, attacker, cfg: dict) -> tuple[str | None, Any | None, dict[str, float]]:
    candidates = []
    own_pos = _safe_vec(attacker, "get_position")
    for bid in env.blue_ids:
        blue = env.blue_planes.get(bid)
        if blue is None or not getattr(blue, "is_alive", False):
            continue
        values = paper_target_score(env, attacker, blue, cfg)
        values["target_id"] = bid
        values["nearest_distance"] = float(np.linalg.norm(_safe_vec(blue, "get_position") - own_pos))
        candidates.append(values)
    if not candidates:
        return None, None, {}
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["target_id"])))
    selected = candidates[0]
    selected["target_rank"] = 1.0
    selected["closest_target_id"] = min(candidates, key=lambda row: row["nearest_distance"])["target_id"]
    bid = str(selected["target_id"])
    return bid, env.blue_planes[bid], selected


def _dodge(env, sim, state: dict) -> dict[str, float]:
    out = {"dodge_lambda": 0.0, "dodge_missile_speed_current": 0.0,
           "dodge_missile_speed_previous": 0.0, "dodge_angle_raw": 0.0,
           "dodge_speed_raw": 0.0, "dodge_raw": 0.0}
    missiles = [m for m in list(getattr(sim, "under_missiles", None) or []) if getattr(m, "is_alive", False)]
    if not missiles:
        return out
    missile = min(missiles, key=lambda m: float(np.linalg.norm(_safe_vec(m, "get_position") - _safe_vec(sim, "get_position"))))
    mid = str(getattr(missile, "uid", getattr(missile, "_uid", id(missile))))
    mv = _safe_vec(missile, "get_velocity")
    los = _safe_vec(sim, "get_position") - _safe_vec(missile, "get_position")
    speed = float(np.linalg.norm(mv)); distance = float(np.linalg.norm(los))
    if speed <= 1e-8 or distance <= 1e-8:
        return out
    cos_lam = float(np.clip(np.dot(mv, los) / (speed * distance), -1.0, 1.0))
    lam = float(math.acos(cos_lam))
    prev = state["missile_speed"].get(mid)
    norm = float(env.tam_happo_paper_formula_v5_config.get("uav", {}).get("dodge_speed_norm_mps", 1000.0))
    r_am = -cos_lam
    r_sm = 0.0 if prev is None else (float(prev) - speed) / max(norm, 1e-8)
    state["missile_speed"][mid] = speed
    out.update({"dodge_lambda": lam, "dodge_missile_speed_current": speed,
                "dodge_missile_speed_previous": 0.0 if prev is None else float(prev),
                "dodge_angle_raw": r_am, "dodge_speed_raw": r_sm,
                "dodge_raw": float(r_am + r_sm)})
    return out


def _lock_and_launch_target(env, aid: str) -> tuple[str, str]:
    lock_id = str((getattr(env, "_lock_target", {}) or {}).get(aid, "") or "")
    launch_id = ""
    for record in getattr(env, "_launch_quality_step_records", []) or []:
        if str(record.get("shooter_id", "")) == aid:
            launch_id = str(record.get("target_id", "") or "")
            break
    return lock_id, launch_id


def _update_missile_events(env, state: dict) -> None:
    state["new_red_hit"] = set()
    for record in getattr(env, "_launch_quality_step_records", []) or []:
        mid = str(record.get("missile_id", "") or "")
        if not mid or mid in state["launches"]:
            continue
        shooter = str(record.get("shooter_id", "") or "")
        state["launches"][mid] = {"shooter": shooter,
                                  "source": str(record.get("launch_track_source", "") or "")}
        state["red_launch" if shooter.startswith("red_") else "blue_launch"].add(mid)
    for record in getattr(env, "_launch_quality_done_step_records", []) or []:
        mid = str(record.get("missile_id", "") or "")
        if not mid or mid in state["terminals"]:
            continue
        state["terminals"].add(mid)
        if str(record.get("raw_termination_reason", "")) != "hit" or mid not in state["launches"]:
            continue
        shooter = state["launches"][mid]["shooter"]
        state["red_hit" if shooter.startswith("red_") else "blue_hit"].add(mid)
        if shooter.startswith("red_"):
            state["new_red_hit"].add(mid)


def _brma_reference(env, sim) -> dict[str, Any]:
    pairs = []
    for bid in env.blue_ids:
        blue = env.blue_planes.get(bid)
        if blue is None or not getattr(blue, "is_alive", False):
            continue
        own = _pair_geometry(env, sim, blue)
        threat = _pair_geometry(env, blue, sim)
        own_term = brma_angle_situation(own["tam_ata_rad"]) * brma_distance_situation(own["target_distance_m"])
        threat_term = brma_angle_situation(threat["tam_ata_rad"]) * brma_distance_situation(threat["target_distance_m"])
        contribution = own_term - 0.8 * threat_term
        pairs.append((bid, float(contribution)))
    total = float(sum(value for _, value in pairs))
    key = max(pairs, key=lambda pair: abs(pair[1])) if pairs else ("", 0.0)
    return {"brma_adv_reference_raw": total, "brma_adv_pair_count": float(len(pairs)),
            "brma_adv_per_target_mean": total / max(len(pairs), 1),
            "brma_key_entity_id": key[0], "brma_key_pair_contribution": key[1]}


def _mav_position(env, mav, cfg: dict) -> dict[str, float]:
    attack = [env.red_planes[rid] for rid in env.red_ids
              if env.agent_roles.get(rid) == "attack_uav"
              and env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive]
    blue = [env.blue_planes[bid] for bid in env.blue_ids
            if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive]
    if not attack or not blue:
        return {"support_rear_projection": 0.0, "support_lateral_offset": 0.0,
                "support_distance": 0.0, "support_position_raw": 0.0}
    cu = np.mean([_safe_vec(x, "get_position") for x in attack], axis=0)
    cb = np.mean([_safe_vec(x, "get_position") for x in blue], axis=0)
    pm = _safe_vec(mav, "get_position")
    direction = cb - cu; norm = float(np.linalg.norm(direction[:2]))
    if norm <= 1e-8:
        return {"support_rear_projection": 0.0, "support_lateral_offset": 0.0,
                "support_distance": float(np.linalg.norm(pm - cu)), "support_position_raw": 0.0}
    toward = np.array([direction[0] / norm, direction[1] / norm, 0.0])
    scfg = cfg.get("mav", {}).get("support_position_adaptation", {})
    rear_offset = float(scfg.get("rear_offset_m", 8000.0))
    d_opt = float(scfg.get("d_opt_m", 8000.0)); d_max = float(scfg.get("d_max_m", 25000.0))
    ideal = cu - toward * rear_offset
    delta = pm - cu
    rear_projection = float(np.dot(delta, -toward))
    lateral = float(np.linalg.norm(delta[:2] - np.dot(delta, toward) * toward[:2]))
    db = float(np.linalg.norm(pm - ideal))
    if db < d_opt:
        r_pos = db / max(d_opt, 1e-8) - 1.0
    elif db < d_max:
        r_pos = 1.0 - (db - d_opt) / max(d_max - d_opt, 1e-8)
    else:
        r_pos = -0.5
    return {"support_rear_projection": rear_projection, "support_lateral_offset": lateral,
            "support_distance": db, "support_position_raw": float(r_pos)}


def _mav_reward(env, aid: str, sim, state: dict, cfg: dict, alive_before: bool) -> tuple[float, dict]:
    vals = {key: 0.0 for key in MAV_FIELDS}
    if not alive_before:
        return 0.0, vals
    alive_blue = [(bid, env.blue_planes[bid]) for bid in env.blue_ids
                  if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive]
    if alive_blue:
        pos = _safe_vec(sim, "get_position")
        nearest_id, nearest, near_d = min(
            ((bid, blue, float(np.linalg.norm(_safe_vec(blue, "get_position") - pos)))
             for bid, blue in alive_blue), key=lambda item: item[2])
        scfg = cfg.get("mav", {}).get("safety", {})
        danger = float(scfg.get("d_danger_m", 8000.0)); safe = float(scfg.get("d_safe_m", 15000.0))
        if near_d < danger:
            r_dist = -(1.0 - near_d / max(danger, 1e-8))
        elif near_d < safe:
            r_dist = -0.5 * (1.0 - (near_d - danger) / max(safe - danger, 1e-8))
        else:
            r_dist = 0.2
        vals.update({"nearest_threat_id": nearest_id, "nearest_threat_distance_m": near_d,
                     "dist_safe_raw": float(r_dist)})
    warning = getattr(sim, "check_missile_warning", lambda: None)()
    vals["missile_safe_raw"] = -1.0 if warning is not None else 0.0
    aspects = []
    mav_feature = env._tam_v2_feature(sim)
    for _bid, blue in alive_blue:
        _ao, ta, _distance = get2d_AO_TA_R(mav_feature, env._tam_v2_feature(blue))
        if ta < math.pi / 4.0:
            aspects.append(-(1.0 - ta / (math.pi / 4.0)))
        else:
            aspects.append(0.0)
    vals["aspect_raw_sum"] = float(sum(aspects))
    vals["aspect_raw_mean"] = float(np.mean(aspects)) if aspects else 0.0
    vals["safety_raw"] = (0.5 * vals["dist_safe_raw"] + 0.3 * vals["missile_safe_raw"]
                          + 0.2 * vals["aspect_raw_sum"])
    vals.update(_mav_position(env, sim, cfg))
    valid = shared = 0
    aware = 0.0
    for rid in env.red_ids:
        if env.agent_roles.get(rid) != "attack_uav" or not getattr(env.red_planes.get(rid), "is_alive", False):
            continue
        for bid, blue in alive_blue:
            valid += 1
            track = env._mav_shared_track_state(rid, bid)
            if track["mav_shared_visible"] and not track["direct_visible"]:
                shared += 1
                geom = _pair_geometry(env, sim, blue)
                ao = float(geom["tam_ata_rad"])
                if ao < math.pi / 2.0:
                    aware += 0.3 * (1.0 - ao / (math.pi / 2.0))
    vals.update({"awareness_shared_pair_count": float(shared),
                 "awareness_valid_pair_count": float(valid),
                 "awareness_ratio": float(shared / max(valid, 1)), "awareness_raw": float(aware)})
    vals["support_raw"] = 0.6 * vals["support_position_raw"] + 0.4 * vals["awareness_raw"]
    alive_after = bool(getattr(sim, "is_alive", False))
    if alive_before and not alive_after and not state["mav_death_seen"]:
        vals["death_event_raw"] = -float(cfg.get("unknown_constants", {}).get("mav_death_penalty", 200.0))
        state["mav_death_seen"] = True
    step_kills = sum(int(getattr(env, "_step_kill_count", {}).get(rid, 0)) for rid in env.red_ids
                     if env.agent_roles.get(rid) == "attack_uav")
    if alive_after:
        per_kill = float(cfg.get("unknown_constants", {}).get("mav_team_credit_per_kill", 100.0))
        cap = float(cfg.get("unknown_constants", {}).get("mav_team_credit_cap", 200.0))
        available = max(0.0, cap - float(state["mav_event_credit_used"]))
        delta = min(per_kill * step_kills, available)
        state["mav_event_credit_used"] += delta
        vals["team_kill_alive_raw"] = float(step_kills)
        vals["event_credit_delta"] = float(delta)
    else:
        vals["team_kill_after_mav_death_raw"] = float(step_kills)
    for mid in state["new_red_hit"]:
        launch = state["launches"].get(mid, {})
        if launch.get("source") == "mav_shared":
            vals["shared_kill_raw"] += 1.0
        elif launch.get("source") in {"direct", "direct_and_mav_shared"}:
            vals["direct_kill_raw"] += 1.0
    vals["event_credit_used"] = float(state["mav_event_credit_used"])
    vals["event_credit_cap"] = float(cfg.get("unknown_constants", {}).get("mav_team_credit_cap", 200.0))
    vals["mav_event_raw"] = vals["death_event_raw"] + vals["event_credit_delta"]
    vals["mav_raw_total"] = vals["safety_raw"] + vals["support_raw"] + vals["mav_event_raw"]
    vals["mav_scaled_total"] = GLOBAL_REWARD_SCALE * vals["mav_raw_total"]
    return float(vals["mav_scaled_total"]), vals


def _uav_reward(env, aid: str, sim, state: dict, cfg: dict, alive_before: bool) -> tuple[float, dict]:
    vals = {key: 0.0 for key in UAV_FIELDS}
    if not alive_before:
        return 0.0, vals
    height_raw, height_logs = env._tam_table1_uav_height_raw(sim, cfg)
    vals.update({"height_pv_raw": float(height_logs["tam_table1_uav_height_pv"]),
                 "height_ph_raw": float(height_logs["tam_table1_uav_height_ph"]),
                 "height_raw": float(height_raw), "height_adaptation_active": 1.0})
    target_id, target, target_logs = _select_target(env, sim, cfg)
    if target is not None:
        own_speed = float(np.linalg.norm(_safe_vec(sim, "get_velocity")))
        target_speed = float(np.linalg.norm(_safe_vec(target, "get_velocity")))
        vals.update(tam_speed_reward(own_speed, target_speed))
        vals.update({"target_id": target_id, "target_score": float(target_logs["score"]),
                     "target_rank": float(target_logs["target_rank"]),
                     "closest_target_id": str(target_logs["closest_target_id"]),
                     "ata": float(target_logs["ata"]), "aa": float(target_logs["aa"]),
                     "angle_raw": tam_angle_reward(target_logs["ata"], target_logs["aa"])})
        vals.update(tam_distance_reward(target_logs["distance_m"]))
    else:
        vals.update(tam_speed_reward(float(np.linalg.norm(_safe_vec(sim, "get_velocity"))), 0.0))
    lock_id, launch_id = _lock_and_launch_target(env, aid)
    vals.update({"lock_target_id": lock_id, "launch_target_id": launch_id,
                 "reward_target_matches_lock": float(bool(target_id and target_id == lock_id)),
                 "reward_target_matches_launch": float(bool(target_id and target_id == launch_id))})
    vals.update(_dodge(env, sim, state))
    vals["kill_event_raw"] = 200.0 * int(getattr(env, "_step_kill_count", {}).get(aid, 0))
    alive_after = bool(getattr(sim, "is_alive", False))
    if alive_before and not alive_after and aid not in state["uav_death_seen"]:
        reason = str(env._brma_tam_death_reason(aid)).lower()
        is_oob_death = any(token in reason for token in ("boundary", "out_of_zone", "outofzone"))
        if is_oob_death:
            vals["oob_event_raw"] = -100.0
            state["uav_oob_seen"].add(aid)
        else:
            vals["death_event_raw"] = -200.0
        state["uav_death_seen"].add(aid)
    if alive_after and env._brma_tam_horizontal_oob(sim) and aid not in state["uav_oob_seen"]:
        vals["oob_event_raw"] = -100.0
        state["uav_oob_seen"].add(aid)
    event = vals["kill_event_raw"] + vals["death_event_raw"] + vals["oob_event_raw"]
    vals["uav_raw_total"] = (10.0 * vals["height_raw"] + 10.0 * vals["speed_raw"]
                             + 15.0 * vals["angle_raw"] + 10.0 * vals["distance_raw"]
                             + 30.0 * vals["dodge_raw"] + event)
    vals["uav_scaled_total"] = GLOBAL_REWARD_SCALE * vals["uav_raw_total"]
    return float(vals["uav_scaled_total"]), vals


def compute_v5_reward(env, base_rewards: dict, components: dict):
    cfg = env.tam_happo_paper_formula_v5_config
    state = getattr(env, "_tam_v5_state", None)
    if state is None:
        raise RuntimeError("v5 episode state is not initialized")
    _update_missile_events(env, state)
    alive_before_map = getattr(env, "_brma_tam_alive_before_step", {})
    red_alive = sum(bool(getattr(env.red_planes.get(rid), "is_alive", False)) for rid in env.red_ids)
    blue_alive = sum(bool(getattr(env.blue_planes.get(bid), "is_alive", False)) for bid in env.blue_ids)
    mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), "")
    mav_alive = bool(mav_id and getattr(env.red_planes.get(mav_id), "is_alive", False))
    round_over = bool(red_alive == 0 or blue_alive == 0 or env.current_step >= env.max_steps)
    timeout = bool(env.current_step >= env.max_steps and red_alive > 0 and blue_alive > 0)
    final_j = float(red_alive / max(len(env.red_ids), 1) - blue_alive / max(len(env.blue_ids), 1))
    brma_end = brma_end_reference(red_alive, blue_alive) if round_over else 0.0
    for aid in env.red_ids:
        sim = env.red_planes.get(aid)
        role = env.agent_roles.get(aid, "")
        alive_before = bool(alive_before_map.get(aid, getattr(sim, "is_alive", False)))
        alive_after = bool(getattr(sim, "is_alive", False))
        vals: dict[str, Any] = {key: 0.0 for key in V5_COMPONENT_FIELDS}
        vals.update({"alive_before": float(alive_before), "alive_after": float(alive_after),
                     "death_transition": float(alive_before and not alive_after),
                     "dead_before": float(not alive_before), "true_final_j": final_j,
                     "environment_timeout": float(timeout), "censored": 0.0,
                     "terminal_observed": float(round_over), "red_alive_final": float(red_alive),
                     "blue_alive_final": float(blue_alive), "mav_alive_final": float(mav_alive),
                     "unique_red_launch": float(len(state["red_launch"])),
                     "unique_red_hit": float(len(state["red_hit"])),
                     "unique_blue_launch": float(len(state["blue_launch"])),
                     "unique_blue_hit": float(len(state["blue_hit"])),
                     "brma_end_reference_raw": float(brma_end), "brma_overlay_enabled": 0.0})
        if alive_before and sim is not None:
            vals.update(_brma_reference(env, sim))
        if role == "mav":
            total, role_vals = _mav_reward(env, aid, sim, state, cfg, alive_before)
            vals.update(role_vals)
            parts = [GLOBAL_REWARD_SCALE * vals["safety_raw"],
                     GLOBAL_REWARD_SCALE * vals["support_raw"],
                     GLOBAL_REWARD_SCALE * vals["mav_event_raw"]]
        else:
            total, role_vals = _uav_reward(env, aid, sim, state, cfg, alive_before)
            vals.update(role_vals)
            parts = [GLOBAL_REWARD_SCALE * 10.0 * vals["height_raw"],
                     GLOBAL_REWARD_SCALE * 10.0 * vals["speed_raw"],
                     GLOBAL_REWARD_SCALE * 15.0 * vals["angle_raw"],
                     GLOBAL_REWARD_SCALE * 10.0 * vals["distance_raw"],
                     GLOBAL_REWARD_SCALE * 30.0 * vals["dodge_raw"],
                     GLOBAL_REWARD_SCALE * (vals["kill_event_raw"] + vals["death_event_raw"] + vals["oob_event_raw"])]
        reconstructed = reconstruct_total(parts)
        vals.update({"total": float(total), "reconstructed_sum": reconstructed,
                     "identity_error": identity_error(total, parts)})
        if abs(vals["identity_error"]) > 1e-8:
            raise ValueError(f"v5 reward identity failed for {aid}: {vals['identity_error']}")
        comp = components.setdefault(aid, {})
        comp.update(vals); comp["total"] = float(total)
        base_rewards[aid] = float(total)
    return base_rewards, components
