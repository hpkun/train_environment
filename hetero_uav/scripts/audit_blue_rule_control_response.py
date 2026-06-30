"""Audit blue brma_rule command-to-aircraft pursuit response.

This script is read-only with respect to environment mechanics. It traces the
same brma_rule policy path used by train/eval and records whether saturated
heading commands are followed by aircraft yaw/track response and range closure.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from algorithms.mappo.opponent_policy import OpponentPolicy  # noqa: E402

_HIGH_ROLL_THRESH_RAD = np.deg2rad(75.0)
_EXTREME_ROLL_THRESH_RAD = np.deg2rad(105.0)

from scripts.audit_blue_rule_pursuit_logic import (  # noqa: E402
    DEFAULT_CONFIG,
    _coordinated_assignment_debug,
    _infer_branch,
    _launch_flags,
    _red_actions,
    _select_target_debug,
)
import rule_based_agent as _blue_rule_module  # noqa: E402
from rule_based_agent import (  # noqa: E402
    HARD_DECK,
    SAFE_COMBAT_ALT,
    _boundary_outward_heading_component,
    _boundary_patrol_pressure,
    _should_override_for_boundary_safety,
)
from uav_env import make_env  # noqa: E402


FIELDS = [
    "episode", "step", "blue_id",
    "pursuit_variant", "simple_target_selection", "desired_heading_source",
    "uses_red_action_bounds", "simple_reacquire_active", "simple_lost_steps",
    "selected_range_m", "selected_AO_rad", "selected_TA_rad", "selected_target_quality",
    "action_heading_abs_rad", "action_heading_norm",
    "branch_state", "selected_red_id",
    "target_quality",
    "target_range_m", "nearest_red_range_m",
    "AO_rad", "TA_rad",
    "lead_bearing_rad", "target_bearing_rad", "desired_heading_rad",
    "heading_error_to_desired_rad", "heading_limit_deg",
    "safe_pursuit_mode_active", "safe_pursuit_direct_heading_used",
    "fallback_to_delta10_reason",
    "heading_error_to_target_rad",
    "heading_cmd_internal", "heading_cmd_saturated",
    "target_heading_cmd_rad",
    "blue_yaw_rad", "blue_track_heading_rad", "blue_heading_source",
    "blue_heading_rate_rad_s", "blue_track_heading_rate_rad_s",
    "blue_roll_rad", "blue_pitch_rad", "blue_speed_mps", "blue_alt_m",
    "lead_bearing_error_rad",
    "pid_target_heading_rad",
    "actual_heading_next_rad", "actual_track_heading_next_rad",
    "actual_heading_delta_next_rad", "actual_track_heading_delta_next_rad",
    "command_tracking_error_rad", "command_track_tracking_error_rad",
    "delta10_heading_error_after_cmd_rad", "direct_probe_heading_cmd_rad",
    "direct_probe_heading_error_after_cmd_rad", "direct_probe_improvement_rad",
    "heading_delta_5_steps_rad", "heading_delta_10_steps_rad", "heading_delta_20_steps_rad",
    "range_delta_5_steps_m", "range_delta_10_steps_m", "range_delta_20_steps_m",
    "red_pos_n", "red_pos_e", "red_alt_m",
    "blue_pos_n", "blue_pos_e", "range_delta_next_m",
    "distance_to_center_m", "boundary_pressure", "outward_component",
    "hard_deck_active", "stall_risk_active", "anti_stall_active",
    "boundary_safety_active", "missile_fired_this_step", "launch_track_ok",
    "launch_geometry_ok", "range_ok", "ao_ok", "ta_ok",
    "death_or_outcome_if_terminal", "death_reason_if_any",
    "episode_outcome",
    # Roll safety diagnostics
    "roll_recovery_active", "extreme_roll_recovery_active",
    "high_roll_active", "extreme_roll_active",
    "action_heading_delta_from_prev_rad", "target_heading_cmd_delta_from_prev_rad",
    "action_heading_norm",
    "desired_heading_source",
    "blue_roll_abs_rad", "blue_roll_abs_deg",
    "blue_speed_mps",
    "aileron_cmd", "elevator_cmd", "throttle_cmd",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _delta10_target_heading(current_heading_rad: float, heading_cmd_internal: float) -> float:
    return _wrap_pi(float(current_heading_rad) + float(heading_cmd_internal) * math.radians(10.0))


def _direct_heading_probe(_current_heading_rad: float, target_bearing_rad: float) -> float:
    """Diagnostic-only absolute heading probe without the blue rule ±10 deg limit."""

    return _wrap_pi(target_bearing_rad)


def _track_heading(velocity: np.ndarray) -> float:
    vel = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if vel.size < 2 or np.linalg.norm(vel[:2]) < 1e-6:
        return 0.0
    return float(math.atan2(float(vel[1]), float(vel[0])))


def _decision_dt(env) -> float:
    sim_freq = float(getattr(env, "sim_freq", 60.0) or 60.0)
    interaction = float(getattr(env, "agent_interaction_steps", 12.0) or 12.0)
    return interaction / sim_freq


def _red_blue_geometry(env, blue_id: str, target_idx: int | None) -> dict[str, Any]:
    blue = env.blue_planes[blue_id]
    bpos = np.asarray(blue.get_position(), dtype=np.float64)
    nearest = []
    for red in env.red_planes.values():
        if red is not None and red.is_alive:
            nearest.append(float(np.linalg.norm(np.asarray(red.get_position(), dtype=np.float64) - bpos)))
    if target_idx is None or target_idx >= len(env.red_ids):
        return {
            "selected_red_id": "",
            "target_range_m": "",
            "nearest_red_range_m": min(nearest) if nearest else float("nan"),
            "target_bearing_rad": "",
            "heading_error_to_target_rad": "",
            "red_pos_n": "",
            "red_pos_e": "",
            "red_alt_m": "",
        }
    rid = env.red_ids[target_idx]
    red = env.red_planes[rid]
    rpos = np.asarray(red.get_position(), dtype=np.float64)
    delta = rpos - bpos
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    yaw = float(blue.get_rpy()[2])
    return {
        "selected_red_id": rid,
        "target_range_m": float(np.linalg.norm(delta)),
        "nearest_red_range_m": min(nearest) if nearest else float("nan"),
        "target_bearing_rad": bearing,
        "heading_error_to_target_rad": abs(_wrap_pi(bearing - yaw)),
        "red_pos_n": float(rpos[0]),
        "red_pos_e": float(rpos[1]),
        "red_alt_m": float(rpos[2]),
    }


def _lead_bearing_from_obs(obs: dict, target_state: np.ndarray | None, own_heading: float) -> float | str:
    if target_state is None or target_state.size < 7:
        return ""
    ao = float(target_state[3]) * math.pi
    ta = float(target_state[4]) * math.pi
    quality_is_awacs = abs(ta) <= 1e-4
    if quality_is_awacs:
        return _wrap_pi(own_heading + ao)

    ego_state = np.asarray(obs.get("ego_state", []), dtype=np.float64).reshape(-1)
    vel = np.asarray(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    if ego_state.size < 11 or vel.size < 2:
        return _wrap_pi(own_heading + ao)
    roll = math.atan2(float(ego_state[7]), float(ego_state[8]))
    pitch = math.atan2(float(ego_state[9]), float(ego_state[10]))
    speed = max(float(np.hypot(vel[0], vel[1])), 1.0)

    r = float(target_state[5]) * 80000.0
    v_tgt = float(target_state[6]) * 600.0
    dx_body = float(target_state[0]) * 40000.0
    dy_body = float(target_state[1]) * 40000.0
    dz_body = -float(target_state[2]) * 10000.0

    c_psi, s_psi = math.cos(own_heading), math.sin(own_heading)
    c_theta, s_theta = math.cos(pitch), math.sin(pitch)
    c_phi, s_phi = math.cos(roll), math.sin(roll)
    dn = (c_psi * c_theta) * dx_body \
        + (c_psi * s_theta * s_phi - s_psi * c_phi) * dy_body \
        + (c_psi * s_theta * c_phi + s_psi * s_phi) * dz_body
    de = (s_psi * c_theta) * dx_body \
        + (s_psi * s_theta * s_phi + c_psi * c_phi) * dy_body \
        + (s_psi * s_theta * c_phi - c_psi * s_phi) * dz_body

    target_heading = own_heading + ao + ta
    t_lead = min(r / max(speed, v_tgt, 1.0), 30.0)
    lead_n = dn + v_tgt * math.cos(target_heading) * t_lead
    lead_e = de + v_tgt * math.sin(target_heading) * t_lead
    return _wrap_pi(math.atan2(lead_e, lead_n))


def _death_reason(env, aid: str) -> str:
    reasons = getattr(env, "_death_reasons", {}) or {}
    return str(reasons.get(aid, ""))


def _terminal_label(terminated: dict, truncated: dict, info: dict) -> str:
    if not (all(terminated.values()) or all(truncated.values())):
        return ""
    winners = info.get("__winner__", "") or info.get("winner", "")
    reasons = info.get("__episode_end_reason__", "") or info.get("episode_end_reason", "")
    return str(reasons or winners or "episode_done")


def _annotate_future_response(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode"]), str(row["blue_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: int(r["step"]))
        for idx, row in enumerate(group):
            yaw0 = row.get("blue_yaw_rad")
            range0 = row.get("nearest_red_range_m")
            for horizon in (5, 10, 20):
                if idx + horizon >= len(group) or yaw0 == "" or range0 == "":
                    row[f"heading_delta_{horizon}_steps_rad"] = ""
                    row[f"range_delta_{horizon}_steps_m"] = ""
                    continue
                later = group[idx + horizon]
                row[f"heading_delta_{horizon}_steps_rad"] = _wrap_pi(
                    float(later["blue_yaw_rad"]) - float(yaw0))
                row[f"range_delta_{horizon}_steps_m"] = (
                    float(later["nearest_red_range_m"]) - float(range0)
                )


def _safe_pursuit_diag(
    opponent_policy: str,
    branch: str,
    target_quality: str,
    lead_bearing: float | str,
    yaw: float,
    ao_rad: float | str,
    alt_m: float,
    speed_mps: float,
    roll_abs: float,
    own_position: np.ndarray,
) -> dict[str, Any]:
    if target_quality == "awacs" and ao_rad != "":
        desired = _wrap_pi(yaw + float(ao_rad))
    elif lead_bearing != "":
        desired = float(lead_bearing)
    else:
        desired = ""
    if opponent_policy != "brma_rule_safe_pursuit" or branch != "combat" or desired == "":
        reason = "not_safe_pursuit" if opponent_policy != "brma_rule_safe_pursuit" else "non_combat_safety_or_no_target"
        return {
            "desired_heading_rad": desired,
            "heading_error_to_desired_rad": abs(_wrap_pi(float(desired) - yaw)) if desired != "" else "",
            "heading_limit_deg": 10.0,
            "safe_pursuit_mode_active": 0,
            "safe_pursuit_direct_heading_used": 0,
            "fallback_to_delta10_reason": reason,
        }
    return {
        "desired_heading_rad": desired,
        "heading_error_to_desired_rad": abs(_wrap_pi(float(desired) - yaw)),
        "heading_limit_deg": 180.0,
        "safe_pursuit_mode_active": 1,
        "safe_pursuit_direct_heading_used": 1,
        "fallback_to_delta10_reason": "",
    }


def run_audit(config: str, episodes: int, max_steps: int, output_dir: Path,
              red_mode: str, opponent_policy: str = "brma_rule") -> None:
    rows: list[dict[str, Any]] = []
    opponent = OpponentPolicy(opponent_policy, seed=23)
    for ep in range(episodes):
        env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
        try:
            obs, info = env.reset(seed=ep)
            opponent.reset_memory()
            _blue_rule_module._last_target_bearing.clear()
            _blue_rule_module._lost_target_steps.clear()
            _blue_rule_module._prev_heading_cmd.clear()
            _blue_rule_module._prev_lead_bearing.clear()
            _blue_rule_module._simple_last_seen_bearing.clear()
            _blue_rule_module._simple_lost_steps.clear()
            _blue_rule_module._simple_debug_state.clear()
            prev_yaw: dict[str, float] = {}
            prev_track: dict[str, float] = {}
            dt = _decision_dt(env)
            for step in range(max_steps):
                engaged_targets = env.refresh_engaged_targets()
                assignments = _coordinated_assignment_debug(
                    obs, env.blue_ids, len(env.blue_ids), len(env.red_ids), engaged_targets)
                actions_blue = opponent.act(obs, env.blue_ids, env=env)
                current: dict[str, dict[str, Any]] = {}
                for bid in env.blue_ids:
                    bobs = obs.get(bid, {})
                    blue = env.blue_planes[bid]
                    bpos = np.asarray(blue.get_position(), dtype=np.float64)
                    bvel = np.asarray(blue.get_velocity(), dtype=np.float64)
                    roll, pitch, yaw = [float(v) for v in blue.get_rpy()]
                    track = _track_heading(bvel)
                    target = _select_target_debug(
                        bobs, len(env.blue_ids), len(env.red_ids),
                        forced_target_idx=assignments.get(bid))
                    geom = _red_blue_geometry(env, bid, target["target_idx"])
                    action = np.asarray(actions_blue.get(bid, np.zeros(3, dtype=np.float32)), dtype=np.float32)
                    target_heading_cmd = _wrap_pi(float(action[1]) * math.pi)
                    heading_cmd_internal = _wrap_pi(target_heading_cmd - yaw) / math.radians(10.0)
                    lead_bearing = _lead_bearing_from_obs(bobs, target["target_state"], yaw)
                    target_bearing = geom["target_bearing_rad"]
                    direct_cmd = (
                        _direct_heading_probe(yaw, float(target_bearing))
                        if target_bearing != "" else ""
                    )
                    branch = _infer_branch(
                        bobs,
                        target["target_idx"],
                        bpos,
                        yaw,
                    )
                    pressure = _boundary_patrol_pressure(bpos)
                    outward = _boundary_outward_heading_component(bpos, yaw)
                    current[bid] = {
                        "episode": ep,
                        "step": step,
                        "blue_id": bid,
                        "pursuit_variant": "legacy_delta10",
                        "simple_target_selection": "",
                        "desired_heading_source": "",
                        "uses_red_action_bounds": 0,
                        "simple_reacquire_active": 0,
                        "simple_lost_steps": "",
                        "selected_range_m": "",
                        "selected_AO_rad": "",
                        "selected_TA_rad": "",
                        "selected_target_quality": "",
                        "action_heading_abs_rad": "",
                        "action_heading_norm": "",
                        "branch_state": branch,
                        "selected_red_id": geom["selected_red_id"],
                        "target_quality": target["target_quality"],
                        "target_range_m": geom["target_range_m"],
                        "nearest_red_range_m": geom["nearest_red_range_m"],
                        "AO_rad": float(target["target_state"][3]) * math.pi if target["target_state"] is not None and target["target_state"].size > 3 else "",
                        "TA_rad": float(target["target_state"][4]) * math.pi if target["target_state"] is not None and target["target_state"].size > 4 else "",
                        "heading_error_to_target_rad": geom["heading_error_to_target_rad"],
                        "heading_cmd_internal": float(np.clip(heading_cmd_internal, -1.0, 1.0)),
                        "heading_cmd_saturated": int(abs(heading_cmd_internal) >= 0.99),
                        "target_heading_cmd_rad": target_heading_cmd,
                        "blue_yaw_rad": yaw,
                        "blue_track_heading_rad": track,
                        "blue_heading_source": "sim_rpy_yaw",
                        "blue_heading_rate_rad_s": (
                            _wrap_pi(yaw - prev_yaw[bid]) / dt if bid in prev_yaw else ""
                        ),
                        "blue_track_heading_rate_rad_s": (
                            _wrap_pi(track - prev_track[bid]) / dt if bid in prev_track else ""
                        ),
                        "blue_roll_rad": roll,
                        "blue_pitch_rad": pitch,
                        "blue_speed_mps": float(np.linalg.norm(bvel)),
                        "blue_alt_m": float(bpos[2]),
                        "target_bearing_rad": target_bearing,
                        "lead_bearing_rad": lead_bearing,
                        "lead_bearing_error_rad": (
                            abs(_wrap_pi(float(lead_bearing) - yaw)) if lead_bearing != "" else ""
                        ),
                        "pid_target_heading_rad": target_heading_cmd,
                        "delta10_heading_error_after_cmd_rad": (
                            abs(_wrap_pi(float(target_bearing) - target_heading_cmd))
                            if target_bearing != "" else ""
                        ),
                        "direct_probe_heading_cmd_rad": direct_cmd,
                        "direct_probe_heading_error_after_cmd_rad": (
                            abs(_wrap_pi(float(target_bearing) - float(direct_cmd)))
                            if direct_cmd != "" and target_bearing != "" else ""
                        ),
                        "direct_probe_improvement_rad": (
                            abs(_wrap_pi(float(target_bearing) - target_heading_cmd))
                            - abs(_wrap_pi(float(target_bearing) - float(direct_cmd)))
                            if direct_cmd != "" and target_bearing != "" else ""
                        ),
                        "red_pos_n": geom["red_pos_n"],
                        "red_pos_e": geom["red_pos_e"],
                        "red_alt_m": geom["red_alt_m"],
                        "blue_pos_n": float(bpos[0]),
                        "blue_pos_e": float(bpos[1]),
                        "distance_to_center_m": float(np.linalg.norm(bpos[:2])),
                        "boundary_pressure": pressure,
                        "outward_component": outward,
                        "hard_deck_active": int(float(bpos[2]) < HARD_DECK),
                        "stall_risk_active": int(float(np.linalg.norm(bvel)) < 250.0),
                        "anti_stall_active": int(branch == "anti_stall"),
                        "boundary_safety_active": int(_should_override_for_boundary_safety(bpos, yaw)),
                        "missile_fired_this_step": 0,
                        "launch_track_ok": 0,
                        "launch_geometry_ok": 0,
                        "range_ok": 0,
                        "ao_ok": 0,
                        "ta_ok": int(target["target_state"] is not None and target["target_state"].size > 4 and abs(float(target["target_state"][4])) > 1e-4),
                        "death_or_outcome_if_terminal": "",
                        "death_reason_if_any": _death_reason(env, bid),
                        "episode_outcome": "",
                    }
                    current[bid].update(_safe_pursuit_diag(
                        opponent_policy=opponent_policy,
                        branch=branch,
                        target_quality=target["target_quality"],
                        lead_bearing=lead_bearing,
                        yaw=yaw,
                        ao_rad=current[bid]["AO_rad"],
                        alt_m=float(bpos[2]),
                        speed_mps=float(np.linalg.norm(bvel)),
                        roll_abs=abs(roll),
                        own_position=bpos,
                    ))
                    if opponent_policy == "brma_rule_safe_pursuit":
                        b_idx = int(bid.split("_")[1])
                        dbg = dict(getattr(_blue_rule_module, "_simple_debug_state", {}).get(b_idx, {}))
                        if dbg:
                            current[bid].update({
                                "pursuit_variant": dbg.get("pursuit_variant", "simple_safe_pursuit"),
                                "simple_target_selection": dbg.get("simple_target_selection", "nearest_valid"),
                                "desired_heading_source": dbg.get("desired_heading_source", ""),
                                "uses_red_action_bounds": dbg.get("uses_red_action_bounds", 1),
                                "simple_reacquire_active": dbg.get("simple_reacquire_active", 0),
                                "simple_lost_steps": dbg.get("simple_lost_steps", ""),
                                "selected_red_id": (
                                    f"red_{dbg['selected_target_idx']}"
                                    if dbg.get("selected_target_idx") is not None else ""
                                ),
                                "selected_range_m": dbg.get("selected_range_m", ""),
                                "selected_AO_rad": dbg.get("selected_AO_rad", ""),
                                "selected_TA_rad": dbg.get("selected_TA_rad", ""),
                                "selected_target_quality": dbg.get("selected_target_quality", ""),
                                "desired_heading_rad": dbg.get("action_heading_abs_rad", ""),
                                "heading_error_to_desired_rad": (
                                    abs(_wrap_pi(float(dbg["action_heading_abs_rad"]) - yaw))
                                    if dbg.get("action_heading_abs_rad", "") != "" else ""
                                ),
                                "action_heading_abs_rad": dbg.get("action_heading_abs_rad", ""),
                                "action_heading_norm": dbg.get("action_heading_norm", ""),
                                "roll_recovery_active": dbg.get("roll_recovery_active", 0),
                                "extreme_roll_recovery_active": dbg.get("extreme_roll_recovery_active", 0),
                            })
                    prev_yaw[bid] = yaw
                    prev_track[bid] = track

                full_actions = {**_red_actions(env, red_mode), **actions_blue}
                obs, _rewards, terminated, truncated, info = env.step(full_actions)
                terminal = _terminal_label(terminated, truncated, info)
                for bid, row in current.items():
                    launch = _launch_flags(info, bid)
                    blue = env.blue_planes[bid]
                    bvel_next = np.asarray(blue.get_velocity(), dtype=np.float64)
                    yaw_next = float(blue.get_rpy()[2])
                    track_next = _track_heading(bvel_next)
                    next_geom = _red_blue_geometry(env, bid, assignments.get(bid))
                    row["actual_heading_next_rad"] = yaw_next
                    row["actual_track_heading_next_rad"] = track_next
                    row["actual_heading_delta_next_rad"] = _wrap_pi(yaw_next - float(row["blue_yaw_rad"]))
                    row["actual_track_heading_delta_next_rad"] = _wrap_pi(track_next - float(row["blue_track_heading_rad"]))
                    row["command_tracking_error_rad"] = abs(_wrap_pi(float(row["target_heading_cmd_rad"]) - yaw_next))
                    row["command_track_tracking_error_rad"] = abs(_wrap_pi(float(row["target_heading_cmd_rad"]) - track_next))
                    # Roll safety diagnostics
                    row["high_roll_active"] = int(abs(roll) > _HIGH_ROLL_THRESH_RAD)
                    row["extreme_roll_active"] = int(abs(roll) > _EXTREME_ROLL_THRESH_RAD)
                    row["blue_roll_abs_rad"] = abs(roll)
                    row["blue_roll_abs_deg"] = float(np.rad2deg(abs(roll)))
                    row["blue_speed_mps"] = float(np.linalg.norm(bvel))
                    # Control surface probes (unavailable unless simulator exposes them)
                    row["aileron_cmd"] = ""
                    row["elevator_cmd"] = ""
                    row["throttle_cmd"] = ""
                    # Heading deltas from previous step
                    row["action_heading_delta_from_prev_rad"] = ""
                    row["target_heading_cmd_delta_from_prev_rad"] = ""
                    row["range_delta_next_m"] = (
                        float(next_geom["nearest_red_range_m"]) - float(row["nearest_red_range_m"])
                        if row["nearest_red_range_m"] != "" and next_geom["nearest_red_range_m"] != "" else ""
                    )
                    row["missile_fired_this_step"] = launch[0]
                    row["launch_track_ok"] = launch[1]
                    row["launch_geometry_ok"] = launch[2]
                    row["range_ok"] = launch[3]
                    row["ao_ok"] = launch[4]
                    row["death_or_outcome_if_terminal"] = terminal
                    row["death_reason_if_any"] = _death_reason(env, bid)
                    row["episode_outcome"] = terminal
                    rows.append(row)
                if terminal:
                    break
        finally:
            env.close()
    _annotate_future_response(rows)
    _write_outputs(output_dir, rows, config, episodes, max_steps, red_mode, opponent_policy)


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        val = row.get(key, "")
        if val == "":
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return vals


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = max(len(rows), 1)
    branch_counts = Counter(str(r.get("branch_state", "")) for r in rows)
    high = [
        r for r in rows
        if r.get("heading_error_to_target_rad") != ""
        and float(r["heading_error_to_target_rad"]) > math.radians(90.0)
    ]
    sat_high = [r for r in high if int(r.get("heading_cmd_saturated", 0)) == 1]
    high_range_next = [
        float(r["range_delta_next_m"]) for r in high
        if r.get("range_delta_next_m") != "" and np.isfinite(float(r["range_delta_next_m"]))
    ]
    high_heading_delta = _finite_values(high, "actual_heading_delta_next_rad")
    high_roll_abs = [abs(v) for v in _finite_values(high, "blue_roll_rad")]
    tracking = _finite_values(rows, "command_tracking_error_rad")
    direct_gain = _finite_values(rows, "direct_probe_improvement_rad")
    range_next = _finite_values(rows, "range_delta_next_m")
    boundary_terminal = [
        r for r in rows
        if r.get("death_or_outcome_if_terminal")
        or r.get("death_reason_if_any")
    ]
    safe_active = [int(r.get("safe_pursuit_mode_active", 0)) for r in rows]
    sources = Counter(str(r.get("desired_heading_source", "")) for r in rows)
    uses_bounds = [
        int(float(r.get("uses_red_action_bounds", 0) or 0))
        for r in rows
    ]
    fallback = [
        r for r in rows
        if str(r.get("fallback_to_delta10_reason", "")) not in {"", "not_safe_pursuit", "non_combat_safety_or_no_target"}
    ]
    roll_abs = [abs(v) for v in _finite_values(rows, "blue_roll_rad")]
    speeds = _finite_values(rows, "blue_speed_mps")
    alts = _finite_values(rows, "blue_alt_m")
    pressures = _finite_values(rows, "boundary_pressure")
    death_reasons = Counter(str(r.get("death_reason_if_any", "")) for r in rows if str(r.get("death_reason_if_any", "")))
    outcomes = Counter(str(r.get("episode_outcome", "")) for r in rows if str(r.get("episode_outcome", "")))
    missiles = _finite_values(rows, "missile_fired_this_step")
    out = [
        {"metric": "samples", "value": total},
        {"metric": "combat_rate", "value": branch_counts.get("combat", 0) / total},
        {"metric": "combat_or_target_visible_rate", "value": sources.get("current_target", 0) / total},
        {"metric": "simple_reacquire_rate", "value": sources.get("reacquire_last_seen", 0) / total},
        {"metric": "center_cruise_rate", "value": sources.get("center_cruise", 0) / total},
        {"metric": "hold_heading_rate", "value": sources.get("hold_heading", 0) / total},
        {"metric": "uses_red_action_bounds_rate", "value": sum(uses_bounds) / total},
        {"metric": "safe_pursuit_active_rate", "value": sum(safe_active) / total},
        {"metric": "fallback_to_delta10_rate", "value": len(fallback) / total},
        {"metric": "branch_counts", "value": dict(branch_counts)},
        {"metric": "high_heading_error_count_gt_90deg", "value": len(high)},
        {"metric": "high_error_cmd_saturation_rate", "value": len(sat_high) / len(high) if high else 0.0},
        {"metric": "high_error_range_increase_rate_next", "value": sum(v > 0.0 for v in high_range_next) / len(high_range_next) if high_range_next else ""},
        {"metric": "high_error_range_delta_next_mean_m", "value": float(np.mean(high_range_next)) if high_range_next else ""},
        {"metric": "high_error_actual_heading_delta_abs_mean_rad", "value": float(np.mean([abs(v) for v in high_heading_delta])) if high_heading_delta else ""},
        {"metric": "high_error_roll_abs_mean_rad", "value": float(np.mean(high_roll_abs)) if high_roll_abs else ""},
        {"metric": "command_tracking_error_mean_rad", "value": float(np.mean(tracking)) if tracking else ""},
        {"metric": "command_tracking_error_p90_rad", "value": float(np.percentile(tracking, 90)) if tracking else ""},
        {"metric": "range_delta_next_mean_m", "value": float(np.mean(range_next)) if range_next else ""},
        {"metric": "range_increase_rate_next", "value": sum(v > 0.0 for v in range_next) / len(range_next) if range_next else ""},
        {"metric": "direct_probe_improvement_mean_rad", "value": float(np.mean(direct_gain)) if direct_gain else ""},
        {"metric": "terminal_or_death_rows", "value": len(boundary_terminal)},
        {"metric": "blue_roll_abs_mean", "value": float(np.mean(roll_abs)) if roll_abs else ""},
        {"metric": "blue_roll_abs_max", "value": float(np.max(roll_abs)) if roll_abs else ""},
        {"metric": "blue_roll_abs_p95", "value": float(np.percentile(roll_abs, 95)) if roll_abs else ""},
        {"metric": "high_roll_rate_gt_75deg", "value": sum(1 for r in rows if int(r.get("high_roll_active", 0)) == 1) / total},
        {"metric": "extreme_roll_rate_gt_105deg", "value": sum(1 for r in rows if int(r.get("extreme_roll_active", 0)) == 1) / total},
        {"metric": "roll_recovery_rate", "value": sum(1 for r in rows if int(r.get("roll_recovery_active", 0)) == 1) / total},
        {"metric": "extreme_roll_recovery_rate", "value": sum(1 for r in rows if int(r.get("extreme_roll_recovery_active", 0)) == 1) / total},
        {"metric": "blue_speed_min", "value": float(np.min(speeds)) if speeds else ""},
        {"metric": "blue_speed_mean", "value": float(np.mean(speeds)) if speeds else ""},
        {"metric": "blue_speed_min_during_high_roll", "value": float(np.min([float(r.get("blue_speed_mps", np.inf)) for r in rows if int(r.get("high_roll_active", 0)) == 1])) if any(int(r.get("high_roll_active", 0)) == 1 for r in rows) else ""},
        {"metric": "high_roll_sources_counts", "value": dict(Counter(str(r.get("desired_heading_source", "")) for r in rows if int(r.get("high_roll_active", 0)) == 1))},
        {"metric": "blue_alt_min", "value": float(np.min(alts)) if alts else ""},
        {"metric": "blue_alt_mean", "value": float(np.mean(alts)) if alts else ""},
        {"metric": "boundary_pressure_max", "value": float(np.max(pressures)) if pressures else ""},
        {"metric": "blue_death_count_by_reason", "value": dict(death_reasons)},
        {"metric": "blue_boundary_death_count", "value": sum(v for k, v in death_reasons.items() if "boundary" in k.lower() or "out" in k.lower())},
        {"metric": "blue_low_altitude_death_count", "value": sum(v for k, v in death_reasons.items() if "alt" in k.lower() or "crash" in k.lower())},
        {"metric": "blue_missiles_fired_mean", "value": float(np.mean(missiles)) if missiles else ""},
        {"metric": "blue_missile_hits_mean", "value": "unavailable"},
        {"metric": "blue_missile_hits_mean_unavailable_reason", "value": "audit records launch flags but does not reconstruct missile hit ownership"},
        {"metric": "episode_outcome_counts", "value": dict(outcomes)},
    ]
    for horizon in (5, 10, 20):
        hd = [abs(v) for v in _finite_values(high, f"heading_delta_{horizon}_steps_rad")]
        rd = _finite_values(high, f"range_delta_{horizon}_steps_m")
        out.extend([
            {"metric": f"high_error_heading_delta_{horizon}_steps_abs_mean_rad", "value": float(np.mean(hd)) if hd else ""},
            {"metric": f"high_error_range_delta_{horizon}_steps_mean_m", "value": float(np.mean(rd)) if rd else ""},
            {"metric": f"high_error_range_increase_rate_{horizon}_steps", "value": sum(v > 0.0 for v in rd) / len(rd) if rd else ""},
        ])
    return out


def _output_prefix(opponent_policy: str) -> str:
    return "blue_rule_safe_pursuit" if opponent_policy == "brma_rule_safe_pursuit" else "blue_rule_control_response"


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], config: str,
                   episodes: int, max_steps: int, red_mode: str,
                   opponent_policy: str = "brma_rule") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _output_prefix(opponent_policy)
    _write_csv(output_dir / f"{prefix}_steps.csv", rows, FIELDS)
    summary = _summary_rows(rows)
    _write_csv(output_dir / f"{prefix}_summary.csv", summary, ["metric", "value"])
    _write_report(output_dir, rows, summary, config, episodes, max_steps, red_mode, opponent_policy)


def _write_report(output_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]],
                  config: str, episodes: int, max_steps: int, red_mode: str,
                  opponent_policy: str = "brma_rule") -> None:
    s = {row["metric"]: row["value"] for row in summary}
    high = [
        r for r in rows
        if r.get("heading_error_to_target_rad") != ""
        and float(r["heading_error_to_target_rad"]) > math.radians(90.0)
    ]
    sample = high[0] if high else None
    classification = []
    classification.append("A decision layer: not confirmed by this audit unless target_idx/cruise issues appear.")
    if float(s.get("direct_probe_improvement_mean_rad") or 0.0) > math.radians(20.0):
        classification.append("B command layer: delta10 limit materially slows large-angle heading target convergence.")
    if float(s.get("command_tracking_error_mean_rad") or 0.0) > math.radians(45.0):
        classification.append("C execution layer: aircraft yaw/track lags the commanded heading target.")
    lines = [
        "# Blue Rule Safe Pursuit Audit" if opponent_policy == "brma_rule_safe_pursuit" else "# Blue Rule Control Response Audit",
        "",
        f"- config: `{config}`",
        f"- opponent_policy: `{opponent_policy}`",
        f"- episodes: {episodes}",
        f"- max_steps: {max_steps}",
        f"- red_mode: `{red_mode}`",
        "- no training was run.",
        "- no reward, missile, PID, aircraft XML, red policy, action space, or observation dimension was modified.",
        "",
        "## Summary",
    ]
    for row in summary:
        lines.append(f"- {row['metric']}: {row['value']}")
    lines.extend([
        "",
        "## Classification",
        *[f"- {item}" for item in classification],
        "",
        "## Safe-Pursuit Assessment",
        f"- safe_pursuit_active_rate: {s.get('safe_pursuit_active_rate')}",
        f"- fallback_to_delta10_rate: {s.get('fallback_to_delta10_rate')}",
        f"- combat_or_target_visible_rate: {s.get('combat_or_target_visible_rate')}",
        f"- simple_reacquire_rate: {s.get('simple_reacquire_rate')}",
        f"- center_cruise_rate: {s.get('center_cruise_rate')}",
        f"- hold_heading_rate: {s.get('hold_heading_rate')}",
        f"- uses_red_action_bounds_rate: {s.get('uses_red_action_bounds_rate')}",
        "- safe-pursuit combat heading output uses the same normalized `action[1]` bounds as red policy actions: [-1, 1].",
        "- safe-pursuit maps `target_heading_abs / pi` directly; no 20/35/60/180 deg staged limiter remains.",
        "- safe-pursuit no longer uses lead pursuit, target velocity, TA-derived target heading, complex target scoring, heading hysteresis, bank damping, G compensation, or multi-condition throttle.",
        "- target selection is nearest valid red slot, with only one-step simple deconfliction across blue aircraft.",
        f"- short lost-target reacquire window: 15 steps.",
        "- safety layers still execute before combat: hard deck, descent safety, stall protection, anti-stall, and boundary safety.",
        f"- blue_death_count_by_reason: {s.get('blue_death_count_by_reason')}",
        f"- blue_boundary_death_count: {s.get('blue_boundary_death_count')}",
        f"- blue_low_altitude_death_count: {s.get('blue_low_altitude_death_count')}",
        f"- blue_roll_abs_max: {s.get('blue_roll_abs_max')}",
        f"- blue_roll_abs_p95: {s.get('blue_roll_abs_p95')}",
        f"- high_roll_rate_gt_75deg: {s.get('high_roll_rate_gt_75deg')}",
        f"- extreme_roll_rate_gt_105deg: {s.get('extreme_roll_rate_gt_105deg')}",
        f"- roll_recovery_rate: {s.get('roll_recovery_rate')}",
        f"- extreme_roll_recovery_rate: {s.get('extreme_roll_recovery_rate')}",
        f"- blue_speed_min: {s.get('blue_speed_min')}",
        f"- blue_speed_min_during_high_roll: {s.get('blue_speed_min_during_high_roll')}",
        f"- high_roll_sources: {s.get('high_roll_sources_counts')}",
        "- Recommendation: use this mode as an opt-in opponent probe. The default `brma_rule` remains the legacy delta10 script.",
        "",
        "## BRMA-MAPPO Action Contract Note",
        "- The environment action contract uses `action[1] * pi` as an absolute target heading, so `action[1] = -1/0/+1` maps to `-pi/0/+pi` rad.",
        "- Red policy actions enter this absolute heading contract without an additional delta-heading limiter in `env.py`.",
        "- The default `brma_rule` first computes an internal heading delta and limits it to +/-10 deg per decision step, then converts it back to an absolute target heading.",
        "- The +/-10 deg limiter is a project legacy autopilot limiter, not a paper-required BRMA-MAPPO action-space constraint.",
        "- `brma_rule_safe_pursuit` is a separate simple-pursuit path: nearest valid target, current AO bearing, 15-step last-seen reacquire, then center cruise or hold heading.",
        "- `direct_lead_heading_probe` is diagnostic only; it is not used by training or evaluation.",
        "",
        "## Representative High-Error Sample",
    ])
    if sample:
        keys = [
            "episode", "step", "blue_id", "selected_red_id",
            "heading_error_to_target_rad", "heading_cmd_internal",
            "target_heading_cmd_rad", "blue_yaw_rad",
            "actual_heading_next_rad", "actual_heading_delta_next_rad",
            "range_delta_next_m", "command_tracking_error_rad",
            "blue_roll_rad", "blue_speed_mps",
        ]
        for key in keys:
            lines.append(f"- {key}: {sample.get(key, '')}")
    else:
        lines.append("- no heading_error > 90 deg sample was observed.")
    lines.extend([
        "",
        "## Outputs",
        f"- step CSV: `{_output_prefix(opponent_policy)}_steps.csv`",
        f"- summary CSV: `{_output_prefix(opponent_policy)}_summary.csv`",
    ])
    _write(output_dir / f"{_output_prefix(opponent_policy)}_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs/blue_rule_control_response_audit")
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["brma_rule", "brma_rule_safe_pursuit"])
    parser.add_argument("--blue-rule-mode", default=None, choices=["safe_pursuit"])
    parser.add_argument("--red-mode", default="zero", choices=["zero", "straight_outward"])
    args = parser.parse_args()
    opponent_policy = args.opponent_policy
    if args.blue_rule_mode == "safe_pursuit":
        opponent_policy = "brma_rule_safe_pursuit"
    out = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    run_audit(args.config, args.episodes, args.max_steps, out, args.red_mode, opponent_policy)
    print(out)


if __name__ == "__main__":
    main()
