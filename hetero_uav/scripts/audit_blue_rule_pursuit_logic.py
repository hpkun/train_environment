"""Audit blue brma_rule pursuit/reacquisition behavior.

This script is read-only with respect to environment mechanics. It runs short
scripted rollouts, calls the same OpponentPolicy("brma_rule") path used by
train/eval, and records target selection, branch inference and heading
alignment diagnostics for blue aircraft.
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
from rule_based_agent import (  # noqa: E402
    HARD_DECK,
    DESCENT_WARN_ALT,
    MAX_DESCENT_RATE,
    SAFE_COMBAT_ALT,
    _STALL_PROTECT_ALT,
    _STALL_SPEED,
    _boundary_outward_heading_component,
    _boundary_patrol_pressure,
    _should_override_for_boundary_safety,
    _target_selection_score,
    _target_track_quality,
)
import rule_based_agent as _blue_rule_module  # noqa: E402
from uav_env import make_env  # noqa: E402


DEFAULT_CONFIG = "uav_env/JSBSim/configs/diagnostic_mav_shared_geo_3v2.yaml"


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
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _own_heading(obs: dict, own_heading_from_env: float | None) -> tuple[float, bool]:
    if own_heading_from_env is not None and np.isfinite(float(own_heading_from_env)):
        return float(own_heading_from_env), True
    vel = np.asarray(obs.get("velocity", []), dtype=np.float32).reshape(-1)
    if vel.size >= 2 and np.linalg.norm(vel[:2]) > 1e-6:
        return float(math.atan2(float(vel[1]), float(vel[0]))), False
    return 0.0, False


def _alive_red_indices(obs: dict, num_blue: int, num_red: int) -> list[int]:
    death_mask = np.asarray(obs.get("death_mask", []), dtype=np.float32).reshape(-1)
    return [
        i for i in range(num_red)
        if death_mask.size > num_blue + i and death_mask[num_blue + i] > 0.5
    ]


def _select_target_debug(
    obs: dict,
    num_blue: int,
    num_red: int,
    forced_target_idx: int | None = None,
) -> dict[str, Any]:
    enemy_states = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
    if enemy_states.ndim != 2:
        enemy_states = np.zeros((0, 0), dtype=np.float32)
    alive_reds = _alive_red_indices(obs, num_blue, num_red)
    target_idx = None
    target_score = 0.0
    reason = ""
    quality_by_idx = {}
    score_by_idx = {}
    zero_by_idx = {}
    for idx in alive_reds:
        state = enemy_states[idx] if idx < enemy_states.shape[0] else np.zeros(0, dtype=np.float32)
        quality = _target_track_quality(state)
        score = _target_selection_score(state)
        quality_by_idx[idx] = quality
        score_by_idx[idx] = score
        zero_by_idx[idx] = bool(state.size == 0 or np.allclose(state, 0.0))
    if forced_target_idx is not None and forced_target_idx in alive_reds:
        state = enemy_states[forced_target_idx] if forced_target_idx < enemy_states.shape[0] else np.zeros(0, dtype=np.float32)
        if _target_track_quality(state) != "invalid":
            target_idx = forced_target_idx
            target_score = float(_target_selection_score(state))
    if target_idx is None:
        best = 0.0
        for idx in alive_reds:
            score = float(score_by_idx.get(idx, 0.0))
            if score > best:
                best = score
                target_idx = idx
                target_score = score
    if not alive_reds:
        reason = "all_red_dead"
    elif target_idx is None:
        invalid = [idx for idx in alive_reds if quality_by_idx.get(idx) == "invalid"]
        reason = "all_alive_red_tracks_invalid" if len(invalid) == len(alive_reds) else "all_scores_zero"
    state = enemy_states[target_idx] if target_idx is not None and target_idx < enemy_states.shape[0] else None
    quality = _target_track_quality(state) if state is not None else "none"
    return {
        "target_idx": target_idx,
        "target_quality": quality,
        "target_score": target_score,
        "reason_if_no_target": reason,
        "alive_reds": alive_reds,
        "quality_by_idx": quality_by_idx,
        "score_by_idx": score_by_idx,
        "zero_by_idx": zero_by_idx,
        "target_state": state,
    }


def _coordinated_assignment_debug(
    obs: dict[str, dict],
    blue_ids: list[str],
    num_blue: int,
    num_red: int,
    engaged_targets: set[str] | None,
) -> dict[str, int | None]:
    if not blue_ids:
        return {}
    engaged_red_indices: set[int] = set()
    if engaged_targets:
        for uid in engaged_targets:
            if uid.startswith("red_"):
                try:
                    engaged_red_indices.add(int(uid.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    first_obs = obs[blue_ids[0]]
    alive_reds = _alive_red_indices(first_obs, num_blue, num_red)
    if not alive_reds:
        return {bid: None for bid in blue_ids}
    score = np.zeros((num_blue, num_red), dtype=np.float32)
    for b_idx, bid in enumerate(blue_ids):
        enemy_states = np.asarray(obs[bid].get("enemy_states", []), dtype=np.float32)
        for r_idx in alive_reds:
            if r_idx in engaged_red_indices:
                score[b_idx, r_idx] = -1.0
                continue
            state = enemy_states[r_idx] if r_idx < enemy_states.shape[0] else np.zeros(0, dtype=np.float32)
            score[b_idx, r_idx] = _target_selection_score(state)
    blue_best_score = np.max(score, axis=1, initial=0.0)
    blue_order = sorted(range(num_blue), key=lambda i: blue_best_score[i], reverse=True)
    taken_reds: set[int] = set()
    assignments: dict[str, int | None] = {}
    for b_idx in blue_order:
        best_r = None
        best_s = 0.0
        for r_idx in alive_reds:
            if r_idx in taken_reds or r_idx in engaged_red_indices:
                continue
            if score[b_idx, r_idx] > best_s:
                best_s = float(score[b_idx, r_idx])
                best_r = r_idx
        bid = blue_ids[b_idx]
        assignments[bid] = best_r
        if best_r is not None:
            taken_reds.add(best_r)
            engaged_red_indices.add(best_r)
    return {bid: assignments.get(bid) for bid in blue_ids}


def _infer_branch(
    obs: dict,
    target_idx: int | None,
    own_position: np.ndarray | None,
    own_heading: float,
) -> str:
    altitude = float(np.asarray(obs.get("altitude", [0.0]), dtype=np.float32).reshape(-1)[0])
    velocity = np.asarray(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    ego_state = np.asarray(obs.get("ego_state", []), dtype=np.float32).reshape(-1)
    ego_vel = float(ego_state[6]) * 600.0 if ego_state.size > 6 else 0.0
    ego_pitch = math.atan2(float(ego_state[9]), float(ego_state[10])) if ego_state.size > 10 else 0.0
    v_up = float(velocity[2]) if velocity.size > 2 else 0.0
    if altitude < HARD_DECK:
        return "hard_deck"
    if altitude < DESCENT_WARN_ALT and v_up < -MAX_DESCENT_RATE:
        return "descent_safety"
    if altitude < _STALL_PROTECT_ALT and ego_vel < _STALL_SPEED:
        return "stall_protect"
    if _should_override_for_boundary_safety(own_position, own_heading):
        return "boundary_override"
    if ego_vel < 250.0 and ego_pitch > 0.0:
        return "anti_stall"
    if target_idx is not None:
        return "combat"
    return "cruise_or_reacquire"


def _target_geometry(env, blue_id: str, target_idx: int | None) -> dict[str, Any]:
    if target_idx is None or target_idx >= len(env.red_ids):
        return {
            "selected_red_id": "",
            "target_range_m": "",
            "red_target_pos_n": "",
            "red_target_pos_e": "",
            "red_target_alt_m": "",
            "nearest_red_range_m": _nearest_red_range(env, blue_id),
            "heading_error_to_target_rad": "",
        }
    bid_sim = env.blue_planes[blue_id]
    rid = env.red_ids[target_idx]
    red_sim = env.red_planes[rid]
    bpos = np.asarray(bid_sim.get_position(), dtype=np.float64)
    rpos = np.asarray(red_sim.get_position(), dtype=np.float64)
    delta = rpos - bpos
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    heading = float(bid_sim.get_rpy()[2])
    return {
        "selected_red_id": rid,
        "target_range_m": float(np.linalg.norm(delta)),
        "red_target_pos_n": float(rpos[0]),
        "red_target_pos_e": float(rpos[1]),
        "red_target_alt_m": float(rpos[2]),
        "nearest_red_range_m": _nearest_red_range(env, blue_id),
        "heading_error_to_target_rad": abs(_wrap_pi(bearing - heading)),
    }


def _nearest_red_range(env, blue_id: str) -> float:
    blue = env.blue_planes[blue_id]
    bpos = np.asarray(blue.get_position(), dtype=np.float64)
    vals = []
    for red in env.red_planes.values():
        if red is not None and red.is_alive:
            vals.append(float(np.linalg.norm(np.asarray(red.get_position(), dtype=np.float64) - bpos)))
    return min(vals) if vals else float("nan")


def _red_actions(env, mode: str) -> dict[str, np.ndarray]:
    actions = {}
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        if sim is None or not sim.is_alive:
            actions[rid] = np.zeros(3, dtype=np.float32)
            continue
        if mode == "straight_outward":
            pos = np.asarray(sim.get_position(), dtype=np.float64)
            heading = math.atan2(float(pos[1]), float(pos[0])) / math.pi if np.linalg.norm(pos[:2]) > 1e-6 else float(sim.get_rpy()[2]) / math.pi
            actions[rid] = np.asarray([0.0, heading, 0.6], dtype=np.float32)
        else:
            actions[rid] = np.zeros(3, dtype=np.float32)
    return actions


def _launch_flags(info: dict, blue_id: str) -> tuple[int, int, int, int, int]:
    fired = int(info.get(blue_id, {}).get("missiles_fired_this_step", 0) > 0)
    diag = info.get("__launch_diag__", {}).get("blue", {})
    return (
        fired,
        int(diag.get("track_ok_pairs", 0) > 0 or diag.get("direct_track_candidates", 0) > 0),
        int(diag.get("geometry_ok_pairs", 0) > 0),
        int(diag.get("range_ok_pairs", 0) > 0),
        int(diag.get("ao_ok_pairs", 0) > 0),
    )


def run_audit(config: str, episodes: int, max_steps: int, output_dir: Path,
              red_mode: str, export_acmi: bool = False) -> None:
    del export_acmi
    rows: list[dict[str, Any]] = []
    opponent = OpponentPolicy("brma_rule", seed=17)
    for ep in range(episodes):
        env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
        try:
            obs, info = env.reset(seed=ep)
            opponent.reset_memory()
            _blue_rule_module._last_target_bearing.clear()
            _blue_rule_module._lost_target_steps.clear()
            _blue_rule_module._prev_heading_cmd.clear()
            _blue_rule_module._prev_lead_bearing.clear()
            for step in range(max_steps):
                engaged_targets = env.refresh_engaged_targets()
                assignments = _coordinated_assignment_debug(
                    obs, env.blue_ids, len(env.blue_ids), len(env.red_ids), engaged_targets)
                actions_blue = opponent.act(obs, env.blue_ids, env=env)
                blue_positions = env.get_blue_own_positions()
                blue_kin = env.get_blue_own_kinematics()
                for bid in env.blue_ids:
                    bidx = int(bid.split("_")[1])
                    bobs = obs.get(bid, {})
                    own_pos = blue_positions.get(bid)
                    own_head = blue_kin.get(bid, {}).get("heading") if blue_kin else None
                    heading, heading_from_env = _own_heading(bobs, own_head)
                    target = _select_target_debug(
                        bobs, len(env.blue_ids), len(env.red_ids),
                        forced_target_idx=assignments.get(bid))
                    action = actions_blue.get(bid, np.zeros(3, dtype=np.float32))
                    target_heading = float(action[1]) * math.pi
                    heading_cmd_internal = _wrap_pi(target_heading - heading) / math.radians(10.0)
                    branch = _infer_branch(bobs, target["target_idx"], own_pos, heading)
                    geom = _target_geometry(env, bid, target["target_idx"])
                    pos = np.asarray(env.blue_planes[bid].get_position(), dtype=np.float64)
                    speed = float(np.linalg.norm(env.blue_planes[bid].get_velocity()))
                    pressure = _boundary_patrol_pressure(own_pos) if own_pos is not None else 0.0
                    outward = _boundary_outward_heading_component(own_pos, heading) if own_pos is not None else 0.0
                    state = target["target_state"]
                    rows.append({
                        "episode": ep,
                        "step": step,
                        "blue_id": bid,
                        "blue_pos_n": float(pos[0]),
                        "blue_pos_e": float(pos[1]),
                        "blue_alt_m": float(pos[2]),
                        "blue_heading_rad": heading,
                        "blue_speed_mps": speed,
                        "selected_red_id": geom["selected_red_id"],
                        "target_idx": "" if target["target_idx"] is None else int(target["target_idx"]),
                        "target_quality": target["target_quality"],
                        "target_score": float(target["target_score"]),
                        "target_range_m": geom["target_range_m"],
                        "AO_rad": float(state[3]) * math.pi if state is not None and state.size > 3 else "",
                        "TA_rad": float(state[4]) * math.pi if state is not None and state.size > 4 else "",
                        "heading_cmd_internal": float(np.clip(heading_cmd_internal, -1.0, 1.0)),
                        "target_heading_rad": target_heading,
                        "heading_error_to_target_rad": geom["heading_error_to_target_rad"],
                        "branch_state": branch,
                        "own_position_available": int(own_pos is not None),
                        "own_heading_available": int(heading_from_env),
                        "boundary_pressure": pressure,
                        "outward_component": outward,
                        "distance_to_center_m": float(np.linalg.norm(pos[:2])),
                        "nearest_red_range_m": geom["nearest_red_range_m"],
                        "red_target_pos_n": geom["red_target_pos_n"],
                        "red_target_pos_e": geom["red_target_pos_e"],
                        "red_target_alt_m": geom["red_target_alt_m"],
                        "missile_fired_this_step": 0,
                        "launch_track_ok": 0,
                        "launch_geometry_ok": 0,
                        "range_ok": 0,
                        "ao_ok": 0,
                        "ta_ok": int(state is not None and state.size > 4 and abs(float(state[4])) > 1e-4),
                        "reason_if_no_target": target["reason_if_no_target"],
                        "enemy_states_all_zero": int(
                            np.asarray(bobs.get("enemy_states", [])).size == 0
                            or np.allclose(np.asarray(bobs.get("enemy_states", []), dtype=np.float32), 0.0)
                        ),
                    })
                full_actions = {**_red_actions(env, red_mode), **actions_blue}
                obs, _rewards, terminated, truncated, info = env.step(full_actions)
                for bid in env.blue_ids:
                    flags = _launch_flags(info, bid)
                    for row in reversed(rows):
                        if row["episode"] == ep and row["step"] == step and row["blue_id"] == bid:
                            row["missile_fired_this_step"] = flags[0]
                            row["launch_track_ok"] = flags[1]
                            row["launch_geometry_ok"] = flags[2]
                            row["range_ok"] = flags[3]
                            row["ao_ok"] = flags[4]
                            break
                if all(terminated.values()) or all(truncated.values()):
                    break
        finally:
            env.close()
    _write_outputs(output_dir, rows, config, episodes, max_steps, red_mode)


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], config: str,
                   episodes: int, max_steps: int, red_mode: str) -> None:
    fields = [
        "episode", "step", "blue_id", "blue_pos_n", "blue_pos_e", "blue_alt_m",
        "blue_heading_rad", "blue_speed_mps", "selected_red_id", "target_idx",
        "target_quality", "target_score", "target_range_m", "AO_rad", "TA_rad",
        "heading_cmd_internal", "target_heading_rad", "heading_error_to_target_rad",
        "branch_state", "own_position_available", "own_heading_available",
        "boundary_pressure", "outward_component", "distance_to_center_m",
        "nearest_red_range_m", "red_target_pos_n", "red_target_pos_e",
        "red_target_alt_m", "missile_fired_this_step", "launch_track_ok",
        "launch_geometry_ok", "range_ok", "ao_ok", "ta_ok", "reason_if_no_target",
        "enemy_states_all_zero",
    ]
    _write_csv(output_dir / "blue_rule_pursuit_steps.csv", rows, fields)
    summary = _summary_rows(rows)
    _write_csv(output_dir / "blue_rule_pursuit_summary.csv", summary, [
        "metric", "value",
    ])
    _write_report(output_dir, rows, summary, config, episodes, max_steps, red_mode)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = max(len(rows), 1)
    branch = Counter(str(r["branch_state"]) for r in rows)
    no_target = sum(1 for r in rows if r["target_idx"] == "")
    invalid = Counter(str(r["reason_if_no_target"]) for r in rows if r["target_idx"] == "")
    nearest = [float(r["nearest_red_range_m"]) for r in rows if str(r["nearest_red_range_m"]) != "" and np.isfinite(float(r["nearest_red_range_m"]))]
    center = [float(r["distance_to_center_m"]) for r in rows if np.isfinite(float(r["distance_to_center_m"]))]
    heading_err = [float(r["heading_error_to_target_rad"]) for r in rows if str(r["heading_error_to_target_rad"]) != ""]
    heading_cmd_abs = [
        abs(float(r["heading_cmd_internal"]))
        for r in rows
        if str(r["heading_cmd_internal"]) != ""
    ]
    high_heading = [
        r for r in rows
        if str(r["heading_error_to_target_rad"]) != ""
        and float(r["heading_error_to_target_rad"]) > math.radians(90.0)
    ]
    out = [
        {"metric": "samples", "value": total},
        {"metric": "target_idx_none_rate", "value": no_target / total},
        {"metric": "combat_rate", "value": branch.get("combat", 0) / total},
        {"metric": "cruise_or_reacquire_rate", "value": branch.get("cruise_or_reacquire", 0) / total},
        {"metric": "boundary_override_rate", "value": branch.get("boundary_override", 0) / total},
        {"metric": "own_heading_available_rate", "value": sum(int(r["own_heading_available"]) for r in rows) / total},
        {"metric": "own_position_available_rate", "value": sum(int(r["own_position_available"]) for r in rows) / total},
        {"metric": "nearest_red_range_mean_m", "value": float(np.mean(nearest)) if nearest else ""},
        {"metric": "nearest_red_range_max_m", "value": float(np.max(nearest)) if nearest else ""},
        {"metric": "distance_to_center_mean_m", "value": float(np.mean(center)) if center else ""},
        {"metric": "distance_to_center_max_m", "value": float(np.max(center)) if center else ""},
        {"metric": "heading_error_to_target_mean_rad", "value": float(np.mean(heading_err)) if heading_err else ""},
        {"metric": "heading_error_to_target_max_rad", "value": float(np.max(heading_err)) if heading_err else ""},
        {"metric": "heading_cmd_abs_mean", "value": float(np.mean(heading_cmd_abs)) if heading_cmd_abs else ""},
        {"metric": "heading_cmd_saturation_rate_abs_ge_0_99", "value": sum(v >= 0.99 for v in heading_cmd_abs) / len(heading_cmd_abs) if heading_cmd_abs else ""},
        {"metric": "high_heading_error_count_gt_90deg", "value": len(high_heading)},
        {"metric": "high_heading_error_cmd_saturation_rate", "value": sum(abs(float(r["heading_cmd_internal"])) >= 0.99 for r in high_heading) / len(high_heading) if high_heading else 0.0},
        {"metric": "no_target_reason_counts", "value": dict(invalid)},
        {"metric": "branch_counts", "value": dict(branch)},
    ]
    return out


def _write_report(output_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]],
                  config: str, episodes: int, max_steps: int, red_mode: str) -> None:
    summary_map = {r["metric"]: r["value"] for r in summary}
    no_target_alive_invalid = [
        r for r in rows
        if r["target_idx"] == "" and r["reason_if_no_target"] != "all_red_dead"
    ]
    high_heading_error = [
        r for r in rows
        if str(r["heading_error_to_target_rad"]) != "" and float(r["heading_error_to_target_rad"]) > math.radians(90)
    ]
    lines = [
        "# Blue BRMA Rule Pursuit Audit",
        "",
        f"- config: `{config}`",
        f"- episodes: {episodes}",
        f"- max_steps: {max_steps}",
        f"- red_mode: `{red_mode}`",
        "- no training was run.",
        "",
        "## Evidence Summary",
        f"- samples: {summary_map.get('samples')}",
        f"- combat branch rate: {summary_map.get('combat_rate')}",
        f"- cruise/reacquire branch rate: {summary_map.get('cruise_or_reacquire_rate')}",
        f"- boundary override rate: {summary_map.get('boundary_override_rate')}",
        f"- target_idx=None rate: {summary_map.get('target_idx_none_rate')}",
        f"- own_heading_available_rate: {summary_map.get('own_heading_available_rate')}",
        f"- own_position_available_rate: {summary_map.get('own_position_available_rate')}",
        f"- nearest red range mean/max m: {summary_map.get('nearest_red_range_mean_m')} / {summary_map.get('nearest_red_range_max_m')}",
        f"- center distance mean/max m: {summary_map.get('distance_to_center_mean_m')} / {summary_map.get('distance_to_center_max_m')}",
        f"- heading error to selected target mean/max rad: {summary_map.get('heading_error_to_target_mean_rad')} / {summary_map.get('heading_error_to_target_max_rad')}",
        f"- heading command abs mean: {summary_map.get('heading_cmd_abs_mean')}",
        f"- heading command saturation rate abs>=0.99: {summary_map.get('heading_cmd_saturation_rate_abs_ge_0_99')}",
        f"- high heading error samples >90 deg: {summary_map.get('high_heading_error_count_gt_90deg')}",
        f"- high heading error command saturation rate: {summary_map.get('high_heading_error_cmd_saturation_rate')}",
        f"- no-target while red alive or track invalid samples: {len(no_target_alive_invalid)}",
        f"- heading error > 90 deg samples: {len(high_heading_error)}",
        "",
        "## BRMA-MAPPO Alignment",
        "- Blue is a fixed rule policy, not a learned policy.",
        "- Target selection is based on BRMA-style `enemy_states` geometry: range R, AO, TA and death_mask.",
        "- The action contract remains `[target_pitch, target_heading, target_velocity]`; rule_based_agent converts an internal delta-heading into the env absolute heading command.",
        "- Extra JSBSim safety layers exist: hard deck, descent safety, stall protection, anti-stall and boundary override. These are engineering safety guards and may override pursuit in extreme states.",
        "- If pursuit fails, it should be attributed to implementation details such as observation normalization, coordinate conversion, safety branch precedence or PID/JSBSim behavior, not to the paper opponent definition itself.",
        "",
        "## Files",
        "- step CSV: `blue_rule_pursuit_steps.csv`",
        "- summary CSV: `blue_rule_pursuit_summary.csv`",
    ]
    _write(output_dir / "blue_rule_pursuit_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs/blue_rule_pursuit_audit")
    parser.add_argument("--opponent-policy", default="brma_rule", choices=["brma_rule"])
    parser.add_argument("--red-mode", default="zero", choices=["zero", "straight_outward"])
    parser.add_argument("--export-acmi", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit(args.config, args.episodes, args.max_steps, output_dir, args.red_mode, args.export_acmi)
    print(output_dir)


if __name__ == "__main__":
    main()
