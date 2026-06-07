"""Audit missile launch/hit contract for the F-22 MAV mainline environment.

Diagnostic only: no training, no model saving, and no changes to missile,
reward, termination, action, evasion, PID, aircraft XML, or MAPPO parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env
from uav_env.JSBSim.env import UavCombatEnv


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _source_text() -> str:
    return (ROOT / "uav_env/JSBSim/env.py").read_text(encoding="utf-8", errors="replace")


def _mainline_metadata(config: str) -> dict:
    env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    try:
        _obs, _info = env.reset(seed=0)
        models = dict(getattr(env, "agent_models", {}))
        types = dict(getattr(env, "agent_types", {}))
        missiles = {
            aid: int(env._get_sim(aid).num_left_missiles)
            for aid in env.agent_ids
            if env._get_sim(aid) is not None
        }
        attack_ids = [aid for aid, type_name in types.items() if type_name == "attack_uav"]
        return {
            "mav_model": models.get("red_0"),
            "mav_num_missiles": missiles.get("red_0"),
            "attack_uav_model": sorted({models.get(aid) for aid in attack_ids})[0],
            "attack_uav_num_missiles": sorted({missiles.get(aid) for aid in attack_ids})[0],
        }
    finally:
        env.close()


def build_static_contract(configs: list[str]) -> dict:
    src = _source_text()
    metadata = _mainline_metadata(configs[0])
    return {
        "launch_range_max_m": float(UavCombatEnv.MISSILE_LAUNCH_RANGE_THRESH),
        "launch_range_min_m": float(UavCombatEnv.MISSILE_LAUNCH_MIN_RANGE),
        "launch_ao_thresh_deg": float(np.rad2deg(UavCombatEnv.MISSILE_LAUNCH_AO_THRESH)),
        "launch_ta_thresh_deg": float(np.rad2deg(UavCombatEnv.MISSILE_LAUNCH_TA_THRESH)),
        "missile_lock_delay_sec": float(UavCombatEnv.MISSILE_LOCK_DELAY_FRAMES / 60.0),
        "missile_cooldown_sec": float(UavCombatEnv.MISSILE_COOLDOWN_STEPS / 60.0),
        "kill_cooldown_steps": int(UavCombatEnv.KILL_COOLDOWN_STEPS),
        "uses_lock_timer": "_lock_timer" in src,
        "uses_lock_target": "_lock_target" in src,
        "uses_engaged_target_deconfliction": "_engaged_targets" in src and "engaged_blocked" in src,
        "uses_same_frame_hot_update": "self._engaged_targets.add(best_enemy.uid)" in src,
        "uses_target_rear_hemisphere": "TA > self.MISSILE_LAUNCH_TA_THRESH" in src,
        "uses_2d_range_or_3d_range": "2d_range_from_get2d_AO_TA_R",
        "launch_quality_records_available": "_launch_quality_records" in src,
        "missile_term_reason_records_available": "_missile_term_reasons" in src,
        **metadata,
    }


def _bounded_red_actions(env, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        aid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        for aid in env.red_ids
    }


def _empty_diag() -> dict:
    fields = [
        "launches",
        "range_ok_pairs",
        "ao_ok_pairs",
        "ta_ok_pairs",
        "geometry_ok_pairs",
        "lock_started",
        "lock_mature_pairs",
        "cooldown_blocked",
        "engaged_blocked",
    ]
    out = {}
    for team in ("red", "blue"):
        for field in fields:
            out[f"{team}_{field}"] = 0
    return out


def _add_launch_diag(acc: dict, info: dict) -> None:
    launch_diag = info.get("__launch_diag__", {})
    for team in ("red", "blue"):
        team_diag = launch_diag.get(team, {})
        for key, value in team_diag.items():
            out_key = f"{team}_{key}"
            if out_key in acc:
                acc[out_key] += int(value)


def _add_quality(records: list[dict], info: dict) -> None:
    records.extend(info.get("__launch_quality_step__", []))
    records.extend(info.get("__launch_quality_done__", []))


def _team_term(info: dict, team: str) -> dict:
    return dict(info.get("__missile_term__", {}).get(team, {}))


def _hit_miss(term: dict) -> tuple[int, int]:
    hits = int(term.get("hit", 0))
    misses = sum(int(v) for k, v in term.items() if k != "hit")
    return hits, misses


def _mean_field(records: list[dict], field: str) -> float | None:
    vals = []
    for record in records:
        value = record.get(field, "")
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return None
    return float(np.mean(vals))


def rollout_config(config: str, steps: int, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    acc = _empty_diag()
    all_quality: list[dict] = []
    nan_detected = False
    final_info = {}
    ammo_remaining: dict[str, int] = {}
    total_steps = 0
    for ep in range(episodes):
        env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
        opponent = OpponentPolicy("greedy_fsm", seed=seed + ep + 17)
        try:
            obs, info = env.reset(seed=seed + ep)
            nan_detected = nan_detected or _contains_nan(obs)
            for _step in range(steps):
                actions = _bounded_red_actions(env, rng)
                actions.update(opponent.act(obs, env.blue_ids, env=env))
                obs, rewards, terminated, truncated, info = env.step(actions)
                total_steps += 1
                final_info = info
                nan_detected = nan_detected or _contains_nan(obs) or _contains_nan(rewards)
                _add_launch_diag(acc, info)
                _add_quality(all_quality, info)
                if all(terminated.values()) or all(truncated.values()):
                    break
            for aid in env.agent_ids:
                agent_info = final_info.get(aid, {})
                if isinstance(agent_info, dict):
                    ammo_remaining[aid] = int(agent_info.get("missiles_left", 0))
        finally:
            env.close()

    red_term = _team_term(final_info, "red")
    blue_term = _team_term(final_info, "blue")
    red_hits, red_misses = _hit_miss(red_term)
    blue_hits, blue_misses = _hit_miss(blue_term)
    red_launches = acc["red_launches"]
    blue_launches = acc["blue_launches"]
    return {
        "config": config,
        "episodes": episodes,
        "steps": total_steps,
        "red_launches": red_launches,
        "blue_launches": blue_launches,
        "red_hits": red_hits,
        "blue_hits": blue_hits,
        "red_misses": red_misses,
        "blue_misses": blue_misses,
        "red_hit_rate": float(red_hits / red_launches) if red_launches else 0.0,
        "blue_hit_rate": float(blue_hits / blue_launches) if blue_launches else 0.0,
        "red_range_ok_pairs": acc["red_range_ok_pairs"],
        "blue_range_ok_pairs": acc["blue_range_ok_pairs"],
        "red_ao_ok_pairs": acc["red_ao_ok_pairs"],
        "blue_ao_ok_pairs": acc["blue_ao_ok_pairs"],
        "red_ta_ok_pairs": acc["red_ta_ok_pairs"],
        "blue_ta_ok_pairs": acc["blue_ta_ok_pairs"],
        "red_geometry_ok_pairs": acc["red_geometry_ok_pairs"],
        "blue_geometry_ok_pairs": acc["blue_geometry_ok_pairs"],
        "red_lock_started": acc["red_lock_started"],
        "blue_lock_started": acc["blue_lock_started"],
        "red_lock_mature_pairs": acc["red_lock_mature_pairs"],
        "blue_lock_mature_pairs": acc["blue_lock_mature_pairs"],
        "red_cooldown_blocked": acc["red_cooldown_blocked"],
        "blue_cooldown_blocked": acc["blue_cooldown_blocked"],
        "red_engaged_blocked": acc["red_engaged_blocked"],
        "blue_engaged_blocked": acc["blue_engaged_blocked"],
        "red_ammo_remaining_by_agent": {
            aid: value for aid, value in ammo_remaining.items() if aid.startswith("red_")
        },
        "blue_ammo_remaining_by_agent": {
            aid: value for aid, value in ammo_remaining.items() if aid.startswith("blue_")
        },
        "missile_term_reasons_red": red_term,
        "missile_term_reasons_blue": blue_term,
        "launch_quality_summary": {
            "mean_range_m": _mean_field(all_quality, "range_m"),
            "mean_AO_deg": _mean_field(all_quality, "AO_deg"),
            "mean_TA_deg": _mean_field(all_quality, "TA_deg"),
            "mean_closing_speed_mps": _mean_field(all_quality, "closing_speed_mps"),
        },
        "nan_detected": bool(nan_detected),
    }


def _blocking(static: dict, rollouts: list[dict]) -> list[str]:
    out = []
    if abs(static["launch_range_max_m"] - 10000.0) > 1e-6:
        out.append("max_range_not_10000m")
    if abs(static["missile_lock_delay_sec"] - 0.25) > 0.02:
        out.append("lock_delay_not_about_0.25s")
    if abs(static["missile_cooldown_sec"] - 0.5) > 0.02:
        out.append("cooldown_not_about_0.5s")
    if not static["uses_target_rear_hemisphere"]:
        out.append("ta_rear_hemisphere_gate_missing")
    if not static["uses_engaged_target_deconfliction"]:
        out.append("target_deconfliction_missing")
    if static["mav_num_missiles"] != 0:
        out.append("mav_missiles_not_zero")
    if static["attack_uav_num_missiles"] != 2:
        out.append("attack_uav_missiles_not_two")
    if any(record["nan_detected"] for record in rollouts):
        out.append("nan_detected")
    return out


def _markdown(data: dict) -> str:
    s = data["static_contract"]
    lines = [
        "# Missile Launch Contract Audit",
        "",
        "Purpose: inspect missile launch / hit logic for the current F-22 MAV",
        "mainline environment. This audit does not modify missile logic.",
        "",
        "## Paper Launch Conditions",
        "",
        "- 10 km electro-optical / infrared detection range",
        "- 0.25 s continuous detection / lock",
        "- 0.5 s launch interval",
        "- same-target deconfliction",
        "- rear hemisphere / 3-9 line",
        "",
        "## Current Implementation Summary",
        "",
        f"- range max/min: {s['launch_range_max_m']} / {s['launch_range_min_m']} m",
        f"- AO threshold: {s['launch_ao_thresh_deg']} deg",
        f"- TA threshold: {s['launch_ta_thresh_deg']} deg",
        f"- lock delay: {s['missile_lock_delay_sec']} s",
        f"- cooldown: {s['missile_cooldown_sec']} s",
        f"- deconfliction: {s['uses_engaged_target_deconfliction']}",
        f"- range source: {s['uses_2d_range_or_3d_range']}",
        "",
        "## Open Questions",
    ]
    lines.extend(f"- {q}" for q in data["open_questions"])
    lines.extend([
        "",
        "## Decision",
        "",
        "No missile changes are recommended unless a blocking mismatch is found.",
        f"- status: {data['status']}",
        f"- blocking_violations: {data['blocking_violations']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default="outputs/environment_audit/missile_launch_contract_audit.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/missile_launch_contract_audit.md")
    args = parser.parse_args()

    static = build_static_contract(args.configs)
    rollouts = [
        rollout_config(config, args.steps, args.episodes, args.seed)
        for config in args.configs
    ]
    blocking = _blocking(static, rollouts)
    data = {
        "static_contract": static,
        "rollout_diagnostics": rollouts,
        "open_questions": [
            "Whether AO=45deg is explicitly specified by the paper.",
            "Whether min range=500m is paper-specified or engineering protection.",
            "Current launch range uses 2D get2d_AO_TA_R range; paper may imply 3D range.",
            "No closing speed gate is present; paper requirement is not explicit.",
        ],
        "blocking_violations": blocking,
        "status": "contract_passed_with_open_questions" if not blocking else "blocking_mismatch_found",
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(data), encoding="utf-8")
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"status: {data['status']}", flush=True)
    print(f"blocking_violations: {blocking}", flush=True)


if __name__ == "__main__":
    main()
