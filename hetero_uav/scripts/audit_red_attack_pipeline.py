"""Audit whether the red attack/fire-control chain is wired in the env."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from red_attack_audit_utils import (
    DEFAULT_CONFIG,
    geometry,
    make_env,
    write_json,
    write_md,
    zero_actions,
)


def _missiles(env) -> dict[str, dict]:
    return {
        aid: {
            "role": env.agent_roles.get(aid),
            "model": env.agent_models.get(aid),
            "configured": int(env._num_missiles_for(aid)),
            "runtime": int((env.red_planes.get(aid) or env.blue_planes.get(aid)).num_left_missiles),
        }
        for aid in env.agent_ids
    }


def _obs_visibility(obs: dict, env) -> dict[str, dict]:
    out = {}
    for aid in env.red_ids:
        agent_obs = obs.get(aid, {})
        enemy_states = np.asarray(agent_obs.get("enemy_states", []), dtype=np.float32)
        enemy_geo = np.asarray(agent_obs.get("enemy_geo_states", []), dtype=np.float32)
        observed = np.asarray(agent_obs.get("enemy_observed_mask", []), dtype=np.float32)
        out[aid] = {
            "enemy_states_nonzero_rows": int(sum(not np.allclose(row, 0.0) for row in enemy_states)),
            "enemy_geo_nonzero_rows": int(sum(not np.allclose(row, 0.0) for row in enemy_geo)),
            "enemy_observed_mask_sum": float(np.sum(observed)) if observed.size else None,
            "has_mav_shared_geo": "enemy_geo_states" in agent_obs,
        }
    return out


def _static_fire_control(env) -> dict:
    source = inspect.getsource(env._check_missile_launch)
    return {
        "iterates_all_agent_ids": "for aid in self.agent_ids" in source,
        "red_blue_enemy_selection_symmetric": 'sim.color == "Blue"' in source
        and "else self.blue_planes" in source,
        "has_blue_only_branch": 'startswith("blue")' in source or '== "blue"' in source,
        "has_red_only_branch": 'startswith("red")' in source or '== "red"' in source,
        "requires_action_selected_target": "selected_target" in source,
        "uses_lock_delay": "_lock_timer" in source and "missile_lock_delay_frames" in source,
        "uses_engaged_targets": "_engaged_targets" in source,
        "uses_cooldown": "_missile_cooldown" in source,
        "launch_range_m": float(env.MISSILE_LAUNCH_RANGE_THRESH),
        "min_launch_range_m": float(env.MISSILE_LAUNCH_MIN_RANGE),
        "launch_ao_thresh_deg": float(np.degrees(env.MISSILE_LAUNCH_AO_THRESH)),
        "launch_ta_thresh_deg": float(np.degrees(env.MISSILE_LAUNCH_TA_THRESH)),
        "lock_delay_frames": int(env.missile_lock_delay_frames),
        "cooldown_frames": int(env.missile_cooldown_frames),
    }


def build_audit(config: str, steps: int) -> dict:
    env = make_env(config)
    try:
        obs, info = env.reset(seed=0)
        initial_missiles = _missiles(env)
        visibility = _obs_visibility(obs, env)
        static = _static_fire_control(env)
        red_geometries = {rid: geometry(env, rid) for rid in env.red_ids}

        launch_diag_samples = []
        last_info = info
        for _ in range(max(1, steps)):
            obs, _rewards, terminated, truncated, last_info = env.step(zero_actions(env))
            launch_diag_samples.append(last_info.get("__launch_diag__", {}))
            if all(terminated.values()) or all(truncated.values()):
                break

        runtime = {
            "post_reset_missile_counts_match_config": all(
                rec["configured"] == rec["runtime"] for rec in initial_missiles.values()
            ),
            "info_has_agent_missiles_fired_this_step": all(
                isinstance(last_info.get(aid), dict)
                and "missiles_fired_this_step" in last_info[aid]
                for aid in env.agent_ids
            ),
            "info_has_missile_term": "__missile_term__" in last_info,
            "info_has_launch_diag": "__launch_diag__" in last_info,
            "info_has_launch_quality_step": "__launch_quality_step__" in last_info,
            "last_launch_diag": last_info.get("__launch_diag__", {}),
        }

        red_uav_missiles_ok = all(
            initial_missiles[rid]["runtime"] == 2
            for rid in env.red_ids
            if env.agent_roles.get(rid) != "mav"
        )
        blue_missiles_ok = all(initial_missiles[bid]["runtime"] == 2 for bid in env.blue_ids)
        mav_unarmed = initial_missiles.get("red_0", {}).get("runtime") == 0

        blocking = []
        if not red_uav_missiles_ok:
            blocking.append("red_uav_missile_count_mismatch")
        if not static["iterates_all_agent_ids"]:
            blocking.append("fire_control_does_not_iterate_all_agents")
        if not any(v["enemy_observed_mask_sum"] for v in visibility.values() if v["enemy_observed_mask_sum"] is not None):
            blocking.append("red_observation_no_enemy_observed_at_reset")

        return {
            "config": config,
            "steps": steps,
            "missile_counts": initial_missiles,
            "red_0_mav_num_missiles_zero": bool(mav_unarmed),
            "red_uav_num_missiles_two": bool(red_uav_missiles_ok),
            "blue_num_missiles_two": bool(blue_missiles_ok),
            "static_fire_control": static,
            "runtime_checks": runtime,
            "red_observation_visibility": visibility,
            "red_initial_geometry": red_geometries,
            "red_auto_fire_logic_enabled": bool(static["iterates_all_agent_ids"]),
            "red_target_assignment_required_by_env": bool(static["requires_action_selected_target"]),
            "red_target_assignment_present": "env_selects_closest_unengaged_enemy_in_fire_control",
            "red_blue_launch_conditions_symmetric": bool(static["red_blue_enemy_selection_symmetric"]),
            "logging_fields_present_for_red": bool(runtime["info_has_agent_missiles_fired_this_step"]),
            "blocking_issues": blocking,
        }
    finally:
        env.close()


def write_report(data: dict, output_md: str) -> None:
    lines = [
        "# Red Attack Pipeline Audit",
        "",
        f"- config: `{data['config']}`",
        f"- red_0 MAV missiles zero: {data['red_0_mav_num_missiles_zero']}",
        f"- red UAV missiles two: {data['red_uav_num_missiles_two']}",
        f"- blue missiles two: {data['blue_num_missiles_two']}",
        f"- red auto-fire logic enabled: {data['red_auto_fire_logic_enabled']}",
        f"- red/blue launch condition symmetry: {data['red_blue_launch_conditions_symmetric']}",
        f"- red target assignment: {data['red_target_assignment_present']}",
        f"- logging fields present for red: {data['logging_fields_present_for_red']}",
        f"- blocking issues: {data['blocking_issues']}",
        "",
        "## Static Fire-Control",
        "```json",
        json.dumps(data["static_fire_control"], indent=2),
        "```",
        "",
        "## Red Observation Visibility",
        "```json",
        json.dumps(data["red_observation_visibility"], indent=2),
        "```",
    ]
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit red attack/fire-control chain")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/red_attack_pipeline/red_attack_pipeline.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/red_attack_pipeline/red_attack_pipeline.md",
    )
    args = parser.parse_args()
    data = build_audit(args.config, args.steps)
    out_json = write_json(args.output_json, data)
    write_report(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"blocking_issues: {data['blocking_issues']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
