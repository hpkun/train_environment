"""Visibility + initial geometry audit for paper-aligned and balanced configs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.mappo.opponent_policy import OpponentPolicy, _wrap_heading_norm

CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]


def _enemy_track_counts(obs: dict, agent_id: str) -> dict:
    """Count direct, shared, and observed tracks for one agent."""
    if agent_id not in obs:
        return {"direct": 0, "shared": 0, "observed": 0}
    o = obs[agent_id]
    # observed mask
    obs_mask = o.get("enemy_observed_mask",
                     np.zeros(o.get("enemy_states", np.zeros((1,))).shape[:1]))
    observed = int(np.sum(np.asarray(obs_mask) > 0.5))
    # track source
    src = np.asarray(o.get("enemy_track_source",
                           np.zeros((len(obs_mask), 2))), dtype=np.float32)
    direct = int(np.sum(src[:, 0] > 0.5))
    shared = int(np.sum(src[:, 1] > 0.5))
    return {"direct": direct, "shared": shared, "observed": observed}


def _alive_sims(sims: dict) -> list:
    return [sim for sim in sims.values() if sim is not None and sim.is_alive]


def _mean_speed(sims: list) -> float:
    if not sims:
        return 0.0
    return float(np.mean([np.linalg.norm(sim.get_velocity()) for sim in sims]))


def _red_blue_distances(red_sims: list, blue_sims: list) -> list[float]:
    distances = []
    for red in red_sims:
        red_pos = red.get_position()
        for blue in blue_sims:
            distances.append(float(np.linalg.norm(red_pos - blue.get_position())))
    return distances


def _mav_sim(env):
    roles = getattr(env, "agent_roles", {})
    for aid in env.red_ids:
        if roles.get(aid) == "mav":
            return env.red_planes.get(aid)
    return env.red_planes.get("red_0")


def _geometry_snapshot(env, step: int) -> dict:
    red_sims = _alive_sims(env.red_planes)
    blue_sims = _alive_sims(env.blue_planes)
    distances = _red_blue_distances(red_sims, blue_sims)
    if distances:
        min_distance = float(np.min(distances))
        mean_distance = float(np.mean(distances))
    else:
        min_distance = float("inf")
        mean_distance = float("inf")

    mav = _mav_sim(env)
    mav_alive = bool(mav is not None and mav.is_alive)
    mav_altitude = float(mav.get_geodetic()[2]) if mav is not None else 0.0
    return {
        "step": int(step),
        "min_red_blue_distance_m": min_distance,
        "mean_red_blue_distance_m": mean_distance,
        "min_blue_to_red_distance_m": min_distance,
        "min_red_to_blue_distance_m": min_distance,
        "blue_mean_speed_mps": _mean_speed(blue_sims),
        "red_mean_speed_mps": _mean_speed(red_sims),
        "mav_altitude_m": mav_altitude,
        "mav_alive": mav_alive,
        "mav_action_trim_enabled": bool(getattr(env, "action_trim_enabled", False)),
        "red_mav_shared_tracks_total": 0,
        "blue_direct_tracks_total": 0,
    }


def _trim_by_role(env) -> dict:
    out = {}
    for key, value in getattr(env, "action_trim_by_role", {}).items():
        out[str(key)] = [round(float(v), 6) for v in value]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--steps-list", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--red-policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--blue-policy", choices=["zero", "rule_nearest", "greedy_fsm"],
                        default="greedy_fsm")
    parser.add_argument("--output-json",
                        default="outputs/environment_audit/hetero_visibility_geometry.json")
    parser.add_argument("--disable-config-trim", action="store_true")
    args = parser.parse_args()

    horizons = list(args.steps_list) if args.steps_list else [args.steps]
    records = []
    for cfg_path in CONFIGS:
        for horizon in horizons:
            records.append(_diagnose_config(cfg_path, horizon, args))

    summary = _build_summary(records, horizons)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    print(f"Saved {args.output_json}")


def _diagnose_config(cfg_path: str, steps: int, args) -> dict:
        env = None
        try:
            env = make_env(cfg_path, env_type="jsbsim_hetero")
            if args.disable_config_trim and hasattr(env, "set_action_trim_enabled"):
                env.set_action_trim_enabled(False)
            rng = np.random.default_rng(args.seed)
            red_opponent = OpponentPolicy(mode="random", seed=args.seed + 99)
            blue_opponent = OpponentPolicy(mode=args.blue_policy, seed=args.seed + 100)
            obs, info = env.reset(seed=args.seed)

            sim_freq = getattr(env, "sim_freq", 60)
            agent_is = getattr(env, "agent_interaction_steps", 12)
            red_count = env.max_num_red
            blue_count = env.max_num_blue
            uav_direct = getattr(env, "uav_direct_observation_range_m", 10000.0)
            mav_range = getattr(env, "mav_observation_range_m", 80000.0)

            total_red_obs = 0
            total_blue_obs = 0
            total_red_direct = 0
            total_red_shared = 0
            total_blue_direct = 0
            steps_red_obs = 0
            steps_blue_obs = 0
            first_red = -1
            first_blue = -1
            mav_alive_any = False
            geometry_steps = [_geometry_snapshot(env, 0)]
            turn_back_heading_deltas: list[float] = []
            turn_back_heading_values: list[float] = []
            turn_back_count = 0

            nan_detected = False
            for step in range(steps):
                # red actions
                if args.red_policy == "zero":
                    red_acts = {rid: np.zeros(3, dtype=np.float32)
                                for rid in env.red_ids}
                else:
                    red_acts = red_opponent.act(obs, env.red_ids)
                blue_acts = blue_opponent.act(obs, env.blue_ids)
                if getattr(blue_opponent, "last_states", None):
                    for bid, state in blue_opponent.last_states.items():
                        if state != "turn_back" or bid not in blue_acts:
                            continue
                        action = np.asarray(blue_acts[bid], dtype=np.float32)
                        current_heading = blue_opponent._get_current_heading_norm(
                            obs.get(bid, {})
                        )
                        heading_delta = _wrap_heading_norm(
                            float(action[1]) - current_heading
                        )
                        turn_back_heading_deltas.append(abs(float(heading_delta)))
                        turn_back_heading_values.append(float(action[1]))
                        turn_back_count += 1
                actions = {**red_acts, **blue_acts}

                # count tracks before step
                step_red_obs = 0
                step_blue_obs = 0
                step_red_shared = 0
                step_blue_direct = 0
                for rid in env.red_ids:
                    tc = _enemy_track_counts(obs, rid)
                    step_red_obs += tc["observed"]
                    total_red_direct += tc["direct"]
                    total_red_shared += tc["shared"]
                    step_red_shared += tc["shared"]
                for bid in env.blue_ids:
                    tc = _enemy_track_counts(obs, bid)
                    step_blue_obs += tc["observed"]
                    total_blue_direct += tc["direct"]
                    step_blue_direct += tc["direct"]
                geometry_steps[-1]["red_mav_shared_tracks_total"] = step_red_shared
                geometry_steps[-1]["blue_direct_tracks_total"] = step_blue_direct

                total_red_obs += step_red_obs
                total_blue_obs += step_blue_obs
                if step_red_obs > 0 and first_red < 0:
                    first_red = step
                if step_blue_obs > 0 and first_blue < 0:
                    first_blue = step
                if step_red_obs > 0:
                    steps_red_obs += 1
                if step_blue_obs > 0:
                    steps_blue_obs += 1

                obs, _r, terminated, truncated, info = env.step(actions)
                geometry_steps.append(_geometry_snapshot(env, step + 1))
                for agent_obs in obs.values():
                    for value in agent_obs.values():
                        arr = np.asarray(value)
                        if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                            nan_detected = True
                mav_sim = env.red_planes.get("red_0") if red_count > 0 else None
                if mav_sim is not None and mav_sim.is_alive:
                    mav_alive_any = True

            red_alive = sum(1 for s in env.red_planes.values() if s.is_alive)
            blue_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
            mav_alive = int(mav_sim is not None and mav_sim.is_alive)
            min_series = [
                snap["min_red_blue_distance_m"] for snap in geometry_steps
                if np.isfinite(snap["min_red_blue_distance_m"])
            ]
            initial_min_distance = min_series[0] if min_series else float("inf")
            final_min_distance = min_series[-1] if min_series else float("inf")
            if min_series:
                closest_distance = float(np.min(min_series))
                closest_step = int(np.argmin(min_series))
            else:
                closest_distance = float("inf")
                closest_step = -1
            decreasing = 0
            transitions = 0
            for prev, current in zip(min_series, min_series[1:]):
                transitions += 1
                if current < prev:
                    decreasing += 1
            blue_closing_fraction = decreasing / max(transitions, 1)
            post_pass_separation = final_min_distance
            turn_back_heading_delta_mean_abs = (
                float(np.mean(turn_back_heading_deltas))
                if turn_back_heading_deltas else 0.0
            )
            mav_altitudes = [snap["mav_altitude_m"] for snap in geometry_steps]
            mav_final = geometry_steps[-1] if geometry_steps else {"mav_altitude_m": 0.0, "mav_alive": False}

            warnings = []
            if first_red < 0:
                warnings.append("red never observed enemy")
            if first_blue < 0:
                warnings.append("blue never observed enemy")
                if args.blue_policy == "greedy_fsm":
                    warnings.append("greedy_fsm search-only likely caused by no visible blue enemy tracks")
            if total_red_shared > 0 and total_blue_direct == 0:
                warnings.append("asymmetric information: red has MAV shared tracks, blue has no direct tracks")
            if first_red < 0 and first_blue < 0 and steps >= 50:
                warnings.append("initial geometry concern: no mutual observation")
            if len(min_series) > 1 and final_min_distance > initial_min_distance:
                warnings.append("red-blue minimum distance increased over horizon")
            if closest_distance > uav_direct:
                warnings.append("blue never entered direct observation range")
            if closest_distance <= uav_direct and first_blue < 0:
                warnings.append("visibility implementation concern: closest distance entered direct range without blue observation")
            if turn_back_count > 0 and post_pass_separation > uav_direct:
                warnings.append("turn_back triggered but did not reduce post-pass separation")

            rec = {
                "config": cfg_path,
                "red_count": red_count, "blue_count": blue_count,
                "sim_freq": sim_freq, "agent_interaction_steps": agent_is,
                "decision_dt": float(env.env_dt),
                "uav_direct_observation_range_m": uav_direct,
                "mav_observation_range_m": mav_range,
                "horizon_steps": steps,
                "steps_executed": steps,
                "first_step_red_observed": first_red,
                "first_step_blue_observed": first_blue,
                "red_observed_any": first_red >= 0,
                "blue_observed_any": first_blue >= 0,
                "red_observed_fraction": steps_red_obs / max(steps, 1),
                "blue_observed_fraction": steps_blue_obs / max(steps, 1),
                "red_mav_shared_fraction": total_red_shared / max(total_red_obs, 1),
                "red_direct_fraction": total_red_direct / max(total_red_obs, 1),
                "blue_direct_fraction": total_blue_direct / max(total_blue_obs, 1),
                "red_mav_shared_tracks_total": total_red_shared,
                "red_direct_tracks_total": total_red_direct,
                "blue_direct_tracks_total": total_blue_direct,
                "step_geometry": geometry_steps,
                "initial_min_red_blue_distance_m": initial_min_distance,
                "final_min_red_blue_distance_m": final_min_distance,
                "min_red_blue_distance_delta_m": final_min_distance - initial_min_distance,
                "closest_red_blue_distance_m": closest_distance,
                "closest_red_blue_distance_step": closest_step,
                "blue_closing_fraction": blue_closing_fraction,
                "blue_ever_within_direct_range": bool(closest_distance <= uav_direct),
                "direct_range_margin_final_m": final_min_distance - uav_direct,
                "direct_range_margin_closest_m": closest_distance - uav_direct,
                "heading_wrap_used": True,
                "turn_back_count": turn_back_count,
                "turn_back_heading_delta_mean_abs": turn_back_heading_delta_mean_abs,
                "turn_back_heading_values_sample": turn_back_heading_values[:10],
                "post_pass_separation_m": post_pass_separation,
                "mav_altitude_min_m": float(np.min(mav_altitudes)) if mav_altitudes else 0.0,
                "mav_altitude_final_m": float(mav_final["mav_altitude_m"]),
                "mav_alive_final": bool(mav_final["mav_alive"]),
                "action_trim_enabled": bool(getattr(env, "action_trim_enabled", False)),
                "action_trim_by_role": _trim_by_role(env),
                "final_red_alive": red_alive, "final_blue_alive": blue_alive,
                "final_mav_alive": mav_alive,
                "nan_detected": nan_detected,
                "warnings": warnings,
            }

            print(f"{Path(cfg_path).stem:45s} "
                  f"horizon={steps:4d} "
                  f"r_obs={rec['red_observed_any']} b_obs={rec['blue_observed_any']} "
                  f"first_r={first_red} first_b={first_blue} "
                  f"min0={initial_min_distance:.1f} minf={final_min_distance:.1f} "
                  f"closest={closest_distance:.1f} "
                  f"closing={blue_closing_fraction:.2f} "
                  f"turn_back={turn_back_count} "
                  f"post_pass={post_pass_separation:.1f} "
                  f"mav_shared={rec['red_mav_shared_fraction']:.2f} "
                  f"b_direct={rec['blue_direct_fraction']:.2f} "
                  f"warnings={len(warnings)}")
            for w in warnings:
                print(f"  WARN: {w}")
            return rec
        finally:
            if env is not None:
                env.close()

def _build_summary(records: list[dict], horizons: list[int]) -> dict:
    by_config: dict[str, dict] = {}
    for cfg in sorted({record["config"] for record in records}):
        cfg_records = [record for record in records if record["config"] == cfg]
        first_blue = next(
            (record["horizon_steps"] for record in sorted(
                cfg_records, key=lambda item: item["horizon_steps"])
             if record["blue_observed_any"]),
            None,
        )
        first_red = next(
            (record["horizon_steps"] for record in sorted(
                cfg_records, key=lambda item: item["horizon_steps"])
             if record["red_observed_any"]),
            None,
        )
        item = {
            "first_horizon_blue_observed": first_blue,
            "first_horizon_red_observed": first_red,
        }
        for horizon in horizons:
            record = next(
                (r for r in cfg_records if r["horizon_steps"] == horizon),
                None,
            )
            item[f"blue_observed_by_{horizon}"] = (
                bool(record["blue_observed_any"]) if record else False)
            item[f"red_observed_by_{horizon}"] = (
                bool(record["red_observed_any"]) if record else False)
        by_config[cfg] = item

    return {
        "configs_audited": len(records),
        "horizons": horizons,
        "horizon_summary_by_config": by_config,
        "red_never_observed": [r["config"] for r in records if not r["red_observed_any"]],
        "blue_never_observed": [r["config"] for r in records if not r["blue_observed_any"]],
        "asymmetric_info": [
            r["config"] for r in records
            if any("asymmetric information" in w for w in r["warnings"])
        ],
        "initial_geometry_concern": [
            r["config"] for r in records
            if any("initial geometry concern" in w for w in r["warnings"])
        ],
        "nan_records": [r["config"] for r in records if r.get("nan_detected")],
    }


if __name__ == "__main__":
    main()
