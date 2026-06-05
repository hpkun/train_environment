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
from algorithms.mappo.opponent_policy import OpponentPolicy

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--red-policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--blue-policy", choices=["zero", "rule_nearest", "greedy_fsm"],
                        default="greedy_fsm")
    parser.add_argument("--output-json",
                        default="outputs/environment_audit/hetero_visibility_geometry.json")
    args = parser.parse_args()

    records = []
    for cfg_path in CONFIGS:
        env = None
        try:
            env = make_env(cfg_path, env_type="jsbsim_hetero")
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

            for step in range(args.steps):
                # red actions
                if args.red_policy == "zero":
                    red_acts = {rid: np.zeros(3, dtype=np.float32)
                                for rid in env.red_ids}
                else:
                    red_acts = red_opponent.act(obs, env.red_ids)
                blue_acts = blue_opponent.act(obs, env.blue_ids)
                actions = {**red_acts, **blue_acts}

                # count tracks before step
                step_red_obs = 0
                step_blue_obs = 0
                for rid in env.red_ids:
                    tc = _enemy_track_counts(obs, rid)
                    step_red_obs += tc["observed"]
                    total_red_direct += tc["direct"]
                    total_red_shared += tc["shared"]
                for bid in env.blue_ids:
                    tc = _enemy_track_counts(obs, bid)
                    step_blue_obs += tc["observed"]
                    total_blue_direct += tc["direct"]

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
                mav_sim = env.red_planes.get("red_0") if red_count > 0 else None
                if mav_sim is not None and mav_sim.is_alive:
                    mav_alive_any = True

            red_alive = sum(1 for s in env.red_planes.values() if s.is_alive)
            blue_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
            mav_alive = int(mav_sim is not None and mav_sim.is_alive)

            warnings = []
            if first_red < 0:
                warnings.append("red never observed enemy")
            if first_blue < 0:
                warnings.append("blue never observed enemy")
                if args.blue_policy == "greedy_fsm":
                    warnings.append("greedy_fsm patrol-only likely caused by no visible blue enemy tracks")
            if total_red_shared > 0 and total_blue_direct == 0:
                warnings.append("asymmetric information: red has MAV shared tracks, blue has no direct tracks")
            if first_red < 0 and first_blue < 0 and args.steps >= 50:
                warnings.append("initial geometry concern: no mutual observation")

            rec = {
                "config": cfg_path,
                "red_count": red_count, "blue_count": blue_count,
                "sim_freq": sim_freq, "agent_interaction_steps": agent_is,
                "decision_dt": float(env.env_dt),
                "uav_direct_observation_range_m": uav_direct,
                "mav_observation_range_m": mav_range,
                "steps_executed": args.steps,
                "first_step_red_observed": first_red,
                "first_step_blue_observed": first_blue,
                "red_observed_any": first_red >= 0,
                "blue_observed_any": first_blue >= 0,
                "red_observed_fraction": steps_red_obs / max(args.steps, 1),
                "blue_observed_fraction": steps_blue_obs / max(args.steps, 1),
                "red_mav_shared_fraction": total_red_shared / max(total_red_obs, 1),
                "red_direct_fraction": total_red_direct / max(total_red_obs, 1),
                "blue_direct_fraction": total_blue_direct / max(total_blue_obs, 1),
                "red_mav_shared_tracks_total": total_red_shared,
                "red_direct_tracks_total": total_red_direct,
                "blue_direct_tracks_total": total_blue_direct,
                "final_red_alive": red_alive, "final_blue_alive": blue_alive,
                "final_mav_alive": mav_alive,
                "warnings": warnings,
            }
            records.append(rec)

            print(f"{Path(cfg_path).stem:45s} "
                  f"r_obs={rec['red_observed_any']} b_obs={rec['blue_observed_any']} "
                  f"first_r={first_red} first_b={first_blue} "
                  f"mav_shared={rec['red_mav_shared_fraction']:.2f} "
                  f"b_direct={rec['blue_direct_fraction']:.2f} "
                  f"warnings={len(warnings)}")
            for w in warnings:
                print(f"  WARN: {w}")
        finally:
            if env is not None:
                env.close()

    summary = {
        "configs_audited": len(records),
        "red_never_observed": [r["config"] for r in records if not r["red_observed_any"]],
        "blue_never_observed": [r["config"] for r in records if not r["blue_observed_any"]],
        "asymmetric_info": [r["config"] for r in records
                            if "asymmetric information" in r["warnings"]],
        "initial_geometry_concern": [r["config"] for r in records
                                     if "initial geometry concern" in r["warnings"]],
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
