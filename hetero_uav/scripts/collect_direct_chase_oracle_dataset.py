"""Collect red-UAV direct-chase oracle samples for behavior cloning."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.red_attack_audit_utils import (
    DEFAULT_CONFIG,
    collect_step_counts,
    direct_chase_action,
    geometry,
    make_env,
    safe_mav_action,
    team_done,
)


DEFAULT_OUTPUT = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"
DEFAULT_SUMMARY = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_summary.json"
ADAPTER_PATH = ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_adapter_class():
    spec = importlib.util.spec_from_file_location("hetero_obs_adapter_v2", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load adapter: {ADAPTER_PATH}")
    spec.loader.exec_module(module)
    return module.HeteroObsAdapterV2


def _blue_actions(mode: str, obs: dict, env, policy: OpponentPolicy) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {bid: np.zeros(3, dtype=np.float32) for bid in env.blue_ids}
    return policy.act(obs, env.blue_ids, env=env)


def _hit_delta(info: dict, prev_hits: dict[str, int]) -> int:
    mt = info.get("__missile_term__", {})
    if not isinstance(mt, dict):
        return 0
    red_total = int(mt.get("red", {}).get("hit", 0) or 0)
    delta = max(red_total - prev_hits.get("red", 0), 0)
    prev_hits["red"] = red_total
    prev_hits["blue"] = int(mt.get("blue", {}).get("hit", prev_hits.get("blue", 0)) or 0)
    return delta


def collect(args) -> tuple[dict[str, np.ndarray], dict]:
    HeteroObsAdapterV2 = _load_adapter_class()
    env = make_env(args.config, hetero_reward_mode="happo_ref_v0", max_steps=args.max_steps)
    adapter = HeteroObsAdapterV2()
    opponent = OpponentPolicy(mode="brma_rule", seed=args.seed + 29)
    arrays: dict[str, list] = {
        "actor_obs": [],
        "oracle_action": [],
        "role_id": [],
        "agent_id": [],
        "episode_id": [],
        "step": [],
        "alive_mask": [],
        "nearest_enemy_distance": [],
        "launch_range_flag": [],
        "launch_angle_flag": [],
        "launch_envelope_flag": [],
        "missile_fired_this_step": [],
        "missile_hit_this_step": [],
    }
    ep_red_fired, ep_red_hits, ep_blue_dead = [], [], []
    range_flags, angle_flags, envelope_flags = [], [], []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            prev_hits = {"red": 0, "blue": 0}
            red_fired_total = 0
            red_hits_total = 0
            for step in range(args.max_steps):
                adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                pending = []
                red_actions: dict[str, np.ndarray] = {}
                for rid in env.red_ids:
                    if env.agent_roles.get(rid) == "mav":
                        red_actions[rid] = safe_mav_action()
                        continue
                    sim = env.red_planes.get(rid)
                    if sim is None or not sim.is_alive:
                        continue
                    action = np.clip(direct_chase_action(env, rid), -1.0, 1.0).astype(np.float32)
                    red_actions[rid] = action
                    g = geometry(env, rid)
                    range_flag = bool(g.get("launch_condition_distance", False))
                    angle_flag = bool(g.get("launch_condition_angle", False))
                    envelope_flag = bool(g.get("launch_condition_all", False))
                    pending.append((rid, action, g, range_flag, angle_flag, envelope_flag))

                actions = dict(red_actions)
                actions.update(_blue_actions(args.opponent_policy, obs, env, opponent))
                obs, _rewards, terminated, truncated, info = env.step(actions)
                counts = collect_step_counts(info)
                red_fired_total += int(counts["red_fired"])
                red_hit_delta = _hit_delta(info, prev_hits)
                red_hits_total += red_hit_delta

                for rid, action, g, range_flag, angle_flag, envelope_flag in pending:
                    arrays["actor_obs"].append(adapted["actor_obs"][rid].astype(np.float32))
                    arrays["oracle_action"].append(action)
                    arrays["role_id"].append(1)
                    arrays["agent_id"].append(rid)
                    arrays["episode_id"].append(ep)
                    arrays["step"].append(step)
                    arrays["alive_mask"].append(1.0)
                    arrays["nearest_enemy_distance"].append(float(g.get("distance_m", np.inf)))
                    arrays["launch_range_flag"].append(float(range_flag))
                    arrays["launch_angle_flag"].append(float(angle_flag))
                    arrays["launch_envelope_flag"].append(float(envelope_flag))
                    arrays["missile_fired_this_step"].append(
                        float(info.get(rid, {}).get("missiles_fired_this_step", 0))
                        if isinstance(info.get(rid, {}), dict) else 0.0
                    )
                    arrays["missile_hit_this_step"].append(float(red_hit_delta > 0))
                    range_flags.append(range_flag)
                    angle_flags.append(angle_flag)
                    envelope_flags.append(envelope_flag)
                    if len(arrays["actor_obs"]) >= args.max_samples:
                        break
                if len(arrays["actor_obs"]) >= args.max_samples or team_done(terminated, truncated):
                    break
            ep_red_fired.append(red_fired_total)
            ep_red_hits.append(red_hits_total)
            ep_blue_dead.append(sum(not sim.is_alive for sim in env.blue_planes.values()))
            if len(arrays["actor_obs"]) >= args.max_samples:
                break
    finally:
        env.close()

    packed = {
        "actor_obs": np.asarray(arrays["actor_obs"], dtype=np.float32).reshape(-1, 96),
        "oracle_action": np.asarray(arrays["oracle_action"], dtype=np.float32).reshape(-1, 3),
        "role_id": np.asarray(arrays["role_id"], dtype=np.int64),
        "agent_id": np.asarray(arrays["agent_id"]),
        "episode_id": np.asarray(arrays["episode_id"], dtype=np.int64),
        "step": np.asarray(arrays["step"], dtype=np.int64),
        "alive_mask": np.asarray(arrays["alive_mask"], dtype=np.float32),
        "nearest_enemy_distance": np.asarray(arrays["nearest_enemy_distance"], dtype=np.float32),
        "launch_range_flag": np.asarray(arrays["launch_range_flag"], dtype=np.float32),
        "launch_angle_flag": np.asarray(arrays["launch_angle_flag"], dtype=np.float32),
        "launch_envelope_flag": np.asarray(arrays["launch_envelope_flag"], dtype=np.float32),
        "missile_fired_this_step": np.asarray(arrays["missile_fired_this_step"], dtype=np.float32),
        "missile_hit_this_step": np.asarray(arrays["missile_hit_this_step"], dtype=np.float32),
    }
    summary = {
        "num_samples": int(packed["actor_obs"].shape[0]),
        "episodes": int(len(ep_red_fired)),
        "config": args.config,
        "opponent_policy": args.opponent_policy,
        "red_missiles_fired_mean": float(np.mean(ep_red_fired)) if ep_red_fired else 0.0,
        "red_missile_hits_mean": float(np.mean(ep_red_hits)) if ep_red_hits else 0.0,
        "blue_dead_mean": float(np.mean(ep_blue_dead)) if ep_blue_dead else 0.0,
        "launch_range_rate": float(np.mean(range_flags)) if range_flags else 0.0,
        "launch_angle_rate": float(np.mean(angle_flags)) if angle_flags else 0.0,
        "launch_envelope_rate": float(np.mean(envelope_flags)) if envelope_flags else 0.0,
    }
    return packed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect direct-chase oracle dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--opponent-policy", choices=["brma_rule", "zero"], default="brma_rule")
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    packed, summary = collect(args)
    out = _rel(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **packed)
    summary_path = _rel(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output: {out}")
    print(f"summary_json: {summary_path}")
    print(f"num_samples: {summary['num_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
