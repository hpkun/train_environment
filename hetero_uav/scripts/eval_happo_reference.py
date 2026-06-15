"""Evaluate HAPPO reference v0 checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import (
    BRMAEntityHAPPOReferencePolicy,
    EntityHAPPOReferencePolicy,
    HAPPOReferencePolicy,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _load_meta(model_path: Path) -> dict:
    meta_path = model_path.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _build_policy_from_meta(meta: dict, device: torch.device):
    policy_arch = meta.get("policy_arch", "flat")
    if policy_arch == "entity_attention":
        return EntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "brma_entity":
        return BRMAEntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "flat":
        return HAPPOReferencePolicy(
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
        ).to(device)
    raise ValueError(f"unsupported checkpoint policy_arch: {policy_arch}")


def _alive_counts(env) -> tuple[int, int]:
    return (
        sum(1 for sim in env.red_planes.values() if sim.is_alive),
        sum(1 for sim in env.blue_planes.values() if sim.is_alive),
    )


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def _episode_result(env, ep_len: int, truncated: dict) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    timeout = bool(all(truncated.values()) or ep_len >= getattr(env, "max_steps", 0))
    if blue_alive == 0 and red_alive > 0:
        reason, winner = "red_win_elimination", "red"
    elif red_alive == 0 and blue_alive > 0:
        reason, winner = "blue_win_elimination", "blue"
    elif red_alive == 0 and blue_alive == 0:
        reason, winner = "mutual_elimination_draw", "draw"
    elif timeout:
        reason = "timeout"
        if red_alive > blue_alive:
            winner = "red_alive_advantage"
        elif blue_alive > red_alive:
            winner = "blue_alive_advantage"
        else:
            winner = "draw"
    else:
        reason, winner = "other", "draw"
    mav = env.red_planes.get("red_0")
    return {
        "red_alive": red_alive,
        "blue_alive": blue_alive,
        "red_dead": max(len(env.red_planes) - red_alive, 0),
        "blue_dead": max(len(env.blue_planes) - blue_alive, 0),
        "mav_alive": bool(mav is not None and mav.is_alive),
        "episode_end_reason": reason,
        "winner": winner,
    }


def _empty_stats() -> dict:
    return {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}


def _update_missile_stats(stats: dict, info: dict, env, prev_hits: dict) -> None:
    for aid in env.agent_ids:
        agent_info = info.get(aid, {})
        fired = int(agent_info.get("missiles_fired_this_step", 0)) if isinstance(agent_info, dict) else 0
        if aid.startswith("red_"):
            stats["red_fired"] += fired
        else:
            stats["blue_fired"] += fired
    mt = info.get("__missile_term__", {})
    if isinstance(mt, dict):
        red_total = int(mt.get("red", {}).get("hit", 0))
        blue_total = int(mt.get("blue", {}).get("hit", 0))
        stats["red_hits"] += max(red_total - prev_hits.get("red", 0), 0)
        stats["blue_hits"] += max(blue_total - prev_hits.get("blue", 0), 0)
        prev_hits["red"] = red_total
        prev_hits["blue"] = blue_total


def evaluate_config(policy, cfg_path: str, args, adapter, device) -> dict:
    env = make_env(cfg_path, env_type="jsbsim_hetero")
    if args.max_steps_override is not None:
        env.max_steps = args.max_steps_override
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 99)
    returns, lengths, red_alive, blue_alive = [], [], [], []
    results = []
    missile_stats = []
    nan_detected = False
    mav_sat_values, uav_sat_values = [], []
    roles = _role_ids(env)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_len = 0
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        mstats = _empty_stats()
        prev_hits = {"red": 0, "blue": 0}
        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                for rid in env.red_ids
            ])
            critic = adapted["critic_state"]
            if np.isnan(actor_obs).any() or np.isnan(critic).any():
                nan_detected = True
                break
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(actor_obs, device=device),
                    roles=roles,
                    critic_state=torch.as_tensor(critic, device=device),
                    deterministic=True,
                )
            actions = out["action"].detach().cpu().numpy()
            if np.isnan(actions).any():
                nan_detected = True
                break
            mav_sat_values.append(float(np.mean(np.abs(actions[0:1]) >= 0.999)))
            if actions.shape[0] > 1:
                uav_sat_values.append(float(np.mean(np.abs(actions[1:]) >= 0.999)))
            action_dict = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            _update_missile_stats(mstats, info, env, prev_hits)
            ep_ret += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
            ep_len += 1
            if _team_done(terminated, truncated):
                break
        ra, ba = _alive_counts(env)
        returns.append(ep_ret)
        lengths.append(ep_len)
        red_alive.append(ra)
        blue_alive.append(ba)
        results.append(_episode_result(env, ep_len, truncated))
        missile_stats.append(mstats)

    reason_counts = Counter(r["episode_end_reason"] for r in results)
    winner_counts = Counter(r["winner"] for r in results)
    n = max(len(results), 1)
    red_dead = [r["red_dead"] for r in results]
    blue_dead = [r["blue_dead"] for r in results]
    red_win = winner_counts["red"] + winner_counts["red_alive_advantage"]
    blue_win = winner_counts["blue"] + winner_counts["blue_alive_advantage"]
    draw = winner_counts["draw"]
    red_fired = [m["red_fired"] for m in missile_stats]
    blue_fired = [m["blue_fired"] for m in missile_stats]
    red_hits = [m["red_hits"] for m in missile_stats]
    blue_hits = [m["blue_hits"] for m in missile_stats]
    return {
        "config": cfg_path,
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
        "red_win_rate": red_win / n,
        "blue_win_rate": blue_win / n,
        "draw_rate": draw / n,
        "timeout_rate": reason_counts["timeout"] / n,
        "red_elimination_win_rate": reason_counts["red_win_elimination"] / n,
        "blue_elimination_win_rate": reason_counts["blue_win_elimination"] / n,
        "red_timeout_alive_advantage_rate": winner_counts["red_alive_advantage"] / n,
        "blue_timeout_alive_advantage_rate": winner_counts["blue_alive_advantage"] / n,
        "timeout_draw_rate": (winner_counts["draw"] / n if reason_counts["timeout"] else 0.0),
        "mav_survival_rate": sum(1 for r in results if r["mav_alive"]) / n,
        "red_alive_final_mean": float(np.mean(red_alive)),
        "blue_alive_final_mean": float(np.mean(blue_alive)),
        "red_dead_mean": float(np.mean(red_dead)),
        "blue_dead_mean": float(np.mean(blue_dead)),
        "kill_death_ratio": float(np.mean(blue_dead) / max(np.mean(red_dead), 1e-6)),
        "red_missiles_fired_mean": float(np.mean(red_fired)),
        "blue_missiles_fired_mean": float(np.mean(blue_fired)),
        "red_missile_hits_mean": float(np.mean(red_hits)),
        "blue_missile_hits_mean": float(np.mean(blue_hits)),
        "mav_action_saturation_rate": float(np.mean(mav_sat_values)) if mav_sat_values else 0.0,
        "uav_action_saturation_rate": float(np.mean(uav_sat_values)) if uav_sat_values else 0.0,
        "episode_end_reason_counts": dict(reason_counts),
        "winner_counts": dict(winner_counts),
        "nan_detected": nan_detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule"])
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-steps-override", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    meta = _load_meta(Path(args.model))
    policy = _build_policy_from_meta(meta, device)
    policy.load(Path(args.model), map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()
    configs = args.configs or DEFAULT_CONFIGS
    records = []
    print("algorithm: happo_reference_v0", flush=True)
    print(f"episodes: {args.episodes}", flush=True)
    for cfg in configs:
        record = evaluate_config(policy, cfg, args, adapter, device)
        records.append(record)
        print(f"=== {cfg} ===", flush=True)
        for key in ["avg_return", "avg_length", "red_win_rate", "blue_win_rate",
                    "draw_rate", "timeout_rate", "mav_survival_rate",
                    "red_missile_hits_mean", "blue_missile_hits_mean"]:
            print(f"{key}: {record[key]}", flush=True)
    if args.summary_json:
        out = ROOT / args.summary_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
