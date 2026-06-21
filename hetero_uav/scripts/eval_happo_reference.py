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
    BRMARecurrentMaskedHAPPOReferencePolicy,
    BRMARecurrentHAPPOReferencePolicy,
    EntityHAPPOReferencePolicy,
    HAPPOReferencePolicy,
)
from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import FEATURE_SCHEMA_VERSION
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
from algorithms.happo.rollout_safety import (
    sanitize_policy_inputs,
    zero_inactive_actions,
    zero_inactive_hidden,
)
from scripts.rich_logging import RichExperimentLogger, write_not_available_attention
try:
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
except ModuleNotFoundError:
    OpponentPolicy = None
    make_env = None
    HeteroObsAdapterV2 = None


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f22_pid.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_7v6_f22_pid.yaml",
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
    if policy_arch == "hetero_entity_recurrent":
        if int(meta.get("action_dim", -1)) != 3:
            raise ValueError("hetero_entity_recurrent checkpoint action_dim must be 3")
        if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported hetero_entity_recurrent feature_schema_version: "
                f"{meta.get('feature_schema_version')!r}"
            )
        if meta.get("adapter_mode") != "hetero_entity_set":
            raise ValueError("hetero_entity_recurrent checkpoint adapter_mode mismatch")
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta["entity_dim"]),
            action_dim=3,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta["rnn_hidden_size"]),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
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
    if policy_arch == "brma_recurrent":
        return BRMARecurrentHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
    if policy_arch == "brma_recurrent_masked":
        return BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            random_scale_mask=bool(meta.get("random_scale_mask", False)),
            random_mask_prob=float(meta.get("random_mask_prob", 0.25)),
            biased_mask=bool(meta.get("biased_mask", False)),
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


def evaluate_config(policy, cfg_path: str, args, adapter, device,
                    rich_logger=None) -> dict:
    cfg_name = Path(cfg_path).stem
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

    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_len = 0
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        mstats = _empty_stats()
        prev_hits = {"red": 0, "blue": 0}
        eval_rnn_hidden = None
        if _rnn_hidden_size > 0:
            eval_rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)
        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            entity_mode = isinstance(adapter, HeteroEntitySetAdapter)
            # Build active mask from info dict (same logic as training)
            active = np.zeros(len(env.red_ids), dtype=np.float32)
            for i, rid in enumerate(env.red_ids):
                agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
                if isinstance(agent_info, dict) and "alive" in agent_info:
                    alive = bool(agent_info["alive"])
                else:
                    sim = env.red_planes.get(rid)
                    alive = bool(sim is not None and sim.is_alive)
                active[i] = 1.0 if alive else 0.0
            san_ctx = {"env_idx": "eval", "episode_id": ep, "total_steps": ep_len}
            active_rows = active > 0.5
            if entity_mode:
                actor_tokens = adapted["actor_entity_tokens"].copy()
                actor_keep = adapted["actor_keep_mask"].copy()
                critic_tokens = adapted["critic_entity_tokens"].copy()
                critic_keep = adapted["critic_keep_mask"].copy()
                actor_tokens[~active_rows] = 0.0
                actor_keep[~active_rows] = 0.0
                actor_keep[~active_rows, 0] = 1.0
                if active_rows.any() and (
                    not np.isfinite(actor_tokens[active_rows]).all()
                    or not np.isfinite(critic_tokens[critic_keep > 0.5]).all()
                ):
                    nan_detected = True
                    break
                if eval_rnn_hidden is not None:
                    eval_rnn_hidden = zero_inactive_hidden(eval_rnn_hidden, active)
            else:
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                    for rid in env.red_ids
                ])
                critic = adapted["critic_state"]
                san = sanitize_policy_inputs(
                    actor_obs, active, critic_state=critic,
                    rnn_hidden=eval_rnn_hidden, context=san_ctx,
                )
                actor_obs = san["actor_obs"]
                critic = san["critic_state"] if san["critic_state"] is not None else critic
                eval_rnn_hidden = san["rnn_hidden"] if san["rnn_hidden"] is not None else eval_rnn_hidden
                if active_rows.any() and (
                    not np.isfinite(actor_obs[active_rows]).all() or not np.isfinite(critic).all()
                ):
                    nan_detected = True
                    break
            act_kwargs = {}
            if eval_rnn_hidden is not None:
                act_kwargs["rnn_hidden"] = torch.as_tensor(eval_rnn_hidden, device=device)
            with torch.no_grad():
                if entity_mode:
                    out = policy.act(
                        actor_tokens, actor_keep, roles, critic_tokens, critic_keep,
                        deterministic=True, **act_kwargs)
                else:
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device), roles=roles,
                        critic_state=torch.as_tensor(critic, device=device),
                        deterministic=True, **act_kwargs)
            if eval_rnn_hidden is not None and "rnn_hidden" in out:
                eval_rnn_hidden = zero_inactive_hidden(
                    out["rnn_hidden"].detach().cpu().numpy(), active)
            actions = zero_inactive_actions(
                out["action"].detach().cpu().numpy(), active)
            active_mask_np = active > 0.5
            if active_mask_np.any():
                if not np.isfinite(actions[active_mask_np]).all():
                    nan_detected = True
                    break
                if not np.isfinite(float(out["value"].item())):
                    nan_detected = True
                    break
            mav_sat_values.append(float(np.mean(np.abs(actions[0:1]) >= 0.999)))
            if actions.shape[0] > 1:
                uav_sat_values.append(float(np.mean(np.abs(actions[1:]) >= 0.999)))
            action_dict = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            _update_missile_stats(mstats, info, env, prev_hits)
            if rich_logger is not None:
                rich_logger.write_missile_events(
                    info, scenario=cfg_name, episode_id=ep,
                    step=ep_len, sim_time=_sim_time(env))
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


def _sim_time(env) -> float:
    try:
        return float(env.jsbsim_exec.get_sim_time()) if hasattr(env, "jsbsim_exec") else 0.0
    except Exception:
        return 0.0


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
    parser.add_argument("--enable-rich-logging", action="store_true")
    parser.add_argument("--rich-log-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    meta = _load_meta(Path(args.model))
    policy = _build_policy_from_meta(meta, device)
    policy.load(Path(args.model), map_location=device)
    policy.eval()
    adapter = (
        HeteroEntitySetAdapter()
        if meta.get("policy_arch") == "hetero_entity_recurrent"
        else HeteroObsAdapterV2()
    )
    rich_logger = None
    if args.enable_rich_logging:
        rich_dir = Path(args.rich_log_dir) if args.rich_log_dir else Path(args.model).parent / "eval_rich_logs"
        rich_logger = RichExperimentLogger(
            rich_dir,
            run_id=Path(args.model).parent.name,
            method_name="happo_reference_v0",
            scenario_name="eval",
            device=str(args.device),
            num_envs=1,
            rollout_length_per_env=0,
            transitions_per_rollout=0,
        )
    configs = args.configs or DEFAULT_CONFIGS
    records = []
    print("algorithm: happo_reference_v0", flush=True)
    print(f"episodes: {args.episodes}", flush=True)
    for cfg in configs:
        record = evaluate_config(policy, cfg, args, adapter, device, rich_logger=rich_logger)
        records.append(record)
        print(f"=== {cfg} ===", flush=True)
        for key in ["avg_return", "avg_length", "red_win_rate", "blue_win_rate",
                    "draw_rate", "timeout_rate", "mav_survival_rate",
                    "red_missile_hits_mean", "blue_missile_hits_mean"]:
            print(f"{key}: {record[key]}", flush=True)
    if rich_logger is not None:
        rich_logger.close()
    if args.summary_json:
        out = ROOT / args.summary_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
