from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy  # noqa: E402


def _load_policy(checkpoint: str | None, policy_arch: str, device):
    from scripts.eval_happo_reference import _build_policy_from_meta, _load_meta
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    if not checkpoint:
        return None, HeteroObsAdapterV2()
    ckpt = Path(checkpoint)
    meta = _load_meta(ckpt)
    if policy_arch:
        meta = dict(meta)
        meta["policy_arch"] = policy_arch
    policy = _build_policy_from_meta(meta, device)
    policy.load(ckpt, map_location=device)
    policy.eval()
    adapter = HeteroEntitySetAdapter() if meta.get("policy_arch") == "hetero_entity_recurrent" else HeteroObsAdapterV2()
    return policy, adapter


def _active_mask(env, info) -> np.ndarray:
    active = np.zeros(len(env.red_ids), dtype=np.float32)
    for i, rid in enumerate(env.red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = env.red_planes.get(rid)
            alive = bool(sim is not None and sim.is_alive)
        active[i] = 1.0 if alive else 0.0
    return active


def _red_policy_actions(policy, adapter, obs, info, env, roles, device, rnn_hidden, ep_len):
    import torch
    from algorithms.happo.rollout_safety import (
        sanitize_policy_inputs,
        zero_inactive_actions,
        zero_inactive_hidden,
    )
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter

    if policy is None:
        return {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}, rnn_hidden
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    active = _active_mask(env, info)
    active_rows = active > 0.5
    kwargs = {}
    if rnn_hidden is not None:
        rnn_hidden = zero_inactive_hidden(rnn_hidden, active)
        kwargs["rnn_hidden"] = torch.as_tensor(rnn_hidden, device=device)
    with torch.no_grad():
        if isinstance(adapter, HeteroEntitySetAdapter):
            actor_tokens = adapted["actor_entity_tokens"].copy()
            actor_keep = adapted["actor_keep_mask"].copy()
            critic_tokens = adapted["critic_entity_tokens"].copy()
            critic_keep = adapted["critic_keep_mask"].copy()
            actor_tokens[~active_rows] = 0.0
            actor_keep[~active_rows] = 0.0
            actor_keep[~active_rows, 0] = 1.0
            out = policy.act(
                actor_tokens,
                actor_keep,
                roles,
                critic_tokens,
                critic_keep,
                deterministic=True,
                critic_counts=torch.as_tensor(adapted.get("critic_counts", np.zeros(4, dtype=np.float32)), device=device),
                **kwargs,
            )
        else:
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                for rid in env.red_ids
            ])
            san = sanitize_policy_inputs(
                actor_obs,
                active,
                critic_state=adapted["critic_state"],
                rnn_hidden=rnn_hidden,
                context={"env_idx": "diagnose", "episode_id": "diag", "total_steps": ep_len},
            )
            out = policy.act(
                torch.as_tensor(san["actor_obs"], device=device),
                roles=roles,
                critic_state=torch.as_tensor(san["critic_state"], device=device),
                deterministic=True,
                **kwargs,
            )
    if rnn_hidden is not None and "rnn_hidden" in out:
        rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].detach().cpu().numpy(), active)
    actions = zero_inactive_actions(out["action"].detach().cpu().numpy(), active)
    return {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}, rnn_hidden


def _death_counts(env) -> tuple[int, int]:
    crash = 0
    missile = 0
    reasons = getattr(env, "_death_reasons", {}) or {}
    for rid in getattr(env, "red_ids", []):
        reason = str(reasons.get(rid, ""))
        if "Missile" in reason or "Shot" in reason or "Hit" in reason:
            missile += 1
        elif "Crash" in reason or "Boundary" in reason or "NonFinite" in reason:
            crash += 1
    return crash, missile


def _run_one_policy(args, opponent_policy: str, policy, adapter, device) -> dict:
    from scripts.eval_happo_reference import (
        _alive_counts,
        _empty_stats,
        _episode_result,
        _role_ids,
        _team_done,
        _update_missile_stats,
    )
    from uav_env import make_env

    env = make_env(args.config, env_type="jsbsim_hetero", max_steps=args.max_steps)
    opponent = OpponentPolicy(opponent_policy, seed=args.seed + 101)
    roles = _role_ids(env)
    rnn_size = int(getattr(policy, "rnn_hidden_size", 0)) if policy is not None else 0
    rows = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            opponent.reset_memory()
            rnn_hidden = np.zeros((len(env.red_ids), rnn_size), dtype=np.float32) if rnn_size > 0 else None
            prev_red_alive, prev_blue_alive = _alive_counts(env)
            red_first_death = math.nan
            blue_first_launch = math.nan
            prev_hits = {"red": 0, "blue": 0}
            mstats = _empty_stats()
            ep_len = 0
            terminated = {aid: False for aid in env.agent_ids}
            truncated = {aid: False for aid in env.agent_ids}
            while True:
                red_actions, rnn_hidden = _red_policy_actions(
                    policy, adapter, obs, info, env, roles, device, rnn_hidden, ep_len)
                blue_actions = opponent.act(obs, env.blue_ids, env=env)
                action_dict = dict(red_actions)
                action_dict.update(blue_actions)
                obs, rewards, terminated, truncated, info = env.step(action_dict)
                _update_missile_stats(mstats, info, env, prev_hits)
                red_alive, blue_alive = _alive_counts(env)
                if math.isnan(red_first_death) and red_alive < prev_red_alive:
                    red_first_death = float(ep_len + 1)
                if math.isnan(blue_first_launch) and int(mstats.get("blue_fired", 0)) > 0:
                    blue_first_launch = float(ep_len + 1)
                prev_red_alive, prev_blue_alive = red_alive, blue_alive
                ep_len += 1
                if _team_done(terminated, truncated):
                    break
            result = _episode_result(env, ep_len, truncated)
            red_crash, red_missile_death = _death_counts(env)
            rows.append({
                "winner": result.get("winner"),
                "steps": ep_len,
                "red_alive_final": prev_red_alive,
                "blue_alive_final": prev_blue_alive,
                "mav_alive": 1.0 if env.red_planes.get("red_0") is not None and env.red_planes["red_0"].is_alive else 0.0,
                "red_first_death_step": red_first_death,
                "blue_first_launch_step": blue_first_launch,
                "red_launches": float(mstats.get("red_fired", 0)),
                "blue_launches": float(mstats.get("blue_fired", 0)),
                "red_hits": float(mstats.get("red_hits", 0)),
                "blue_hits": float(mstats.get("blue_hits", 0)),
                "red_crash_count": red_crash,
                "red_missile_death_count": red_missile_death,
            })
    finally:
        env.close()
    return _summarize_rows(opponent_policy, rows)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _summarize_rows(opponent_policy: str, rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "opponent_policy": opponent_policy,
        "episodes": len(rows),
        "red_win_rate": sum(1 for r in rows if r["winner"] == "red") / n,
        "blue_win_rate": sum(1 for r in rows if r["winner"] == "blue") / n,
        "draw_rate": sum(1 for r in rows if r["winner"] == "draw") / n,
        "avg_steps": _nanmean([r["steps"] for r in rows]),
        "red_alive_final_mean": _nanmean([r["red_alive_final"] for r in rows]),
        "blue_alive_final_mean": _nanmean([r["blue_alive_final"] for r in rows]),
        "mav_survival_rate": _nanmean([r["mav_alive"] for r in rows]),
        "red_first_death_step_mean": _nanmean([r["red_first_death_step"] for r in rows]),
        "blue_first_launch_step_mean": _nanmean([r["blue_first_launch_step"] for r in rows]),
        "red_launches_mean": _nanmean([r["red_launches"] for r in rows]),
        "blue_launches_mean": _nanmean([r["blue_launches"] for r in rows]),
        "red_hits_mean": _nanmean([r["red_hits"] for r in rows]),
        "blue_hits_mean": _nanmean([r["blue_hits"] for r in rows]),
        "red_crash_count": int(sum(r["red_crash_count"] for r in rows)),
        "red_missile_death_count": int(sum(r["red_missile_death_count"] for r in rows)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "opponent_policy", "episodes", "red_win_rate", "blue_win_rate", "draw_rate",
        "avg_steps", "red_alive_final_mean", "blue_alive_final_mean",
        "mav_survival_rate", "red_first_death_step_mean",
        "blue_first_launch_step_mean", "red_launches_mean", "blue_launches_mean",
        "red_hits_mean", "blue_hits_mean", "red_crash_count", "red_missile_death_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rollout-only blue pressure diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy-arch", default="")
    parser.add_argument("--reward-mode", default="")
    parser.add_argument("--opponent-policies", nargs="+",
                        default=["brma_rule", "tam_greedy_easy", "brma_rule_safe_pursuit_easy", "brma_rule_safe_pursuit"],
                        help="Opponent modes to compare, e.g. brma_rule tam_greedy_easy brma_rule_safe_pursuit_easy brma_rule_safe_pursuit.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--summary-json", default="outputs/blue_pressure_diagnostics/summary.json")
    parser.add_argument("--csv", default="outputs/blue_pressure_diagnostics/summary.csv")
    args = parser.parse_args()

    del args.reward_mode  # Reward is configured by the env config; this option records CLI compatibility.
    import torch

    device = torch.device(args.device)
    policy, adapter = _load_policy(args.checkpoint, args.policy_arch, device)
    records = [
        _run_one_policy(args, name, policy, adapter, device)
        for name in args.opponent_policies
    ]
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_safe({"records": records}), indent=2), encoding="utf-8")
    _write_csv(Path(args.csv), records)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
