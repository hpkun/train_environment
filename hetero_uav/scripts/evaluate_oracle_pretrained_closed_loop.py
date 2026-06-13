"""Evaluate the oracle-pretrained UAV actor before PPO fine-tuning.

This script isolates whether the behavior-cloned UAV actor can close the loop
in the easy-combat environment.  It intentionally does not train or alter
environment mechanics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy
from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.red_attack_audit_utils import (
    alive_counts,
    collect_step_counts,
    geometry,
    safe_mav_action,
    team_done,
)


DEFAULT_CHECKPOINT = "outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt"
DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml"
)
DEFAULT_OUTPUT_DIR = "outputs/oracle_pretrained_closed_loop_eval"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_meta(checkpoint: Path) -> dict:
    meta_path = checkpoint.parent / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _episode_winner(env, length: int, truncated: dict) -> str:
    counts = alive_counts(env)
    red_alive = int(counts["red_alive"])
    blue_alive = int(counts["blue_alive"])
    timeout = bool(all(truncated.values()) or length >= getattr(env, "max_steps", 0))
    if blue_alive == 0 and red_alive > 0:
        return "red"
    if red_alive == 0 and blue_alive > 0:
        return "blue"
    if timeout:
        if red_alive > blue_alive:
            return "red"
        if blue_alive > red_alive:
            return "blue"
    return "draw"


def _as_action_dict(env, actions: np.ndarray, mav_safe_fixed: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, rid in enumerate(env.red_ids):
        if i == 0 and mav_safe_fixed:
            out[rid] = safe_mav_action()
        else:
            out[rid] = np.asarray(actions[i], dtype=np.float32)
    return out


def _finalize_action_stats(values: list[np.ndarray]) -> dict:
    if not values:
        return {
            "uav_action_mean": [0.0, 0.0, 0.0],
            "uav_action_std": [0.0, 0.0, 0.0],
            "uav_action_saturation_rate": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    return {
        "uav_action_mean": arr.mean(axis=0).astype(float).tolist(),
        "uav_action_std": arr.std(axis=0).astype(float).tolist(),
        "uav_action_saturation_rate": float(np.mean(np.any(np.abs(arr) > 0.95, axis=1))),
    }


def evaluate(args) -> dict:
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    checkpoint = _rel(args.checkpoint)
    config = args.config
    output_dir = _rel(args.output_dir)
    device = torch.device(args.device)
    meta = _load_meta(checkpoint)
    policy = HAPPOReferencePolicy(
        int(meta.get("actor_obs_dim", 96)),
        int(meta.get("critic_state_dim", 480)),
    ).to(device)
    policy.load(checkpoint, map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 777)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    winners: list[str] = []
    mav_alive: list[float] = []
    red_fired: list[int] = []
    red_hits: list[int] = []
    blue_dead: list[float] = []
    range_hits = 0
    angle_hits = 0
    envelope_hits = 0
    gate_denominator = 0
    uav_actions: list[np.ndarray] = []
    nan_detected = False

    for ep in range(args.episodes):
        env = make_env(config, env_type="jsbsim_hetero", max_steps=args.max_steps)
        obs, info = env.reset(seed=args.seed + ep)
        roles = _role_ids(env)
        ep_return = 0.0
        ep_red_fired = 0
        ep_red_hits = 0
        prev_hits = {"red": 0, "blue": 0}
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        length = 0
        while length < args.max_steps:
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
                    torch.as_tensor(actor_obs, dtype=torch.float32, device=device),
                    roles=roles,
                    critic_state=torch.as_tensor(critic, dtype=torch.float32, device=device),
                    deterministic=args.deterministic,
                )
            actions = out["action"].detach().cpu().numpy()
            if np.isnan(actions).any():
                nan_detected = True
                break
            for idx, rid in enumerate(env.red_ids):
                if env.agent_roles.get(rid) == "mav":
                    continue
                geo = geometry(env, rid)
                if bool(geo.get("alive", False)):
                    gate_denominator += 1
                    range_hits += int(bool(geo.get("launch_condition_distance", False)))
                    angle_hits += int(bool(geo.get("launch_condition_angle", False)))
                    envelope_hits += int(bool(geo.get("launch_condition_all", False)))
                    uav_actions.append(actions[idx].astype(np.float32))
            action_dict = _as_action_dict(env, actions, args.mav_safe_fixed)
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            counts = collect_step_counts(info)
            ep_red_fired += int(counts["red_fired"])
            ep_red_hits += max(int(counts["red_hits_total"]) - prev_hits["red"], 0)
            prev_hits["red"] = int(counts["red_hits_total"])
            prev_hits["blue"] = int(counts["blue_hits_total"])
            ep_return += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
            length += 1
            if team_done(terminated, truncated):
                break
        counts = alive_counts(env)
        episode_returns.append(ep_return)
        episode_lengths.append(length)
        winners.append(_episode_winner(env, length, truncated))
        mav = env.red_planes.get("red_0")
        mav_alive.append(1.0 if mav is not None and mav.is_alive else 0.0)
        red_fired.append(ep_red_fired)
        red_hits.append(ep_red_hits)
        blue_dead.append(float(counts["blue_dead"]))

    n = max(len(winners), 1)
    action_stats = _finalize_action_stats(uav_actions)
    result = {
        "checkpoint": str(checkpoint),
        "config": str(_rel(config)),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "deterministic": bool(args.deterministic),
        "mav_control": "safe_fixed" if args.mav_safe_fixed else "policy",
        "opponent_policy": args.opponent_policy,
        "avg_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "red_missiles_fired_mean": float(np.mean(red_fired)) if red_fired else 0.0,
        "red_missile_hits_mean": float(np.mean(red_hits)) if red_hits else 0.0,
        "blue_dead_mean": float(np.mean(blue_dead)) if blue_dead else 0.0,
        "mav_survival_rate": float(np.mean(mav_alive)) if mav_alive else 0.0,
        "red_win_rate": sum(1 for w in winners if w == "red") / n,
        "blue_win_rate": sum(1 for w in winners if w == "blue") / n,
        "timeout_rate": sum(1 for l in episode_lengths if l >= args.max_steps) / n,
        "launch_range_rate": float(range_hits / max(gate_denominator, 1)),
        "launch_angle_rate": float(angle_hits / max(gate_denominator, 1)),
        "launch_envelope_rate": float(envelope_hits / max(gate_denominator, 1)),
        "gate_denominator": int(gate_denominator),
        "nan_detected": bool(nan_detected),
        **action_stats,
    }
    result["oracle_closed_loop_fires"] = bool(result["red_missiles_fired_mean"] > 0.0)
    result["blocking_issue"] = (
        "oracle_pretrained_closed_loop_no_red_fire"
        if not result["oracle_closed_loop_fires"] else ""
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mode = f"{'det' if args.deterministic else 'stoch'}_{result['mav_control']}"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    mode_json = output_dir / f"summary_{mode}.json"
    mode_md = output_dir / f"summary_{mode}.md"
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    mode_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(summary_md, result)
    _write_md(mode_md, result)
    print(f"output_json: {summary_json}", flush=True)
    print(f"output_md: {summary_md}", flush=True)
    print(f"red_missiles_fired_mean: {result['red_missiles_fired_mean']:.3f}", flush=True)
    print(f"launch_envelope_rate: {result['launch_envelope_rate']:.3f}", flush=True)
    print(f"blocking_issue: {result['blocking_issue'] or 'none'}", flush=True)
    return result


def _write_md(path: Path, result: dict) -> None:
    lines = [
        "# Oracle-Pretrained Closed-Loop Evaluation",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- deterministic: {result['deterministic']}",
        f"- mav_control: {result['mav_control']}",
        f"- opponent_policy: {result['opponent_policy']}",
        f"- red_missiles_fired_mean: {result['red_missiles_fired_mean']:.3f}",
        f"- red_missile_hits_mean: {result['red_missile_hits_mean']:.3f}",
        f"- blue_dead_mean: {result['blue_dead_mean']:.3f}",
        f"- mav_survival_rate: {result['mav_survival_rate']:.3f}",
        f"- red_win_rate: {result['red_win_rate']:.3f}",
        f"- blue_win_rate: {result['blue_win_rate']:.3f}",
        f"- launch_range_rate: {result['launch_range_rate']:.3f}",
        f"- launch_angle_rate: {result['launch_angle_rate']:.3f}",
        f"- launch_envelope_rate: {result['launch_envelope_rate']:.3f}",
        f"- uav_action_mean: {result['uav_action_mean']}",
        f"- uav_action_std: {result['uav_action_std']}",
        f"- uav_action_saturation_rate: {result['uav_action_saturation_rate']:.3f}",
        f"- nan_detected: {result['nan_detected']}",
        f"- blocking_issue: {result['blocking_issue'] or 'none'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle-pretrained UAV actor in closed-loop easy combat"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--mav-safe-fixed", dest="mav_safe_fixed", action="store_true", default=True)
    parser.add_argument("--mav-policy", dest="mav_safe_fixed", action="store_false")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
