"""Evaluate HAPPO checkpoints in deterministic and stochastic policy modes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _model_path(exp_dir: Path, checkpoint: str) -> Path:
    model = exp_dir / checkpoint / "model.pt"
    if not model.exists():
        raise FileNotFoundError(f"checkpoint not found: {model}")
    return model


def _load_meta(model: Path) -> dict:
    meta = model.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def _alive_counts(env) -> tuple[int, int, bool]:
    red = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    mav = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
    return red, blue, mav


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _episode_result(env, step: int, truncated: dict) -> dict:
    red_alive, blue_alive, mav_alive = _alive_counts(env)
    timeout = bool(all(truncated.values()) or step >= int(getattr(env, "max_steps", 0)))
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
    return {
        "red_alive": red_alive,
        "blue_alive": blue_alive,
        "red_dead": max(len(env.red_planes) - red_alive, 0),
        "blue_dead": max(len(env.blue_planes) - blue_alive, 0),
        "mav_alive": mav_alive,
        "episode_end_reason": reason,
        "winner": winner,
    }


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
        for side in ("red", "blue"):
            total = int(mt.get(side, {}).get("hit", 0))
            stats[f"{side}_hits"] += max(total - prev_hits.get(side, 0), 0)
            prev_hits[side] = total


def _evaluate_one(model: Path, checkpoint: str, mode: str, cfg: str, args) -> dict:
    import numpy as np
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from algorithms.happo import HAPPOReferencePolicy
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    device = torch.device(args.device)
    meta = _load_meta(model)
    policy = HAPPOReferencePolicy(
        int(meta.get("actor_obs_dim", 96)),
        int(meta.get("critic_state_dim", 480)),
    ).to(device)
    policy.load(model, map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()
    env = make_env(cfg, env_type="jsbsim_hetero")
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)
    returns, lengths, results, missile_stats = [], [], [], []
    action_abs, action_sat = [], []
    nan_detected = False
    roles = _role_ids(env)
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            ep_ret = 0.0
            step = 0
            prev_hits = {"red": 0, "blue": 0}
            mstats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
            terminated = {aid: False for aid in env.agent_ids}
            truncated = {aid: False for aid in env.agent_ids}
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
                        deterministic=(mode == "deterministic"),
                    )
                acts = out["action"].detach().cpu().numpy()
                if np.isnan(acts).any():
                    nan_detected = True
                    break
                action_abs.append(float(np.mean(np.abs(acts))))
                action_sat.append(float(np.mean(np.abs(acts) >= 0.999)))
                actions = {rid: acts[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
                actions.update(opponent.act(obs, env.blue_ids, env=env))
                obs, rewards, terminated, truncated, info = env.step(actions)
                _update_missile_stats(mstats, info, env, prev_hits)
                ep_ret += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
                step += 1
                if _team_done(terminated, truncated):
                    break
            returns.append(ep_ret)
            lengths.append(step)
            results.append(_episode_result(env, step, truncated))
            missile_stats.append(mstats)
    finally:
        if hasattr(env, "close"):
            env.close()

    import numpy as np

    winners = Counter(r["winner"] for r in results)
    reasons = Counter(r["episode_end_reason"] for r in results)
    n = max(len(results), 1)
    red_win = winners["red"] + winners["red_alive_advantage"]
    blue_win = winners["blue"] + winners["blue_alive_advantage"]
    red_dead = [r["red_dead"] for r in results]
    blue_dead = [r["blue_dead"] for r in results]
    red_hits = [m["red_hits"] for m in missile_stats]
    blue_hits = [m["blue_hits"] for m in missile_stats]
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "model": str(model),
        "config": cfg,
        "episodes": args.episodes,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "red_win_rate": red_win / n,
        "blue_win_rate": blue_win / n,
        "draw_rate": winners["draw"] / n,
        "timeout_rate": reasons["timeout"] / n,
        "mav_survival_rate": sum(1 for r in results if r["mav_alive"]) / n,
        "red_alive_final_mean": float(np.mean([r["red_alive"] for r in results])) if results else 0.0,
        "blue_alive_final_mean": float(np.mean([r["blue_alive"] for r in results])) if results else 0.0,
        "red_dead_mean": float(np.mean(red_dead)) if red_dead else 0.0,
        "blue_dead_mean": float(np.mean(blue_dead)) if blue_dead else 0.0,
        "red_missile_hits_mean": float(np.mean(red_hits)) if red_hits else 0.0,
        "blue_missile_hits_mean": float(np.mean(blue_hits)) if blue_hits else 0.0,
        "action_mean_abs": float(np.mean(action_abs)) if action_abs else 0.0,
        "action_saturation_rate": float(np.mean(action_sat)) if action_sat else 0.0,
        "episode_end_reason_counts": dict(reasons),
        "winner_counts": dict(winners),
        "nan_detected": bool(nan_detected),
    }


def _write_md(path: Path, payload: dict) -> None:
    lines = ["# HAPPO Policy Mode Evaluation", ""]
    for record in payload["records"]:
        lines.extend([
            f"## {record['checkpoint']} {record['mode']} - {Path(record['config']).name}",
            f"- red_win_rate: {record['red_win_rate']}",
            f"- blue_win_rate: {record['blue_win_rate']}",
            f"- timeout_rate: {record['timeout_rate']}",
            f"- mav_survival_rate: {record['mav_survival_rate']}",
            f"- blue_dead_mean: {record['blue_dead_mean']}",
            f"- red_missile_hits_mean: {record['red_missile_hits_mean']}",
            f"- action_saturation_rate: {record['action_saturation_rate']}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HAPPO policy modes under deterministic/stochastic actions")
    parser.add_argument("--experiment-dir", "--output-dir", dest="experiment_dir", default=DEFAULT_DIR)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--modes", nargs="+", choices=["deterministic", "stochastic"], default=["deterministic", "stochastic"])
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "latest"], default=["best", "latest"])
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    exp_dir = _rel(args.experiment_dir)
    if not exp_dir.exists():
        print(f"checkpoint not found: {exp_dir / 'best' / 'model.pt'}", file=sys.stderr)
        return 2
    try:
        models = [(name, _model_path(exp_dir, name)) for name in args.checkpoints]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    records = []
    for checkpoint, model in models:
        for mode in args.modes:
            for cfg in args.configs:
                record = _evaluate_one(model, checkpoint, mode, cfg, args)
                records.append(record)
                print(
                    f"{checkpoint} {mode} {Path(cfg).name}: "
                    f"red_win={record['red_win_rate']:.3f} "
                    f"blue_win={record['blue_win_rate']:.3f} "
                    f"mav_surv={record['mav_survival_rate']:.3f}",
                    flush=True,
                )
    payload = {
        "experiment_dir": str(exp_dir),
        "episodes": args.episodes,
        "modes": args.modes,
        "checkpoints": args.checkpoints,
        "records": records,
    }
    out_dir = exp_dir / "policy_mode_eval"
    out_json = _rel(args.output_json) if args.output_json else out_dir / "happo_policy_mode_eval.json"
    out_md = _rel(args.output_md) if args.output_md else out_dir / "happo_policy_mode_eval.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out_md, payload)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
