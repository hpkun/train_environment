"""Audit red_0 MAV failure modes for HAPPO reference checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"


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


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _missile_hits(info: dict) -> dict[str, int]:
    mt = info.get("__missile_term__", {})
    if not isinstance(mt, dict):
        return {"red": 0, "blue": 0}
    return {
        "red": int(mt.get("red", {}).get("hit", 0)),
        "blue": int(mt.get("blue", {}).get("hit", 0)),
    }


def _eval_checkpoint(model: Path, checkpoint: str, args) -> dict:
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
    env = make_env(args.config, env_type="jsbsim_hetero")
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 41)
    roles = _role_ids(env)
    episodes = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            step = 0
            prev_alive = {aid: True for aid in env.red_ids + env.blue_ids}
            prev_hits = _missile_hits(info)
            death_order: list[str] = []
            red0_death_step = None
            red0_death_with_blue_hit = False
            red0_min_alt = math.inf
            red0_max_pitch = 0.0
            red0_max_roll = 0.0
            action_abs, action_sat = [], []
            nan_detected = False
            terminated = {aid: False for aid in env.agent_ids}
            truncated = {aid: False for aid in env.agent_ids}
            while True:
                red0 = env.red_planes.get("red_0")
                if red0 is not None:
                    try:
                        red0_min_alt = min(red0_min_alt, float(red0.get_position()[2]))
                        roll, pitch, _yaw = red0.get_rpy()
                        red0_max_roll = max(red0_max_roll, abs(math.degrees(float(roll))))
                        red0_max_pitch = max(red0_max_pitch, abs(math.degrees(float(pitch))))
                    except Exception:
                        nan_detected = True
                adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                    for rid in env.red_ids
                ])
                if np.isnan(actor_obs).any() or np.isnan(adapted["critic_state"]).any():
                    nan_detected = True
                    break
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device),
                        roles=roles,
                        critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                        deterministic=not args.stochastic,
                    )
                acts = out["action"].detach().cpu().numpy()
                if np.isnan(acts).any():
                    nan_detected = True
                    break
                action_abs.append(float(np.mean(np.abs(acts[0]))))
                action_sat.append(float(np.mean(np.abs(acts[0]) >= 0.999)))
                actions = {rid: acts[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
                actions.update(opponent.act(obs, env.blue_ids, env=env))
                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1
                hits = _missile_hits(info)
                blue_hit_delta = max(hits["blue"] - prev_hits["blue"], 0)
                prev_hits = hits
                for aid in env.red_ids + env.blue_ids:
                    sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
                    alive = bool(sim and sim.is_alive)
                    if prev_alive.get(aid, True) and not alive:
                        death_order.append(aid)
                        if aid == "red_0":
                            red0_death_step = step
                            red0_death_with_blue_hit = blue_hit_delta > 0
                    prev_alive[aid] = alive
                if _team_done(terminated, truncated):
                    break
            red0_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
            if red0_alive:
                death_reason = "survived"
            elif red0_death_with_blue_hit:
                death_reason = "missile_hit_likely"
            elif red0_min_alt < 1000.0:
                death_reason = "low_altitude_or_crash_likely"
            else:
                death_reason = "unknown_environment_death"
            episodes.append({
                "episode": ep,
                "steps": step,
                "red0_alive": red0_alive,
                "red0_death_step": red0_death_step,
                "red0_first_death": bool(death_order and death_order[0] == "red_0"),
                "death_order": death_order,
                "red0_death_reason": death_reason,
                "red0_missile_death_likely": bool(red0_death_with_blue_hit),
                "red0_low_altitude_death_likely": bool((not red0_alive) and red0_min_alt < 1000.0),
                "red0_min_altitude": None if red0_min_alt == math.inf else red0_min_alt,
                "red0_max_abs_pitch_deg": red0_max_pitch,
                "red0_max_abs_roll_deg": red0_max_roll,
                "red0_action_mean_abs": float(np.mean(action_abs)) if action_abs else 0.0,
                "red0_action_saturation_rate": float(np.mean(action_sat)) if action_sat else 0.0,
                "nan_detected": nan_detected,
            })
    finally:
        if hasattr(env, "close"):
            env.close()
    reasons = Counter(ep["red0_death_reason"] for ep in episodes)
    deaths = [ep for ep in episodes if not ep["red0_alive"]]
    summary = {
        "checkpoint": checkpoint,
        "model": str(model),
        "episodes": len(episodes),
        "mav_death_rate": len(deaths) / max(len(episodes), 1),
        "mav_first_death_rate": sum(1 for ep in episodes if ep["red0_first_death"]) / max(len(episodes), 1),
        "mav_missile_death_rate": reasons["missile_hit_likely"] / max(len(episodes), 1),
        "mav_crash_death_rate": reasons["low_altitude_or_crash_likely"] / max(len(episodes), 1),
        "death_reason_counts": dict(reasons),
        "mean_death_step": float(np.mean([ep["red0_death_step"] for ep in deaths if ep["red0_death_step"] is not None])) if deaths else None,
        "mean_min_altitude": float(np.mean([ep["red0_min_altitude"] for ep in episodes if ep["red0_min_altitude"] is not None])) if episodes else None,
        "mean_max_abs_pitch_deg": float(np.mean([ep["red0_max_abs_pitch_deg"] for ep in episodes])) if episodes else 0.0,
        "mean_max_abs_roll_deg": float(np.mean([ep["red0_max_abs_roll_deg"] for ep in episodes])) if episodes else 0.0,
        "mean_action_saturation_rate": float(np.mean([ep["red0_action_saturation_rate"] for ep in episodes])) if episodes else 0.0,
        "nan_detected": any(ep["nan_detected"] for ep in episodes),
    }
    if summary["mav_death_rate"] >= 0.9:
        summary["conclusion"] = "MAV failure is systematic in this checkpoint."
    elif summary["mav_death_rate"] > 0:
        summary["conclusion"] = "MAV failure occurs in a subset of episodes."
    else:
        summary["conclusion"] = "MAV survived all audited episodes."
    return {"summary": summary, "episodes": episodes}


def _write_md(path: Path, payload: dict) -> None:
    lines = ["# HAPPO MAV Failure Mode Audit", ""]
    for record in payload["records"]:
        summary = record["summary"]
        lines.extend([
            f"## {summary['checkpoint']}",
            f"- mav_death_rate: {summary['mav_death_rate']}",
            f"- mav_first_death_rate: {summary['mav_first_death_rate']}",
            f"- mav_missile_death_rate: {summary['mav_missile_death_rate']}",
            f"- mav_crash_death_rate: {summary['mav_crash_death_rate']}",
            f"- death_reason_counts: {summary['death_reason_counts']}",
            f"- conclusion: {summary['conclusion']}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HAPPO MAV failure modes")
    parser.add_argument("--experiment-dir", "--output-dir", dest="experiment_dir", default=DEFAULT_DIR)
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "latest"], default=["best", "latest"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--stochastic", action="store_true")
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
        record = _eval_checkpoint(model, checkpoint, args)
        records.append(record)
        summary = record["summary"]
        print(
            f"{checkpoint}: mav_death_rate={summary['mav_death_rate']:.3f} "
            f"first_death={summary['mav_first_death_rate']:.3f} "
            f"missile_likely={summary['mav_missile_death_rate']:.3f}",
            flush=True,
        )
    payload = {
        "experiment_dir": str(exp_dir),
        "config": args.config,
        "policy_mode": "stochastic" if args.stochastic else "deterministic",
        "records": records,
    }
    out_dir = exp_dir / "mav_failure_audit"
    out_json = _rel(args.output_json) if args.output_json else out_dir / "happo_mav_failure_modes.json"
    out_md = _rel(args.output_md) if args.output_md else out_dir / "happo_mav_failure_modes.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out_md, payload)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
