"""Export HAPPO reference checkpoint rollout to Tacview ACMI."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy
from algorithms.mappo.opponent_policy import OpponentPolicy


DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _model_path(exp_dir: Path, checkpoint: str) -> Path:
    return exp_dir / checkpoint / "model.pt"


def _load_meta(model: Path) -> dict:
    meta = model.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _acmi_id(agent_id: str) -> int:
    side, idx = agent_id.split("_", 1)
    return (100 if side == "red" else 200) + int(idx)


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _alive_counts(env) -> tuple[int, int, bool]:
    red = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    mav = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
    return red, blue, mav


def _entries(env) -> list[dict]:
    entries = []
    for aid in env.red_ids + env.blue_ids:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        lon, lat, alt = sim.get_geodetic()
        roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        entries.append({
            "acmi_id": _acmi_id(aid),
            "type": "Air+FixedWing",
            "lon": float(lon),
            "lat": float(lat),
            "alt": float(alt),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "name": f"{aid}_{env.agent_roles.get(aid, '')}_{env.agent_models.get(aid, '')}",
            "color": "Red" if aid.startswith("red_") else "Blue",
            "alive": bool(sim.is_alive),
        })
    return entries


def _team_done(terminated: dict, truncated: dict) -> bool:
    return all(terminated.values()) or all(truncated.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HAPPO reference checkpoint to ACMI")
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    exp_dir = _rel(args.experiment_dir)
    model = _rel(args.model) if args.model else _model_path(exp_dir, args.checkpoint)
    if not model.exists():
        print(f"checkpoint not found: {model}", file=sys.stderr)
        return 2
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from uav_env.JSBSim.render_tacview import TacviewLogger

    output = _rel(args.output) if args.output else exp_dir / "acmi" / f"{args.checkpoint}_3v2_episode0.acmi"
    summary_path = _rel(args.summary_json) if args.summary_json else output.with_name(output.stem + "_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(model)
    device = torch.device(args.device)
    policy = HAPPOReferencePolicy(
        int(meta.get("actor_obs_dim", 96)),
        int(meta.get("critic_state_dim", 480)),
    ).to(device)
    policy.load(model, map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()
    env = make_env(args.config, env_type="jsbsim_hetero")
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 33)
    logger = TacviewLogger(reference_time="2026-01-01T00:00:00Z")
    death_order: list[str] = []
    prev_alive: dict[str, bool] = {}
    missile_stats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
    prev_hits = {"red": 0, "blue": 0}
    mav_sat, uav_sat = [], []
    red0_pitch, red0_roll, red0_alt = [], [], []
    outcome = "unknown"

    try:
        obs, info = env.reset(seed=args.seed)
        logger.record_frame(0.0, _entries(env), [])
        step = 0
        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                for rid in env.red_ids
            ])
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(actor_obs, device=device),
                    roles=_role_ids(env),
                    critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                    deterministic=True,
                )
            acts_np = out["action"].cpu().numpy()
            mav_sat.append(float(np.mean(np.abs(acts_np[0:1]) >= 0.999)))
            if len(env.red_ids) > 1:
                uav_sat.append(float(np.mean(np.abs(acts_np[1:]) >= 0.999)))
            actions = {rid: acts_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            actions.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            for aid in env.agent_ids:
                fired = int(info.get(aid, {}).get("missiles_fired_this_step", 0)) if isinstance(info.get(aid, {}), dict) else 0
                if aid.startswith("red_"):
                    missile_stats["red_fired"] += fired
                else:
                    missile_stats["blue_fired"] += fired
            mt = info.get("__missile_term__", {})
            if isinstance(mt, dict):
                for side in ("red", "blue"):
                    total = int(mt.get(side, {}).get("hit", 0))
                    missile_stats[f"{side}_hits"] += max(total - prev_hits[side], 0)
                    prev_hits[side] = total
            for aid in env.red_ids + env.blue_ids:
                sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
                alive = bool(sim.is_alive)
                if prev_alive.get(aid, True) and not alive:
                    death_order.append(aid)
                prev_alive[aid] = alive
            red0 = env.red_planes.get("red_0")
            if red0 is not None:
                r, p, _y = red0.get_rpy()
                red0_roll.append(abs(math.degrees(float(r))))
                red0_pitch.append(abs(math.degrees(float(p))))
                red0_alt.append(float(red0.get_position()[2]))
            logger.record_frame(step * float(env.env_dt), _entries(env), [])
            if _team_done(terminated, truncated):
                break
            if step > int(getattr(env, "max_steps", 1000)) + 5:
                break
        red_alive, blue_alive, mav_alive = _alive_counts(env)
        if blue_alive == 0 and red_alive > 0:
            outcome = "red_win_elimination"
        elif red_alive == 0 and blue_alive > 0:
            outcome = "blue_win_elimination"
        elif red_alive == 0 and blue_alive == 0:
            outcome = "mutual_elimination_draw"
        elif step >= int(getattr(env, "max_steps", 1000)):
            outcome = "timeout"
        logger.write(str(output))
        summary = {
            "checkpoint": args.checkpoint,
            "model": str(model),
            "config": args.config,
            "outcome": outcome,
            "steps": step,
            "red_alive_final": red_alive,
            "blue_alive_final": blue_alive,
            "mav_alive": mav_alive,
            "death_order": death_order,
            "death_reason": "environment_info_not_explicit",
            "missiles_fired": missile_stats["red_fired"] + missile_stats["blue_fired"],
            "missile_hits": missile_stats["red_hits"] + missile_stats["blue_hits"],
            "red_missiles_fired": missile_stats["red_fired"],
            "blue_missiles_fired": missile_stats["blue_fired"],
            "red_missile_hits": missile_stats["red_hits"],
            "blue_missile_hits": missile_stats["blue_hits"],
            "red_0_max_abs_roll_deg": max(red0_roll) if red0_roll else 0.0,
            "red_0_max_abs_pitch_deg": max(red0_pitch) if red0_pitch else 0.0,
            "red_0_min_altitude": min(red0_alt) if red0_alt else None,
            "mav_action_saturation_rate": float(np.mean(mav_sat)) if mav_sat else 0.0,
            "uav_action_saturation_rate": float(np.mean(uav_sat)) if uav_sat else 0.0,
            "output_acmi": str(output),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        if hasattr(env, "close"):
            env.close()
    print(f"output_acmi: {output}")
    print(f"output_summary: {summary_path}")
    print(f"outcome: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
