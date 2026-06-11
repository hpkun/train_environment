"""Shared helpers for HAPPO MAV diagnostic scripts."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAPPO_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
DEFAULT_F16_SURROGATE_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate.yaml"
DEFAULT_EXPERIMENT_DIR = "outputs/happo_3v2_reference_200k"
SAFE_MAV_ACTION = np.asarray([0.0, 0.0, 0.3], dtype=np.float32)


def rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: str | Path, default):
    p = rel(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict) -> Path:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def write_md(path: str | Path, lines: list[str]) -> Path:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def checkpoint_path(exp_dir: str | Path, checkpoint: str) -> Path:
    model = rel(exp_dir) / checkpoint / "model.pt"
    if not model.exists():
        raise FileNotFoundError(f"checkpoint not found: {model}")
    return model


def load_policy(model: Path, device_name: str = "cpu"):
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from algorithms.happo import HAPPOReferencePolicy

    meta = read_json(model.parent / "meta.json", {})
    device = torch.device(device_name)
    policy = HAPPOReferencePolicy(
        int(meta.get("actor_obs_dim", 96)),
        int(meta.get("critic_state_dim", 480)),
    ).to(device)
    policy.load(model, map_location=device)
    policy.eval()
    return policy, device


def make_hetero_env(config: str, max_steps_override: int | None = None):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from uav_env import make_env

    env = make_env(config, env_type="jsbsim_hetero")
    if max_steps_override is not None:
        env.max_steps = int(max_steps_override)
    return env


def role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def sim_metrics(sim) -> dict:
    if sim is None:
        return {
            "altitude": None, "speed": None,
            "roll_deg": None, "pitch_deg": None, "heading_deg": None,
        }
    out = {"altitude": None, "speed": None, "roll_deg": None, "pitch_deg": None, "heading_deg": None}
    try:
        out["altitude"] = float(sim.get_geodetic()[2])
    except Exception:
        pass
    try:
        out["speed"] = float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))
    except Exception:
        pass
    try:
        roll, pitch, heading = sim.get_rpy()
        out["roll_deg"] = float(math.degrees(float(roll)))
        out["pitch_deg"] = float(math.degrees(float(pitch)))
        out["heading_deg"] = float(math.degrees(float(heading)))
    except Exception:
        pass
    return out


def update_missile_stats(stats: dict, info: dict, env, prev_hits: dict) -> None:
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


def summarize_episode(env, steps: int, truncated: dict, missile_stats: dict, red0_series: list[dict],
                      death_events: list[dict], red0_actions: list[np.ndarray]) -> dict:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    timeout = bool(all(truncated.values()) or steps >= int(getattr(env, "max_steps", 0)))
    if blue_alive == 0 and red_alive > 0:
        winner, reason = "red", "red_win_elimination"
    elif red_alive == 0 and blue_alive > 0:
        winner, reason = "blue", "blue_win_elimination"
    elif red_alive == 0 and blue_alive == 0:
        winner, reason = "draw", "mutual_elimination_draw"
    elif timeout:
        reason = "timeout"
        winner = "red_alive_advantage" if red_alive > blue_alive else "blue_alive_advantage" if blue_alive > red_alive else "draw"
    else:
        winner, reason = "draw", "other"

    red0_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
    red0_deaths = [event for event in death_events if event.get("agent_id") == "red_0"]
    first_dead = death_events[0].get("agent_id") if death_events else None
    altitudes = [x["altitude"] for x in red0_series if x.get("altitude") is not None]
    rolls = [abs(x["roll_deg"]) for x in red0_series if x.get("roll_deg") is not None]
    pitches = [abs(x["pitch_deg"]) for x in red0_series if x.get("pitch_deg") is not None]
    speeds = [x["speed"] for x in red0_series if x.get("speed") is not None]
    acts = np.stack(red0_actions) if red0_actions else np.zeros((0, 3), dtype=np.float32)
    return {
        "steps": int(steps),
        "winner": winner,
        "episode_end_reason": reason,
        "red_alive_final": red_alive,
        "blue_alive_final": blue_alive,
        "mav_alive": red0_alive,
        "mav_death": not red0_alive,
        "mav_first_death": bool(first_dead == "red_0"),
        "mav_death_step": red0_deaths[0].get("step") if red0_deaths else None,
        "mav_death_reason": red0_deaths[0].get("death_reason") if red0_deaths else "survived",
        "mav_death_reason_source": red0_deaths[0].get("death_reason_source") if red0_deaths else None,
        "blue_dead": max(len(env.blue_planes) - blue_alive, 0),
        "red_dead": max(len(env.red_planes) - red_alive, 0),
        "red_missile_hits": int(missile_stats.get("red_hits", 0)),
        "blue_missile_hits": int(missile_stats.get("blue_hits", 0)),
        "red_missiles_fired": int(missile_stats.get("red_fired", 0)),
        "blue_missiles_fired": int(missile_stats.get("blue_fired", 0)),
        "red0_max_abs_roll_deg": max(rolls) if rolls else 0.0,
        "red0_max_abs_pitch_deg": max(pitches) if pitches else 0.0,
        "red0_min_altitude": min(altitudes) if altitudes else None,
        "red0_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "red0_action_mean_abs": float(np.mean(np.abs(acts))) if acts.size else 0.0,
        "red0_action_saturation_rate": float(np.mean(np.abs(acts) >= 0.999)) if acts.size else 0.0,
        "death_events": death_events,
    }


def aggregate_records(records: list[dict]) -> dict:
    n = max(len(records), 1)
    reasons = Counter(r.get("mav_death_reason", "unknown") for r in records)
    death_steps = [r["mav_death_step"] for r in records if r.get("mav_death_step") is not None]
    return {
        "episodes": len(records),
        "red_win_rate": sum(1 for r in records if r["winner"] in {"red", "red_alive_advantage"}) / n,
        "blue_win_rate": sum(1 for r in records if r["winner"] in {"blue", "blue_alive_advantage"}) / n,
        "draw_rate": sum(1 for r in records if r["winner"] == "draw") / n,
        "timeout_rate": sum(1 for r in records if r["episode_end_reason"] == "timeout") / n,
        "mav_survival_rate": sum(1 for r in records if r["mav_alive"]) / n,
        "mav_death_rate": sum(1 for r in records if r["mav_death"]) / n,
        "mav_first_death_rate": sum(1 for r in records if r["mav_first_death"]) / n,
        "mav_mean_death_step": float(np.mean(death_steps)) if death_steps else None,
        "mav_death_reason_counts": dict(reasons),
        "blue_dead_mean": float(np.mean([r["blue_dead"] for r in records])) if records else 0.0,
        "red_missile_hits_mean": float(np.mean([r["red_missile_hits"] for r in records])) if records else 0.0,
        "red_alive_final_mean": float(np.mean([r["red_alive_final"] for r in records])) if records else 0.0,
        "blue_alive_final_mean": float(np.mean([r["blue_alive_final"] for r in records])) if records else 0.0,
        "red0_max_abs_roll_deg": max([r["red0_max_abs_roll_deg"] for r in records], default=0.0),
        "red0_max_abs_pitch_deg": max([r["red0_max_abs_pitch_deg"] for r in records], default=0.0),
        "red0_min_altitude": min([r["red0_min_altitude"] for r in records if r["red0_min_altitude"] is not None], default=None),
        "red0_action_mean_abs": float(np.mean([r["red0_action_mean_abs"] for r in records])) if records else 0.0,
        "red0_action_saturation_rate": float(np.mean([r["red0_action_saturation_rate"] for r in records])) if records else 0.0,
    }
