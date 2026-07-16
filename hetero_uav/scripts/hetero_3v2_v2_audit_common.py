"""Shared deterministic utilities for the formal-v1/v2 learnability audits."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.formal_v1.geometry import combat_geometry, unit
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from uav_env.JSBSim.formal_v1.sensing import red_track_sources
from uav_env.make_env import make_env


V1_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml"
AUDIT_DIR = ROOT / "outputs" / "hetero_3v2_v2_audit"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def perturbation(seed: int, scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        aid: {
            "lat_deg": float(rng.uniform(-0.004, 0.004) * scale),
            "lon_deg": float(rng.uniform(-0.004, 0.004) * scale),
            "altitude_m": float(rng.uniform(-250.0, 250.0) * scale),
            "speed_mps": float(rng.uniform(-15.0, 15.0) * scale),
            "yaw_deg": float(rng.uniform(-8.0, 8.0) * scale),
        }
        for aid in ("red_0", "red_1", "red_2", "blue_0", "blue_1")
    }


def hold_actions(env) -> dict[str, np.ndarray]:
    actions = {}
    for aid in env.red_ids:
        aircraft = env.aircraft[aid]
        yaw = float(aircraft.get_rpy()[2])
        speed = float(np.linalg.norm(aircraft.get_velocity()))
        actions[aid] = np.asarray([
            0.0,
            np.clip(yaw / np.pi, -1.0, 1.0),
            np.clip(2.0 * (speed - 102.0) / 306.0 - 1.0, -1.0, 1.0),
        ], np.float32)
    return actions


def safe_actions(env) -> dict[str, np.ndarray]:
    actions = hold_actions(env)
    for aid, action in actions.items():
        altitude = float(env.aircraft[aid].get_position()[2])
        action[0] = np.clip((6_000.0 - altitude) / 4_000.0, -0.15, 0.15)
        action[2] = np.clip(2.0 * (250.0 - 102.0) / 306.0 - 1.0, -1.0, 1.0)
    return actions


def escape_actions(env) -> dict[str, np.ndarray]:
    actions = safe_actions(env)
    for aid, action in actions.items():
        aircraft = env.aircraft[aid]
        enemies = [env.aircraft[x] for x in env.blue_ids if env.aircraft[x].is_alive]
        if enemies:
            centroid = np.mean([enemy.get_position() for enemy in enemies], axis=0)
            away = aircraft.get_position() - centroid
            action[1] = np.clip(math.atan2(away[1], away[0]) / np.pi, -1.0, 1.0)
    return actions


def geometry_actions(env, mode: str = "geometry") -> dict[str, np.ndarray]:
    actions = safe_actions(env)
    for aid in ("red_1", "red_2"):
        shooter = env.aircraft[aid]
        if not shooter.is_alive:
            continue
        tracks = red_track_sources(env, aid)
        targets = [
            env.aircraft[bid] for bid in env.blue_ids
            if env.aircraft[bid].is_alive and tracks[bid]["observable"]
        ]
        if not targets:
            continue
        target = min(targets, key=lambda item: np.linalg.norm(
            item.get_position() - shooter.get_position()))
        rel = target.get_position() - shooter.get_position()
        target_direction = unit(target.get_velocity())
        if mode in ("ta", "geometry"):
            aim = target.get_position() - 5_000.0 * target_direction
            desired = aim - shooter.get_position()
            if np.linalg.norm(desired[:2]) < 1_000.0:
                side = np.asarray([-target_direction[1], target_direction[0], 0.0])
                desired = rel + 4_000.0 * side
        elif mode == "ata":
            desired = rel
        elif mode == "range":
            desired = rel if np.linalg.norm(rel) > 7_000.0 else -rel
        else:
            desired = rel
        heading = math.atan2(desired[1], desired[0])
        horizontal = max(float(np.linalg.norm(desired[:2])), 1e-6)
        pitch = math.atan2(desired[2], horizontal)
        actions[aid] = np.asarray([
            np.clip(pitch / (np.pi / 2), -0.25, 0.25),
            np.clip(heading / np.pi, -1.0, 1.0),
            np.clip(2.0 * (300.0 - 102.0) / 306.0 - 1.0, -1.0, 1.0),
        ], np.float32)
    return actions


def controller_actions(env, name: str, rng: np.random.Generator | None = None):
    if name == "rule":
        return PaperGreedyOpponent().actions(env, "red")
    if name == "hold":
        return hold_actions(env)
    if name == "safe":
        return safe_actions(env)
    if name == "escape":
        return escape_actions(env)
    if name in ("ata", "ta", "range", "geometry"):
        return geometry_actions(env, name)
    if name == "random":
        rng = rng or np.random.default_rng(0)
        return {aid: rng.uniform(-1.0, 1.0, 3).astype(np.float32)
                for aid in env.red_ids}
    raise ValueError(f"unknown audit controller: {name}")


def _first(current, step):
    return step if current is None else current


def run_episode(config: str, seed: int, controller: str, max_steps: int = 1000,
                perturb: bool = True) -> dict:
    env = make_env(str(ROOT / config))
    rng = np.random.default_rng(seed + 100_000)
    _, info = env.reset(
        seed=seed,
        options={"audit_initial_perturbation": perturbation(seed) if perturb else {}},
    )
    initial = {}
    for aid in ("red_1", "red_2"):
        geometries = [combat_geometry(env.aircraft[aid], env.aircraft[bid])
                      for bid in env.blue_ids]
        best = min(geometries, key=lambda row: row["range_m"])
        initial[aid] = best
    first = {
        "blue_geometry": None, "blue_launch": None, "red_range": None,
        "red_ata": None, "red_ta": None, "red_geometry": None,
        "red_launch": None, "mav_death": None, "red_1_death": None,
        "red_2_death": None,
    }
    returns = {aid: 0.0 for aid in env.red_ids}
    dense_returns = {aid: 0.0 for aid in env.red_ids}
    event_returns = {aid: 0.0 for aid in env.red_ids}
    geometry_samples = 0
    gate_counts = {"range": 0, "ata": 0, "ta": 0, "geometry": 0}
    shared_steps = 0
    for step in range(max_steps):
        actions = controller_actions(env, controller, rng)
        _, rewards, _, _, info = env.step(actions)
        for aid, value in rewards.items():
            returns[aid] += float(value)
            component = info["reward_components"]["per_agent"][aid]
            dense_returns[aid] += float(component.get("dense", 0.0))
            event_returns[aid] += float(component.get("event", 0.0))
        shared_steps += int(
            info["reward_components"]["per_agent"]["red_0"].get(
                "shared_information", 0.0) > 0.0)
        for aid in ("red_1", "red_2"):
            gate = info["fire_gates"].get(aid, {})
            if env.aircraft[aid].is_alive and any(
                    env.aircraft[bid].is_alive for bid in env.blue_ids):
                geometry_samples += 1
                gate_counts["range"] += int(gate.get("range_ok", False))
                gate_counts["ata"] += int(gate.get("ata_ok", False))
                gate_counts["ta"] += int(gate.get("ta_ok", False))
                gate_counts["geometry"] += int(gate.get("geometry_ok", False))
                if gate.get("range_ok"):
                    first["red_range"] = _first(first["red_range"], step)
                if gate.get("ata_ok"):
                    first["red_ata"] = _first(first["red_ata"], step)
                if gate.get("ta_ok"):
                    first["red_ta"] = _first(first["red_ta"], step)
                if gate.get("geometry_ok"):
                    first["red_geometry"] = _first(first["red_geometry"], step)
        for aid in env.blue_ids:
            gate = info["fire_gates"].get(aid, {})
            if gate.get("geometry_ok"):
                first["blue_geometry"] = _first(first["blue_geometry"], step)
        for event in info["step_events"]:
            if event.get("event") != "launch":
                continue
            key = "red_launch" if str(event["shooter_id"]).startswith("red") else "blue_launch"
            first[key] = _first(first[key], step)
        for aid, key in (
                ("red_0", "mav_death"), ("red_1", "red_1_death"),
                ("red_2", "red_2_death")):
            if not env.aircraft[aid].is_alive:
                first[key] = _first(first[key], step)
        if info["team_done"]:
            break
    events = env.event_log
    row = {
        "seed": seed, "controller": controller, "steps": step + 1,
        "outcome": info["outcome"], "end_reason": info["end_reason"],
        "red_alive_final": info["red_alive"], "blue_alive_final": info["blue_alive"],
        "mav_alive_final": int(info["mav_alive"]),
        "red_launches": sum(e["event"] == "launch" and str(e["shooter_id"]).startswith("red")
                            for e in events),
        "blue_launches": sum(e["event"] == "launch" and str(e["shooter_id"]).startswith("blue")
                             for e in events),
        "red_hits": sum(e["event"] == "hit" and str(e["shooter_id"]).startswith("red")
                        for e in events),
        "blue_hits": sum(e["event"] == "hit" and str(e["shooter_id"]).startswith("blue")
                         for e in events),
        "team_return": float(np.mean(list(returns.values()))),
        "mav_return": returns["red_0"],
        "uav_return": float(np.mean([returns["red_1"], returns["red_2"]])),
        "dense_return": float(np.mean(list(dense_returns.values()))),
        "event_return": float(np.mean(list(event_returns.values()))),
        "shared_positive_step_rate": shared_steps / max(step + 1, 1),
        "range_rate": gate_counts["range"] / max(geometry_samples, 1),
        "ata_rate": gate_counts["ata"] / max(geometry_samples, 1),
        "ta_rate": gate_counts["ta"] / max(geometry_samples, 1),
        "geometry_rate": gate_counts["geometry"] / max(geometry_samples, 1),
        "geometry_samples": geometry_samples,
        **{f"first_{key}_step": value for key, value in first.items()},
        "initial_nearest_range_m": float(np.mean(
            [initial[aid]["range_m"] for aid in initial])),
        "initial_nearest_ata_rad": float(np.mean(
            [initial[aid]["ata_rad"] for aid in initial])),
        "initial_nearest_ta_rad": float(np.mean(
            [initial[aid]["ta_rad"] for aid in initial])),
        "finite": int(np.isfinite(np.asarray(list(returns.values()))).all()),
    }
    env.close()
    return row


def mean_ci(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(values.size) if values.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_low": mean - half,
            "ci95_high": mean + half}


def paired_effect(a: list[float], b: list[float]) -> dict:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    stats = mean_ci(delta.tolist())
    std = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    stats["cohens_dz"] = float(delta.mean() / std) if std > 1e-12 else 0.0
    return stats
