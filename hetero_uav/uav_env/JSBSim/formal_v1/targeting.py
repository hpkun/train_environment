"""TAM target assessment and minimal automatic fire-control contract."""
from __future__ import annotations

import numpy as np

from .geometry import combat_geometry
from .sensing import red_track_sources


def target_score(shooter, target, hmax_m=10_000.0, vmax_mps=400.0) -> float:
    geom = combat_geometry(shooter, target)
    e_angle = 1.0 - (geom["ata_rad"] + geom["ta_rad"]) / (2.0 * np.pi)
    e_distance = float(geom["range_m"] <= 14_000.0)
    e_height = np.clip((shooter.get_position()[2] - target.get_position()[2]) / hmax_m, -1.0, 1.0)
    e_speed = np.clip(np.linalg.norm(shooter.get_velocity() - target.get_velocity()) / vmax_mps, 0.0, 1.0)
    return float(0.35 * e_angle + 0.25 * e_distance + 0.20 * e_height + 0.20 * e_speed)


def select_target(env, shooter_id: str) -> str | None:
    shooter = env.aircraft[shooter_id]
    if not shooter.is_alive:
        return None
    if shooter_id.startswith("red"):
        tracks = red_track_sources(env, shooter_id)
        candidates = [bid for bid in env.blue_ids if env.aircraft[bid].is_alive and tracks[bid]["observable"]]
    else:
        candidates = [rid for rid in env.red_ids if env.aircraft[rid].is_alive]
    return max(candidates, key=lambda aid: target_score(shooter, env.aircraft[aid]), default=None)


def fire_gate(env, shooter_id: str, target_id: str | None) -> tuple[bool, dict]:
    shooter = env.aircraft[shooter_id]
    target = env.aircraft[target_id] if target_id else None
    observable = bool(target is not None and target.is_alive)
    if shooter_id.startswith("red") and target_id:
        observable = red_track_sources(env, shooter_id)[target_id]["observable"]
    geom = combat_geometry(shooter, target) if target is not None else {
        "range_m": np.inf, "ata_rad": np.pi, "ta_rad": 0.0}
    duplicate = bool(target_id and any(m.is_launched and m.target_id == target_id for m in env.missiles))
    cooldown = env.sim_time_sec - env.last_launch_time[shooter_id] >= env.attack_interval_sec
    allowed = bool(
        env.roles[shooter_id] == "attack_uav" and shooter.is_alive and observable
        and shooter.num_left_missiles > 0 and geom["range_m"] <= env.attack_range_m
        and geom["ata_rad"] <= env.launch_ata_rad and geom["ta_rad"] >= env.launch_ta_rad
        and cooldown and not duplicate
    )
    return allowed, {"range_m": geom["range_m"], "ata_rad": geom["ata_rad"],
                     "ta_rad": geom["ta_rad"], "observable": observable,
                     "cooldown_ready": cooldown, "duplicate_target_blocked": duplicate,
                     "allowed": allowed}
