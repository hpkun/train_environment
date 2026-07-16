"""V2 observation: V1 entities plus explicit red fire-control state."""
from __future__ import annotations

import numpy as np

from ..formal_v1.observation import build_actor_observation as build_v1_observation
from .contract import ACTOR_OBS_DIM

FIRE_CONTROL_DIM = 5
FIRE_CONTROL_FIELDS = (
    "missile_cooldown_remaining_norm",
    "fire_ready",
    "own_inflight_any",
    "team_inflight_to_enemy_slot_0",
    "team_inflight_to_enemy_slot_1",
)


def _launched_red_missiles(env):
    return [
        missile for missile in env.missiles
        if missile.is_launched and str(missile.shooter_id).startswith("red")
    ]


def build_fire_control_state(env, agent_id: str) -> np.ndarray:
    aircraft = env.aircraft[agent_id]
    is_mav = env.roles[agent_id] == "mav"
    missiles = _launched_red_missiles(env)
    if is_mav:
        cooldown_remaining = 0.0
        fire_ready = 0.0
        own_inflight = 0.0
    else:
        elapsed = float(env.sim_time_sec - env.last_launch_time[agent_id])
        remaining = max(0.0, float(env.attack_interval_sec) - elapsed)
        cooldown_remaining = remaining / max(float(env.attack_interval_sec), 1e-6)
        fire_ready = float(
            aircraft.is_alive
            and aircraft.num_left_missiles > 0
            and remaining <= 1e-9
        )
        own_inflight = float(any(
            missile.shooter_id == agent_id for missile in missiles))
    occupied = [
        float(any(missile.target_id == target_id for missile in missiles))
        for target_id in env.blue_ids
    ]
    state = np.asarray(
        [cooldown_remaining, fire_ready, own_inflight, *occupied],
        dtype=np.float32,
    )
    if state.shape != (FIRE_CONTROL_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"invalid formal v2 fire-control state for {agent_id}")
    return np.clip(state, 0.0, 1.0)


def build_actor_observation(env, agent_id: str) -> dict:
    base = build_v1_observation(env, agent_id)
    fire_control = build_fire_control_state(env, agent_id)
    flat = np.concatenate([base["flat"], fire_control]).astype(np.float32)
    if flat.shape != (ACTOR_OBS_DIM,) or not np.isfinite(flat).all():
        raise ValueError(
            f"invalid formal v2 actor observation for {agent_id}: {flat.shape}")
    return {**base, "fire_control": fire_control, "flat": flat}


def build_team_observations(env) -> tuple[dict, np.ndarray]:
    observations = {
        agent_id: build_actor_observation(env, agent_id)
        for agent_id in env.red_ids
    }
    critic = np.concatenate([
        observations[agent_id]["flat"]
        if env.aircraft[agent_id].is_alive
        else np.zeros(ACTOR_OBS_DIM, np.float32)
        for agent_id in env.red_ids
    ]).astype(np.float32)
    if not np.isfinite(critic).all():
        raise ValueError("non-finite formal v2 critic state")
    return observations, critic
