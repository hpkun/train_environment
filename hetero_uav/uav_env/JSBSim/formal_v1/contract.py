"""Frozen public contract for the formal heterogeneous 3v2 environment."""
from __future__ import annotations

import numpy as np

ENV_TYPE = "hetero_3v2_pure_happo_v1"
ACTION_DIM = 3
ACTION_ORDER = ("target_pitch", "target_heading", "target_speed")
ROLE_ORDER = ("mav", "attack_uav")
RED_IDS = ("red_0", "red_1", "red_2")
BLUE_IDS = ("blue_0", "blue_1")
SIM_FREQ = 60
DECISION_FREQ = 5
PHYSICS_STEPS_PER_ACTION = 12
MAX_STEPS = 1000

ALTITUDE_SCALE_M = 10_000.0
SPEED_SCALE_MPS = 400.0
POSITION_SCALE_M = 80_000.0
RELATIVE_SPEED_SCALE_MPS = 800.0

EGO_DIM = 11
ALLY_DIM = 11
ENEMY_DIM = 14
MISSILE_DIM = 7
MAX_ALLIES = 2
MAX_ENEMIES = 2
ACTOR_OBS_DIM = EGO_DIM + MAX_ALLIES * ALLY_DIM + MAX_ENEMIES * ENEMY_DIM + MISSILE_DIM
CRITIC_STATE_DIM = len(RED_IDS) * ACTOR_OBS_DIM


def require_action(action, agent_id: str) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"{agent_id} action must have shape ({ACTION_DIM},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{agent_id} action contains NaN or Inf")
    return np.clip(value, -1.0, 1.0)


def validate_formal_config(config: dict) -> None:
    required = {
        "env_type": ENV_TYPE,
        "scenario": "3v2",
        "sim_freq": SIM_FREQ,
        "decision_freq": DECISION_FREQ,
        "agent_interaction_steps": PHYSICS_STEPS_PER_ACTION,
        "max_steps": MAX_STEPS,
        "credit_mode": "shared_alive_team_mean",
        "action_mode": "continuous_high_level_pid",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"formal v1 requires {key}={expected!r}, got {config.get(key)!r}")
    forbidden = (
        "scripted_evasion", "blue_gcas", "role_local_credit", "brma_overlay",
        "communication_dropout", "random_hit", "action_trim_by_role",
    )
    enabled = [key for key in forbidden if config.get(key) not in (None, False, 0, {}, [])]
    if enabled:
        raise ValueError(f"formal v1 forbids legacy mechanisms: {enabled}")
