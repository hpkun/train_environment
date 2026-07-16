"""Public contract for the paper-aligned formal heterogeneous 3v2 V2."""
from __future__ import annotations

from ..formal_v1.contract import (
    ACTION_DIM, ACTION_ORDER, BLUE_IDS, DECISION_FREQ, MAX_STEPS,
    PHYSICS_STEPS_PER_ACTION, RED_IDS, ROLE_ORDER, SIM_FREQ,
)

ENV_TYPE = "hetero_3v2_pure_happo_v2"
OBSERVATION_CONTRACT = "formal_entity_fire_state_v2"
REWARD_CONTRACT_VERSION = "paper_aligned_role_reward_v3"
ACTOR_OBS_DIM = 73
CRITIC_STATE_DIM = len(RED_IDS) * ACTOR_OBS_DIM


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
            raise ValueError(
                f"formal v2 requires {key}={expected!r}, got {config.get(key)!r}")
    forbidden = (
        "scripted_evasion", "blue_gcas", "role_local_credit", "brma_overlay",
        "communication_dropout", "random_hit", "action_trim_by_role",
    )
    enabled = [
        key for key in forbidden
        if config.get(key) not in (None, False, 0, {}, [])
    ]
    if enabled:
        raise ValueError(f"formal v2 forbids legacy mechanisms: {enabled}")
