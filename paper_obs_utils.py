"""Placeholder adapter for paper-style 10-dim entity observations.

This module does not change the environment observation space.  It provides a
temporary 10-dim interface so future attention experiments can be wired against
the paper-shaped entity dimension before strict Table 1 / Table 2 reconstruction
is implemented.
"""
from __future__ import annotations

import numpy as np

from entity_obs_utils import build_entity_observation


def build_paper_entity_observation_from_env_obs(
    obs_np: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a placeholder 10-dim paper-style entity tensor from env obs.

    This adapter provides a 10-dim interface placeholder. It does not yet
    reconstruct the exact Table 1/Table 2 physical variables.

    Current env entity layout is:
    [dx_body, dy_body, dz_body, AO_signed, TA, R, V_tgt,
     sin_roll, cos_roll, sin_pitch, cos_pitch]

    The placeholder projection keeps the first 10 values:
    [dx_body, dy_body, dz_body, AO_signed, TA, R, V_tgt,
     sin_roll, cos_roll, sin_pitch]

    Entity order remains ego_state, ally_states, enemy_states.
    """
    current_entities, entity_mask = build_entity_observation(obs_np)
    paper_entities = current_entities[:, :10].astype(np.float32)
    return paper_entities, entity_mask.astype(np.int64)


def infer_paper_entity_layout(obs_np: dict) -> dict:
    ally_states = np.asarray(obs_np["ally_states"])
    enemy_states = np.asarray(obs_np["enemy_states"])
    n_allies = int(ally_states.shape[0])
    n_enemies = int(enemy_states.shape[0])
    return {
        "n_ego": 1,
        "n_allies": n_allies,
        "n_enemies": n_enemies,
        "n_entities": 1 + n_allies + n_enemies,
        "entity_dim": 10,
        "adapter": "placeholder_11_to_10",
    }


def compare_current_and_paper_adapter_shapes(obs_np: dict) -> dict:
    current_entities, current_mask = build_entity_observation(obs_np)
    paper_entities, paper_mask = build_paper_entity_observation_from_env_obs(obs_np)
    return {
        "current_entities_shape": tuple(current_entities.shape),
        "paper_entities_shape": tuple(paper_entities.shape),
        "current_mask_shape": tuple(current_mask.shape),
        "paper_mask_shape": tuple(paper_mask.shape),
    }
