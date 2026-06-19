"""Utilities for converting Dict observations into entity-wise tensors.

This module is intentionally not wired into the current vanilla MAPPO training
loop.  It prepares a stable tensor format for future EntityObservationEncoder
experiments while preserving the existing environment observation space.
"""
from __future__ import annotations

import numpy as np


def build_entity_observation(obs_np: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build an entity tensor and attention mask from one agent observation.

    Entity order is fixed:
    1. ego_state
    2. ally_states
    3. enemy_states

    The current environment uses an 11-dim engineering entity vector.  This
    helper keeps that representation unchanged instead of forcing the paper's
    Table 1 / Table 2 10-dim schema prematurely.

    Returns:
        entities: shape (N_entities, 11)
        entity_mask: shape (N_entities,), where 1 means invalid/dead/masked and
            0 means valid/kept.
    """
    ego = np.asarray(obs_np["ego_state"], dtype=np.float32).reshape(1, -1)
    allies = np.asarray(obs_np["ally_states"], dtype=np.float32)
    enemies = np.asarray(obs_np["enemy_states"], dtype=np.float32)

    if allies.size == 0:
        allies = allies.reshape(0, ego.shape[1])
    if enemies.size == 0:
        enemies = enemies.reshape(0, ego.shape[1])

    entities = np.concatenate([ego, allies, enemies], axis=0).astype(np.float32)
    entity_mask = np.zeros((entities.shape[0],), dtype=np.int64)
    entity_mask[np.all(np.isclose(entities, 0.0), axis=1)] = 1
    return entities, entity_mask


def infer_entity_layout(obs_np: dict) -> dict:
    """Infer entity counts and feature dimension from one agent observation."""
    ally_states = np.asarray(obs_np["ally_states"])
    enemy_states = np.asarray(obs_np["enemy_states"])
    entity_dim = int(np.asarray(obs_np["ego_state"]).shape[-1])
    n_allies = int(ally_states.shape[0])
    n_enemies = int(enemy_states.shape[0])
    return {
        "n_ego": 1,
        "n_allies": n_allies,
        "n_enemies": n_enemies,
        "n_entities": 1 + n_allies + n_enemies,
        "entity_dim": entity_dim,
    }


def flatten_entities_for_debug(entities: np.ndarray,
                               entity_mask: np.ndarray) -> np.ndarray:
    """Flatten entities and mask into one debug vector.

    This is for tests and sanity checks only.  It should not be used by the
    current vanilla MAPPO training loop.
    """
    entities = np.asarray(entities, dtype=np.float32)
    entity_mask = np.asarray(entity_mask, dtype=np.float32)
    return np.concatenate([entities.ravel(), entity_mask.ravel()]).astype(np.float32)
