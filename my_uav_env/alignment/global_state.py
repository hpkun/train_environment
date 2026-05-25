"""Paper-style critic global state candidates from strict 10-dim team observations.

This module does not import the environment or JSBSim.  It provides pure
functions for flattening ``get_strict_team_observations()`` output into a
fixed-dim vector suitable as centralized critic input.

The current training critic uses flattened 11-dim engineering observations.
These candidates replace that with strict Table 1 / Table 2 team entities.
"""
from __future__ import annotations

import numpy as np


def flatten_strict_team_observations(
    team_obs: dict,
    agent_ids: list[str] | None = None,
    include_masks: bool = True,
) -> np.ndarray:
    """Flatten strict team observations into a 1-D critic input vector.

    Args:
        team_obs: ``{agent_id: (entities, mask, meta)}`` from
            ``UavCombatEnv.get_strict_team_observations()``.
        agent_ids: ordering of agent IDs.  If None, uses ``sorted(team_obs.keys())``.
        include_masks: if True, append entity_mask (float32) after each agent's
            flattened entities.

    Returns:
        1-D float32 array.

    Raises:
        KeyError: if a requested ``agent_id`` is missing from ``team_obs``.
    """
    if agent_ids is None:
        agent_ids = sorted(team_obs.keys())

    parts: list[np.ndarray] = []
    for aid in agent_ids:
        if aid not in team_obs:
            raise KeyError(
                f"agent_id {aid!r} not found in team_obs "
                f"(available: {sorted(team_obs.keys())})")
        entities, mask, _meta = team_obs[aid]
        parts.append(np.asarray(entities, dtype=np.float32).ravel())
        if include_masks:
            parts.append(np.asarray(mask, dtype=np.float32).ravel())
    return np.concatenate(parts).astype(np.float32)


def infer_strict_team_global_state_dim(
    num_red: int,
    num_blue: int,
    entity_dim: int = 10,
    include_masks: bool = True,
) -> int:
    """Return the flattened dimension for a strict team global state.

    Each red agent sees: 1 (ego) + (num_red - 1) (allies) + num_blue (enemies)
    = num_red + num_blue entities.  Each entity contributes ``entity_dim``
    floats, plus 1 mask float if ``include_masks``.
    """
    n_entities_per_agent = num_red + num_blue
    per_agent_dim = n_entities_per_agent * entity_dim
    if include_masks:
        per_agent_dim += n_entities_per_agent
    return num_red * per_agent_dim


def build_strict_team_global_state(
    team_obs: dict,
    num_red: int,
    num_blue: int,
    agent_prefix: str = "red",
    include_masks: bool = True,
) -> np.ndarray:
    """Build a fixed-dim strict team global state and validate the shape.

    Args:
        team_obs: strict team observations from the environment.
        num_red, num_blue: team sizes.
        agent_prefix: ``"red"`` or ``"blue"``.
        include_masks: forwarded to ``flatten_strict_team_observations``.

    Returns:
        1-D float32 array whose length equals
        ``infer_strict_team_global_state_dim(num_red, num_blue, ...)``.
    """
    n_agents = num_red if agent_prefix == "red" else num_blue
    agent_ids = [f"{agent_prefix}_{i}" for i in range(n_agents)]
    expected_dim = infer_strict_team_global_state_dim(
        num_red, num_blue, entity_dim=10, include_masks=include_masks,
    )
    flat = flatten_strict_team_observations(team_obs, agent_ids=agent_ids,
                                            include_masks=include_masks)
    if flat.shape[0] != expected_dim:
        raise ValueError(
            f"Expected global state dim {expected_dim}, got {flat.shape[0]}. "
            f"num_red={num_red}, num_blue={num_blue}, "
            f"include_masks={include_masks}")
    return flat


def describe_strict_global_state_layout(
    num_red: int,
    num_blue: int,
    entity_dim: int = 10,
    include_masks: bool = True,
) -> dict:
    """Return a human-readable layout description."""
    n_entities = num_red + num_blue
    per_agent = n_entities * entity_dim
    if include_masks:
        per_agent += n_entities
    return {
        "num_red": num_red,
        "num_blue": num_blue,
        "n_entities_per_red_agent": n_entities,
        "entity_dim": entity_dim,
        "include_masks": include_masks,
        "per_agent_dim": per_agent,
        "global_state_dim": num_red * per_agent,
        "schema": "strict_team_entities_flattened_for_critic_candidate",
    }
