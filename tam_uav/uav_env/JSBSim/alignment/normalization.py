"""Normalize strict 10-dim entity observations for neural network input.

Strict entities from ``state_extractor`` carry raw physical units (metres,
radians, m/s).  Without scaling the critic and actor see large / unbounded
values that inflate loss magnitudes.

This module provides per-entity-type scale vectors so the self entity (row 0)
and relative entities (rows 1..N) are normalised independently to roughly
[-1, 1] before being fed into the network.
"""
from __future__ import annotations

import numpy as np

# ---- Scale vectors (per-column divisors) ----
# Self entity (Table 1): [x, y, h, V, roll, pitch, heading, alpha, beta, Vd]
SELF_SCALE = np.array([
    40000.0,   # x           m  →  battlefield half-size
    40000.0,   # y           m  →  battlefield half-size
    10000.0,   # h           m  →  altitude ceiling
    600.0,     # V           m/s →  max speed
    np.pi,     # roll        rad
    np.pi,     # pitch       rad
    np.pi,     # heading     rad
    np.pi,     # alpha       rad
    np.pi,     # beta        rad
    600.0,     # Vd          m/s →  max vertical speed
], dtype=np.float32)

# Relative entity (Table 2): [x_body, y_body, z_body, theta_v_body,
#   psi_v_body, V_target, theta_LOS_body, psi_LOS_body, q_LOS, d]
REL_SCALE = np.array([
    40000.0,   # x_body       m  →  battlefield half-size
    40000.0,   # y_body       m  →  battlefield half-size
    10000.0,   # z_body       m  →  altitude ceiling
    np.pi,     # theta_v_body rad
    np.pi,     # psi_v_body   rad
    600.0,     # V_target     m/s →  max speed
    np.pi,     # theta_LOS_body rad
    np.pi,     # psi_LOS_body rad
    np.pi,     # q_LOS        rad
    40000.0,   # d            m  →  battlefield half-size
], dtype=np.float32)

CLIP_MIN, CLIP_MAX = -5.0, 5.0


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def normalize_strict_entities(
    entities: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise a strict entity tensor row-by-row.

    Row 0 is the self entity, rows 1..N are relative entities.
    Dead / padded rows (mask == 1) are kept as all-zero.

    Args:
        entities: (N, 10) float array, not modified in-place.
        mask:      (N,) int array, 1 = invalid / dead / padded.

    Returns:
        (N, 10) float32 array, clipped to ``[CLIP_MIN, CLIP_MAX]``.
    """
    entities = np.asarray(entities, dtype=np.float32)
    out = entities.copy()
    n = out.shape[0]
    if n == 0:
        return out

    # Row 0: self
    out[0] = np.clip(out[0] / SELF_SCALE, CLIP_MIN, CLIP_MAX)

    # Rows 1..N: relative
    if n > 1:
        out[1:] = np.clip(out[1:] / REL_SCALE.reshape(1, -1), CLIP_MIN, CLIP_MAX)

    # Zero out padded rows
    if mask is not None:
        mask = np.asarray(mask, dtype=np.int64)
        pad_rows = np.where(mask != 0)[0]
        if len(pad_rows) > 0:
            out[pad_rows] = 0.0

    return out.astype(np.float32)


def normalize_strict_team_observations(team_obs: dict) -> dict:
    """Normalise entities in an entire team observation dict.

    Input / output format matches ``UavCombatEnv.get_strict_team_observations()``:
        ``{agent_id: (entities, mask, meta)}``
    """
    result = {}
    for aid, (entities, mask, meta) in team_obs.items():
        result[aid] = (
            normalize_strict_entities(entities, mask),
            np.asarray(mask, dtype=np.int64),
            meta,
        )
    return result
