"""Reward helper functions for paper-alignment audits.

These helpers document and test situation-reward Ta/Td formulas.  The
``current`` functions preserve historical behavior for audits; the fixed Ta
function is the normalized non-negative version used by the environment.
"""
from __future__ import annotations

import math
from typing import Callable

REWARD_VERSION = "fixed_ta_v1"
"""Reward version identifier for logs and evaluation outputs.

``fixed_ta_v1`` means the situation reward Ta function has been changed from
the historical current piecewise formula to a continuous, non-negative,
normalized curve.  The legacy behavior remains available through
``ta_angle_advantage_current()`` for audits only.
"""


def ta_angle_advantage_current(q_deg: float) -> float:
    """Current angle-advantage formula copied from env._situation_reward().

    This preserves the existing behavior exactly, including the negative value
    near 15 degrees and the discontinuity after 15 degrees.  It exists so future
    paper-alignment changes can be compared against the current training signal.
    """
    if q_deg <= 4.0:
        return 1.0
    if q_deg <= 15.0:
        return 1.0 - 2.0 * (q_deg - 4.0) / 15.0
    if q_deg <= 35.0:
        return 1.0 - 3.5 * (q_deg - 15.0) / 180.0
    return 0.0


def td_distance_advantage_current(distance_m: float) -> float:
    """Current distance-advantage formula copied from env._situation_reward()."""
    distance_km = distance_m / 1000.0
    if distance_km <= 15.0:
        return 1.0
    return math.exp(1.0 - distance_km / 15.0)


def ta_angle_advantage_fixed(q_deg: float) -> float:
    """Normalized, continuous, non-negative angle-advantage curve.

    This keeps the current reward scale near [0, 1].  It does not adopt the
    possible paper ``10`` scale; that should be handled as a separate reward
    scale ablation if needed.
    """
    q = abs(q_deg)
    if q <= 4.0:
        value = 1.0
    elif q <= 15.0:
        value = 1.0 - 0.5 * (q - 4.0) / (15.0 - 4.0)
    elif q <= 35.0:
        value = 0.5 * (1.0 - (q - 15.0) / (35.0 - 15.0))
    else:
        value = 0.0
    return max(0.0, min(1.0, value))


def ta_angle_advantage_candidate_continuous(q_deg: float) -> float:
    """Backward-compatible alias for the fixed Ta curve.

    Kept for comparison scripts from pass18.  The curve is a normalized
    candidate, not a claim that the paper's original eq.20 uses this scale.
    """
    return ta_angle_advantage_fixed(q_deg)


def td_distance_advantage(distance_m: float) -> float:
    """Distance-advantage function used by the environment."""
    return td_distance_advantage_current(distance_m)


def sample_ta_table(func: Callable[[float], float]) -> list[tuple[float, float]]:
    """Sample a Ta function at fixed diagnostic angles."""
    angles = [0.0, 4.0, 10.0, 15.0, 20.0, 35.0, 40.0]
    return [(angle, float(func(angle))) for angle in angles]
