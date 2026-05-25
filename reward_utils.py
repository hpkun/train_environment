"""Reward helper functions for paper-alignment audits.

These helpers are intentionally not wired into ``UavCombatEnv`` yet.  They
document and test the current situation-reward Ta/Td formulas before any later
paper-formula replacement.
"""
from __future__ import annotations

import math
from typing import Callable


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


def ta_angle_advantage_candidate_continuous(q_deg: float) -> float:
    """Candidate continuous non-negative Ta curve for later comparison only.

    This is not the paper formula.  It must not be wired into the environment
    until eq.20 is checked against the original paper text.
    """
    if q_deg <= 4.0:
        value = 1.0
    elif q_deg <= 15.0:
        value = 1.0 - 0.5 * (q_deg - 4.0) / (15.0 - 4.0)
    elif q_deg <= 35.0:
        value = 0.5 * (35.0 - q_deg) / (35.0 - 15.0)
    else:
        value = 0.0
    return max(0.0, min(1.0, value))


def sample_ta_table(func: Callable[[float], float]) -> list[tuple[float, float]]:
    """Sample a Ta function at fixed diagnostic angles."""
    angles = [0.0, 4.0, 10.0, 15.0, 20.0, 35.0, 40.0]
    return [(angle, float(func(angle))) for angle in angles]
