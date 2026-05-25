"""Reward helper functions for paper-alignment audits.

These helpers document and test situation-reward Ta/Td formulas.  The
``current`` functions preserve historical behavior for audits; the fixed Ta
function is the normalized non-negative version used by the environment.
"""
from __future__ import annotations

import math
from typing import Callable

REWARD_VERSION = "fixed_ta_alt_eq17_3dlos_v1"
"""Reward version identifier for logs and evaluation outputs.

``fixed_ta_alt_eq17_3dlos_v1`` means:

1. situation reward Ta uses the ``fixed_ta_v1`` continuous, non-negative,
   normalized curve;
2. altitude reward uses a pairwise eq.17-style curve with the high-altitude
   0.1 tail;
3. situation reward geometry has switched from 2D horizontal AO/TA to
   3D body-x q_LOS and 3D Euclidean distance.

``fixed_ta_alt_eq17_v1`` and earlier logs should not be mixed with
``fixed_ta_alt_eq17_3dlos_v1`` results.
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


def pitch_penalty_current(theta_rad: float) -> float:
    """Current pitch penalty copied from env._pitch_penalty()."""
    theta = abs(theta_rad)
    if theta > math.pi / 3.0:
        return -1.0
    if theta > math.pi / 4.0:
        return -(theta / math.pi - 0.25) / 12.0
    return 0.0


def pitch_penalty_paper_candidate(theta_rad: float) -> float:
    """Candidate paper eq.15 pitch penalty.

    NEEDS PAPER TEXT VERIFICATION: this candidate currently mirrors the current
    implementation because eq.15 slope/scale is not fully verified.
    """
    return pitch_penalty_current(theta_rad)


def speed_penalty_current(mach: float) -> float:
    """Current speed penalty copied from env._speed_penalty() Mach logic."""
    if mach < 0.2:
        return -1.0
    if mach < 0.3:
        return -(0.3 - mach) / 0.1
    return 0.0


def speed_penalty_paper_candidate(mach: float) -> float:
    """Candidate paper eq.19 speed penalty.

    NEEDS PAPER TEXT VERIFICATION: this candidate currently mirrors the current
    implementation because eq.19 slope/scale is not fully verified.
    """
    return speed_penalty_current(mach)


def altitude_reward_current(dz_m: float) -> float:
    """Current altitude reward curve copied from env._altitude_reward().

    ``dz_m`` is ego altitude minus the mean enemy altitude in the current
    environment implementation.
    """
    h_min = 0.0
    h_att = 2000.0
    h_adv = 5000.0
    h_max = 10000.0

    if dz_m <= h_min:
        reward = 0.0
    elif dz_m < h_att:
        x = (dz_m - h_att) / (h_att - h_min)
        reward = 1.0 - x * x
    elif dz_m <= h_adv:
        reward = 1.0
    elif dz_m <= h_max:
        x = (dz_m - h_adv) / (h_max - h_adv)
        reward = 1.0 - x * x
    else:
        reward = 0.0
    return max(0.0, min(1.0, reward))


def altitude_reward_paper_eq17(dz_m: float) -> float:
    """Paper eq.17-style altitude curve with a high-altitude 0.1 tail.

    This follows the pass21 reading of paper eq.17 using the current project
    thresholds because exact h1/h2 and altitude constants still need visual
    verification against the paper.
    """
    h_min = 0.0
    h_att = 2000.0
    h_adv = 5000.0
    h_max = 10000.0
    tail = 0.1

    if dz_m <= h_min:
        reward = 0.0
    elif dz_m < h_att:
        x = (dz_m - h_att) / (h_att - h_min)
        reward = 1.0 - x * x
    elif dz_m <= h_adv:
        reward = 1.0
    elif dz_m <= h_max:
        x = (dz_m - h_adv) / (h_max - h_adv)
        reward = 1.0 - (1.0 - tail) * x * x
    else:
        reward = tail
    return max(0.0, min(1.0, reward))


def altitude_reward_paper_candidate(dz_m: float) -> float:
    """Compatibility alias for the paper eq.17-style altitude curve."""
    return altitude_reward_paper_eq17(dz_m)


def altitude_reward_pairwise_mean_eq17(
    ego_alt_m: float,
    enemy_altitudes_m: list[float],
) -> float:
    """Mean paper eq.17-style altitude reward over pairwise enemy deltas."""
    if not enemy_altitudes_m:
        return 0.0
    values = [
        altitude_reward_paper_eq17(ego_alt_m - enemy_alt)
        for enemy_alt in enemy_altitudes_m
    ]
    return float(sum(values) / len(values))


def altitude_reward_pairwise_mean_candidate(
    ego_alt_m: float,
    enemy_altitudes_m: list[float],
) -> float:
    """Compatibility alias for pairwise mean paper eq.17-style altitude reward."""
    return altitude_reward_pairwise_mean_eq17(ego_alt_m, enemy_altitudes_m)


def sample_altitude_table(
    func: Callable[[float], float],
) -> list[tuple[float, float]]:
    """Sample an altitude reward function at fixed diagnostic deltas."""
    dz_values = [-1000.0, 0.0, 1000.0, 2000.0,
                 5000.0, 7500.0, 10000.0, 12000.0]
    return [(dz, float(func(dz))) for dz in dz_values]


def sample_pitch_table(
    func: Callable[[float], float],
) -> list[tuple[float, float]]:
    """Sample a pitch penalty function at fixed diagnostic degrees."""
    degrees = [0.0, 30.0, 45.0, 50.0, 60.0, 70.0]
    return [(deg, float(func(math.radians(deg)))) for deg in degrees]


def sample_speed_table(
    func: Callable[[float], float],
) -> list[tuple[float, float]]:
    """Sample a speed penalty function at fixed diagnostic Mach values."""
    mach_values = [0.0, 0.1, 0.2, 0.25, 0.3, 0.5, 1.2]
    return [(mach, float(func(mach))) for mach in mach_values]


def sample_ta_table(func: Callable[[float], float]) -> list[tuple[float, float]]:
    """Sample a Ta function at fixed diagnostic angles."""
    angles = [0.0, 4.0, 10.0, 15.0, 20.0, 35.0, 40.0]
    return [(angle, float(func(angle))) for angle in angles]
