"""Reward helper functions for paper-alignment audits.

These helpers document and test situation-reward Ta/Td formulas.  The
``current`` functions preserve historical behavior for audits; the fixed Ta
function implements the paper Eq.20 piecewise curve used by the environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

REWARD_VERSION = "paper_literal_eq15_eq20_ta1_tail01_joint_v4"
"""Reward version identifier for logs and evaluation outputs.

``paper_literal_eq15_eq20_ta1_tail01_joint_v4`` means:

1. pitch penalty uses the literal discontinuous Eq.15 ``/ 12`` segment;
2. situation reward Ta uses ``Ta=1`` when ``q_LOS < 4 deg`` and preserves the
   existing literal lower branches at and above 4 deg;
3. altitude reward uses the paper-explicit ``0.1`` high-altitude tail with
   separately versioned engineering thresholds;
4. situation reward geometry uses an inferred 3D observer-velocity-to-LOS
   angle and 3D Euclidean distance; the exact paper q_LOS geometry is unresolved.

All earlier reward-version logs should not be mixed with this version's
results.
"""


@dataclass(frozen=True)
class AltitudeRewardConfig:
    version: str = "eq17_engineering_thresholds_unbounded_tail_v2"
    h_min_m: float = 0.0
    h_att_m: float = 2000.0
    h_adv_m: float = 5000.0
    h_max_m: float = 10000.0
    d_att_max_m: float | None = None
    high_altitude_tail: float = 0.1

    def __post_init__(self):
        if not (self.h_min_m < self.h_att_m <= self.h_adv_m < self.h_max_m):
            raise ValueError(
                "AltitudeRewardConfig requires "
                "h_min_m < h_att_m <= h_adv_m < h_max_m")
        if self.d_att_max_m is not None and not self.h_max_m < self.d_att_max_m:
            raise ValueError(
                "AltitudeRewardConfig requires h_max_m < d_att_max_m")
        if not 0.0 <= self.high_altitude_tail <= 1.0:
            raise ValueError("high_altitude_tail must be in [0, 1]")
        if self.d_att_max_m is None and "unbounded_tail" not in self.version:
            raise ValueError(
                "an unbounded Eq.17 tail must be explicit in the config version")


DEFAULT_ALTITUDE_REWARD_CONFIG = AltitudeRewardConfig()


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
    if distance_m <= 15000.0:
        return 1.0
    return math.exp(1.0 - distance_m / 15000.0)


def ta_angle_advantage_fixed(q_deg: float) -> float:
    """Paper Eq.20 angle-advantage curve with its explicit first branch."""
    q = abs(q_deg)
    if q < 4.0:
        return 1.0
    if q <= 15.0:
        return 1.0 + 2.0 * (15.0 - q) / 15.0
    if q <= 35.0:
        return 1.0 - (q - 15.0) / (35.0 - 15.0)
    return 0.0


def ta_angle_advantage_candidate_continuous(q_deg: float) -> float:
    """Backward-compatible alias for the current paper Eq.20 Ta curve."""
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
    """Compatibility alias for the literal paper Eq.15 pitch penalty."""
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


def altitude_reward_paper_eq17(
    dz_m: float,
    config: AltitudeRewardConfig = DEFAULT_ALTITUDE_REWARD_CONFIG,
) -> float:
    """Eq.17 structure using explicit engineering thresholds.

    The paper does not publish the thresholds or polynomial coefficients. The
    two quadratic segments are derived from endpoint continuity. ``None`` for
    ``d_att_max_m`` explicitly selects an engineering unbounded-tail variant.
    """
    h_min = config.h_min_m
    h_att = config.h_att_m
    h_adv = config.h_adv_m
    h_max = config.h_max_m
    d_att_max_m = config.d_att_max_m
    tail = config.high_altitude_tail

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
    elif d_att_max_m is None or dz_m < d_att_max_m:
        reward = tail
    else:
        reward = 0.0
    return max(0.0, min(1.0, reward))


def altitude_reward_paper_candidate(dz_m: float) -> float:
    """Compatibility alias for the paper eq.17-style altitude curve."""
    return altitude_reward_paper_eq17(dz_m)


def altitude_reward_pairwise_sum_eq17(
    ego_alt_m: float,
    enemy_altitudes_m: list[float],
    config: AltitudeRewardConfig = DEFAULT_ALTITUDE_REWARD_CONFIG,
) -> float:
    """Sum Eq.17 pair rewards over alive enemies.

    The paper does not explicitly state Eq.17 multi-enemy aggregation.  The
    paper profile uses a sum, matching the explicit all-enemy sum in Eq.22;
    this interpretation is recorded in the paper specification metadata.
    """
    if not enemy_altitudes_m:
        return 0.0
    values = [
        altitude_reward_paper_eq17(ego_alt_m - enemy_alt, config=config)
        for enemy_alt in enemy_altitudes_m
    ]
    return float(sum(values))


def altitude_reward_pairwise_mean_eq17(
    ego_alt_m: float,
    enemy_altitudes_m: list[float],
    config: AltitudeRewardConfig = DEFAULT_ALTITUDE_REWARD_CONFIG,
) -> float:
    """Legacy engineering mean retained for old experiment compatibility."""
    if not enemy_altitudes_m:
        return 0.0
    return altitude_reward_pairwise_sum_eq17(
        ego_alt_m, enemy_altitudes_m, config=config) / len(enemy_altitudes_m)


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
