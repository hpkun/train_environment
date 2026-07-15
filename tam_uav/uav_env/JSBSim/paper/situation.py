"""Paper Eq. (10)-(12) geometry and target selection in SI/radians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0)))


@dataclass(frozen=True)
class SituationScore:
    relative_speed_mps: float
    relative_altitude_m: float
    distance_m: float
    ata_rad: float
    aa_rad: float
    e_angle: float
    e_distance: float
    e_height: float
    e_speed: float
    score: float


def assess_pair(ego_position: np.ndarray, ego_velocity: np.ndarray,
                target_position: np.ndarray, target_velocity: np.ndarray,
                maximum_attack_range_m: float, height_norm_m: float,
                speed_norm_mps: float) -> SituationScore:
    """Return the fixed-weight paper score from ego i to target j.

    LOS is ``target_position - ego_position``. ATA is the angle from ego
    velocity to LOS. In accordance with paper Eq. (10), AA is the angle from
    target velocity to the same LOS vector (the paper calls this its extension).
    """
    los = np.asarray(target_position, dtype=np.float64) - np.asarray(ego_position, dtype=np.float64)
    ego_v = np.asarray(ego_velocity, dtype=np.float64)
    target_v = np.asarray(target_velocity, dtype=np.float64)
    rel_v = ego_v - target_v
    distance = float(np.linalg.norm(los))
    rel_alt = float(ego_position[2] - target_position[2])
    ata = _angle(ego_v, los)
    aa = _angle(target_v, los)
    e_angle = 1.0 - (ata + aa) / (2.0 * np.pi)
    e_distance = 1.0 if distance <= maximum_attack_range_m else 0.0
    e_height = float(rel_alt / height_norm_m)
    e_speed = float(np.linalg.norm(rel_v) / speed_norm_mps)
    score = 0.35 * e_angle + 0.25 * e_distance + 0.20 * e_height + 0.20 * e_speed
    return SituationScore(float(np.linalg.norm(rel_v)), rel_alt, distance, ata, aa,
                          e_angle, e_distance, e_height, e_speed, float(score))


def select_best_target(scores: dict[str, SituationScore]) -> str | None:
    if not scores:
        return None
    return max(sorted(scores), key=lambda target_id: scores[target_id].score)
