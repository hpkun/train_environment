"""Finite three-dimensional geometry helpers."""
from __future__ import annotations

import numpy as np


def unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 and np.isfinite(norm) else np.zeros(3)


def angle(a, b) -> float:
    ua, ub = unit(a), unit(b)
    if not ua.any() or not ub.any():
        return float(np.pi)
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


def combat_geometry(shooter, target) -> dict:
    rel = np.asarray(target.get_position()) - np.asarray(shooter.get_position())
    distance = float(np.linalg.norm(rel))
    ata = angle(shooter.get_velocity(), rel)
    ta = angle(target.get_velocity(), -rel)
    return {"relative_position": rel, "range_m": distance, "ata_rad": ata, "ta_rad": ta}
