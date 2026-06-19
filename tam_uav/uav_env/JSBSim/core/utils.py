"""Small numerical and configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def heading_to_unit(heading: float, pitch: float = 0.0) -> np.ndarray:
    cp = np.cos(pitch)
    return np.array([cp * np.cos(heading), cp * np.sin(heading), np.sin(pitch)],
                    dtype=np.float32)


def safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec) + 1e-8)


def los_angle(forward: np.ndarray, rel_pos: np.ndarray) -> float:
    denom = safe_norm(forward) * safe_norm(rel_pos)
    return float(np.arccos(np.clip(float(np.dot(forward, rel_pos)) / denom, -1.0, 1.0)))


class Box:
    """Small fallback Box compatible with the parts used by scripts/tests."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class DictSpace(dict):
    """Minimal dict space fallback."""

    def sample(self) -> dict:
        return {k: v.sample() for k, v in self.items()}


def make_box(low: float, high: float, shape: tuple[int, ...]):
    try:
        from gymnasium import spaces
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    except Exception:
        return Box(low=low, high=high, shape=shape, dtype=np.float32)


def make_dict_space(spaces_dict: dict):
    try:
        from gymnasium import spaces
        return spaces.Dict(spaces_dict)
    except Exception:
        return DictSpace(spaces_dict)
