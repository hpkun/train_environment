"""Fixed, reproducible 1v1 scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import NEU2LLA


ORIGIN = (120.0, 60.0, 0.0)


@dataclass(frozen=True)
class AircraftInitialState:
    altitude_m: float
    speed_mps: float
    heading_deg: float
    roll_deg: float
    pitch_deg: float
    longitude_deg: float
    latitude_deg: float


def tail_chase(randomize=False, rng=None):
    """Return red and blue initial conditions for the fixed tail chase."""
    rng = np.random.default_rng() if rng is None else rng
    north, east, altitude = 5000.0, 0.0, 6000.0
    red_speed, blue_speed = 270.0, 240.0
    if randomize:
        north += float(rng.uniform(-100.0, 100.0))
        east += float(rng.uniform(-50.0, 50.0))
        altitude += float(rng.uniform(-25.0, 25.0))
        red_speed += float(rng.uniform(-2.0, 2.0))
        blue_speed += float(rng.uniform(-2.0, 2.0))
    blue_lon, blue_lat, blue_alt = NEU2LLA(
        north, east, altitude, *ORIGIN)
    red = AircraftInitialState(
        6000.0, red_speed, 0.0, 0.0, 0.0, ORIGIN[0], ORIGIN[1])
    blue = AircraftInitialState(
        float(blue_alt), blue_speed, 0.0, 0.0, 0.0,
        float(blue_lon), float(blue_lat))
    return red, blue
