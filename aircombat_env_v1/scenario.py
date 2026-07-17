"""Reproducible tail-chase scenario definitions."""

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


SCENARIO_MODES = (
    "fixed_tail_chase",
    "randomized_tail_chase",
    "offset_tail_chase",
)


def curriculum_scenario(rng):
    """Stage-2 engineering mixture: 20% fixed and 80% randomized."""
    return ("fixed_tail_chase" if rng.random() < 0.2
            else "randomized_tail_chase")


def _placed_blue(distance_m, lateral_m, altitude_m, red_heading_deg):
    heading = np.deg2rad(red_heading_deg)
    north = distance_m * np.cos(heading) - lateral_m * np.sin(heading)
    east = distance_m * np.sin(heading) + lateral_m * np.cos(heading)
    return NEU2LLA(north, east, altitude_m, *ORIGIN)


def make_scenario(mode="fixed_tail_chase", rng=None):
    """Return red and blue initial conditions for a supported tail chase."""
    if mode not in SCENARIO_MODES:
        raise ValueError(f"unsupported scenario mode: {mode}")
    rng = np.random.default_rng() if rng is None else rng
    distance, lateral, altitude_delta = 5000.0, 0.0, 0.0
    red_heading, blue_heading_delta = 0.0, 0.0
    red_speed, blue_speed = 270.0, 240.0
    if mode == "randomized_tail_chase":
        distance = float(rng.uniform(4000.0, 6500.0))
        lateral = float(rng.uniform(-1000.0, 1000.0))
        altitude_delta = float(rng.uniform(-300.0, 300.0))
        red_heading = float(rng.uniform(-10.0, 10.0))
        blue_heading_delta = float(rng.uniform(-10.0, 10.0))
        red_speed = float(rng.uniform(250.0, 285.0))
        blue_speed = float(rng.uniform(225.0, 255.0))
    elif mode == "offset_tail_chase":
        distance, lateral, altitude_delta = 5500.0, 1500.0, 300.0
        blue_heading_delta = 15.0
    blue_lon, blue_lat, blue_alt = _placed_blue(
        distance, lateral, 6000.0 + altitude_delta, red_heading)
    red = AircraftInitialState(
        6000.0, red_speed, red_heading, 0.0, 0.0, ORIGIN[0], ORIGIN[1])
    blue = AircraftInitialState(
        float(blue_alt), blue_speed, red_heading + blue_heading_delta,
        0.0, 0.0,
        float(blue_lon), float(blue_lat))
    return red, blue


def tail_chase(randomize=False, rng=None):
    """Backward-compatible fixed/randomized tail-chase helper."""
    mode = "randomized_tail_chase" if randomize else "fixed_tail_chase"
    return make_scenario(mode, rng)
