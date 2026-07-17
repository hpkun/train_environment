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
    "paper_nominal_1v1",
    "generalization_low",
    "generalization_medium",
    "generalization_high",
)

# Copied from tam_paper_env_v1_2v2.yaml.  These are official-project
# perturbation assumptions; the paper does not publish these bounds.
PERTURBATION_LEVELS = {
    "generalization_low": dict(altitude_m=500.0, latitude_deg=0.005,
                               longitude_deg=0.005, heading_deg=5.0,
                               speed_mps=20.0),
    "generalization_medium": dict(altitude_m=1000.0, latitude_deg=0.01,
                                  longitude_deg=0.01, heading_deg=10.0,
                                  speed_mps=35.0),
    "generalization_high": dict(altitude_m=1500.0, latitude_deg=0.015,
                                longitude_deg=0.015, heading_deg=15.0,
                                speed_mps=50.0),
}


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
    if mode == "paper_nominal_1v1" or mode in PERTURBATION_LEVELS:
        red_values = dict(altitude_m=6000.0, speed_mps=250.0,
                          heading_deg=0.0, longitude_deg=120.0,
                          latitude_deg=60.0)
        blue_values = dict(altitude_m=6000.0, speed_mps=250.0,
                           heading_deg=180.0, longitude_deg=120.0,
                           latitude_deg=60.2)
        if mode in PERTURBATION_LEVELS:
            bounds = PERTURBATION_LEVELS[mode]
            for values in (red_values, blue_values):
                for key, bound in bounds.items():
                    values[key] += float(rng.uniform(-bound, bound))
        red = AircraftInitialState(roll_deg=0.0, pitch_deg=0.0, **red_values)
        blue = AircraftInitialState(roll_deg=0.0, pitch_deg=0.0, **blue_values)
        return red, blue
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
