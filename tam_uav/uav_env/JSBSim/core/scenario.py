"""Scenario creation from YAML configs."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .aircraft import JSBSimAircraftPlatform, SimpleKinematicAircraftPlatform
from .aircraft_types import build_aircraft_types
from .geo import lla_to_local

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ScenarioBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.aircraft_types = build_aircraft_types(config)
        self.dynamics_backend = str(config.get("dynamics_backend", "simple"))
        self.model_root = Path(config.get("jsbsim_model_root", "uav_env/JSBSim/models"))
        if not self.model_root.is_absolute():
            self.model_root = PROJECT_ROOT / self.model_root
        self.reference_lat = float(config.get("reference_lat", 60.0))
        self.reference_lon = float(config.get("reference_lon", 120.0))
        self.reference_alt = float(config.get("reference_alt", 0.0))
        self.simulation_frequency = int(config.get("simulation_frequency", 60))

    def build(self, rng: np.random.Generator) -> list[AircraftPlatform]:
        agents: list[AircraftPlatform] = []
        agents.extend(self._build_side("red", self.config.get("red_agents", []), rng))
        agents.extend(self._build_side("blue", self.config.get("blue_agents", []), rng))
        return agents

    def _build_side(self, side: str, entries: list[dict], rng: np.random.Generator):
        result = []
        pos_range = self.config.get("initial_position_range", {})
        alt_range = self.config.get("initial_altitude_range", [5500.0, 6500.0])
        vel_range = self.config.get("initial_velocity_range", [220.0, 280.0])
        side_cfg = pos_range.get(side, {})
        x_range = side_cfg.get("x", [-8000.0, -6000.0] if side == "red" else [6000.0, 8000.0])
        y_range = side_cfg.get("y", [-1200.0, 1200.0])
        heading_default = 0.0 if side == "red" else 180.0
        for idx, entry in enumerate(entries):
            type_name = str(entry.get("type", entry.get("role", "attack_uav")))
            if type_name not in self.aircraft_types:
                raise KeyError(f"unknown aircraft type {type_name!r}")
            type_spec = self.aircraft_types[type_name]
            agent_id = str(entry.get("id", f"{side}_{idx}"))
            perturb = self._perturbation(rng)
            if "lon_deg" in entry and "lat_deg" in entry:
                position = lla_to_local(
                    float(entry["lon_deg"]) + perturb["longitude_deg"],
                    float(entry["lat_deg"]) + perturb["latitude_deg"],
                    float(entry.get("altitude_m", 6000.0)) + perturb["altitude_m"],
                    self.reference_lat, self.reference_lon, self.reference_alt)
            else:
                position = np.array([
                    float(entry.get("x", rng.uniform(*x_range))),
                    float(entry.get("y", rng.uniform(*y_range))),
                    float(entry.get("altitude", rng.uniform(*alt_range))),
                ], dtype=np.float32)
            speed = float(entry.get("speed_mps", entry.get("speed", rng.uniform(*vel_range)))) + perturb["speed_mps"]
            heading = np.deg2rad(float(entry.get("heading_deg", heading_default)) + perturb["heading_deg"])
            velocity = np.array([np.cos(heading) * speed, np.sin(heading) * speed, 0.0],
                                dtype=np.float32)
            if self.dynamics_backend == "jsbsim":
                platform = JSBSimAircraftPlatform(
                    agent_id, side, type_spec, position, velocity, heading,
                    model_root=self.model_root,
                    model_name=type_spec.aircraft_model,
                    reference_lat=self.reference_lat,
                    reference_lon=self.reference_lon,
                    reference_alt=self.reference_alt,
                    simulation_frequency=self.simulation_frequency,
                )
                platform.warmup_and_recenter(
                    float(self.config.get("inferred_parameters", {}).get(
                        "reset_warmup_seconds", 0.0)),
                    float(self.config.get("inferred_parameters", {}).get(
                        "reset_warmup_throttle", 0.9)))
            elif self.dynamics_backend == "simple":
                platform = SimpleKinematicAircraftPlatform(
                    agent_id, side, type_spec, position, velocity, heading)
            else:
                raise ValueError(f"unknown dynamics_backend {self.dynamics_backend!r}")
            platform.reset_runtime()
            result.append(platform)
        return result

    def _perturbation(self, rng: np.random.Generator) -> dict[str, float]:
        level = str(self.config.get("initial_perturbation", "none"))
        bounds = self.config.get("initial_perturbation_levels", {}).get(level)
        if not bounds:
            return {key: 0.0 for key in ("altitude_m", "longitude_deg", "latitude_deg",
                                         "heading_deg", "speed_mps")}
        return {key: float(rng.uniform(-float(value), float(value)))
                for key, value in bounds.items()}
