"""Sensor proxy models."""

from __future__ import annotations

from .aircraft import AircraftPlatform
from .utils import safe_norm


class SensorSuite:
    def __init__(self, config: dict):
        params = config.get("sensor_params", {})
        self.default_radar_range = float(params.get("default_radar_range", 90000.0))

    def can_detect(self, observer: AircraftPlatform, target: AircraftPlatform) -> bool:
        if not observer.alive or not target.alive:
            return False
        radar_range = observer.aircraft_type.radar_range or self.default_radar_range
        return safe_norm(target.position - observer.position) <= radar_range

    def missile_warning(self, _agent: AircraftPlatform) -> float:
        return 0.0
