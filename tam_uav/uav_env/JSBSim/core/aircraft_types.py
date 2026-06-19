"""Aircraft type definitions used to express heterogeneity."""

from __future__ import annotations

from dataclasses import dataclass, field


TYPE_IDS = {
    "mav": 0,
    "attack_uav": 1,
    "scout_uav": 2,
    "interceptor_uav": 3,
}


@dataclass(frozen=True)
class AircraftType:
    name: str
    aircraft_model: str
    model_path: str
    role: str
    radar_range: float
    missile_num: int
    max_speed_scale: float
    max_g: float
    reward_role: str
    control: dict[str, float] = field(default_factory=lambda: {
        "elevator_sign": 1.0,
        "aileron_sign": 1.0,
        "rudder_sign": 1.0,
        "throttle_sign": 1.0,
        "heading_sign": 1.0,
    })

    @property
    def type_id(self) -> int:
        return TYPE_IDS.get(self.role, TYPE_IDS.get(self.name, -1))


def build_aircraft_types(config: dict) -> dict[str, AircraftType]:
    raw_types = config.get("aircraft_type_params", {})
    result = {}
    for name, raw in raw_types.items():
        raw_control = raw.get("control", {})
        result[name] = AircraftType(
            name=name,
            aircraft_model=str(raw.get("aircraft_model", "F-16")),
            model_path=str(raw.get("model_path", "")),
            role=str(raw.get("role", name)),
            radar_range=float(raw.get("radar_range", 90000.0)),
            missile_num=int(raw.get("missile_num", 2)),
            max_speed_scale=float(raw.get("max_speed_scale", 1.0)),
            max_g=float(raw.get("max_g", 9.0)),
            reward_role=str(raw.get("reward_role", raw.get("role", name))),
            control={
                "elevator_sign": float(raw_control.get("elevator_sign", 1.0)),
                "aileron_sign": float(raw_control.get("aileron_sign", 1.0)),
                "rudder_sign": float(raw_control.get("rudder_sign", 1.0)),
                "throttle_sign": float(raw_control.get("throttle_sign", 1.0)),
                "heading_sign": float(raw_control.get("heading_sign", 1.0)),
            },
        )
    return result
