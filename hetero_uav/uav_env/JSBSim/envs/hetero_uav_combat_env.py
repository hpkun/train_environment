"""Minimal MAV/UAV heterogeneous extension of the BRMA environment."""

from __future__ import annotations

from copy import deepcopy

from ..env import UavCombatEnv

FT_PER_M = 1.0 / 0.3048
FPS_PER_MPS = 1.0 / 0.3048

DEFAULT_AIRCRAFT_TYPE_PARAMS = {
    "mav": {
        "aircraft_model": "A-4",
        "role": "mav",
        "num_missiles": 2,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "attack_uav": {
        "aircraft_model": "f16",
        "role": "attack_uav",
        "num_missiles": 2,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "scout_uav": {
        "aircraft_model": "f16",
        "role": "scout_uav",
        "num_missiles": 0,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "interceptor_uav": {
        "aircraft_model": "f16",
        "role": "interceptor_uav",
        "num_missiles": 2,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
}


class HeteroUavCombatEnv(UavCombatEnv):
    """BRMA environment with per-agent aircraft model, role, and missile count.

    This first heterogeneous version deliberately preserves the original BRMA
    observation, reward, missile, action, PID, and termination logic.
    """

    def __init__(
        self,
        *args,
        red_agent_types: list[str] | None = None,
        blue_agent_types: list[str] | None = None,
        aircraft_type_params: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aircraft_type_params = deepcopy(DEFAULT_AIRCRAFT_TYPE_PARAMS)
        if aircraft_type_params:
            for name, params in aircraft_type_params.items():
                merged = dict(self.aircraft_type_params.get(name, {}))
                merged.update(params or {})
                self.aircraft_type_params[name] = merged

        self.red_agent_types = self._fit_agent_types(
            red_agent_types, self.max_num_red, ["mav", "attack_uav"]
        )
        self.blue_agent_types = self._fit_agent_types(
            blue_agent_types, self.max_num_blue, ["attack_uav", "attack_uav"]
        )
        self.agent_types: dict[str, str] = {}
        self.agent_roles: dict[str, str] = {}
        self.agent_models: dict[str, str] = {}
        self._refresh_agent_metadata()

    @staticmethod
    def _fit_agent_types(values: list[str] | None, count: int, default: list[str]) -> list[str]:
        selected = list(values) if values is not None else list(default)
        if len(selected) < count:
            selected.extend([selected[-1] if selected else "attack_uav"] * (count - len(selected)))
        return selected[:count]

    def _refresh_agent_metadata(self) -> None:
        self.agent_types.clear()
        self.agent_roles.clear()
        self.agent_models.clear()
        for i, aid in enumerate(self.red_ids):
            self._set_agent_metadata(aid, self.red_agent_types[i])
        for i, aid in enumerate(self.blue_ids):
            self._set_agent_metadata(aid, self.blue_agent_types[i])

    def _set_agent_metadata(self, agent_id: str, type_name: str) -> None:
        params = self.aircraft_type_params.get(type_name, self.aircraft_type_params["attack_uav"])
        self.agent_types[agent_id] = type_name
        self.agent_roles[agent_id] = str(params.get("role", type_name))
        self.agent_models[agent_id] = str(params.get("aircraft_model", "f16"))

    def _aircraft_model_for(self, agent_id: str, color: str, index: int) -> str:
        return self.agent_models.get(agent_id, "f16")

    def _num_missiles_for(self, agent_id: str) -> int:
        type_name = self.agent_types.get(agent_id, "attack_uav")
        params = self.aircraft_type_params.get(type_name, self.aircraft_type_params["attack_uav"])
        return int(params.get("num_missiles", self.num_missiles_per_plane))

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = super()._get_info(reward_components)
        info["agent_types"] = dict(self.agent_types)
        info["agent_roles"] = dict(self.agent_roles)
        info["agent_models"] = dict(self.agent_models)
        info["agent_init_offsets"] = {}
        for aid in self.agent_ids:
            info["agent_init_offsets"][aid] = self._init_offsets_for(aid)
        return info

    def _init_offsets_for(self, agent_id: str) -> dict:
        type_name = self.agent_types.get(agent_id, "attack_uav")
        params = self.aircraft_type_params.get(
            type_name, self.aircraft_type_params["attack_uav"])
        return {
            "altitude_offset_m": float(params.get("init_altitude_offset_m", 0.0)),
            "speed_offset_mps": float(params.get("init_speed_offset_mps", 0.0)),
        }

    def _make_init_state(self, color: str, index: int) -> dict:
        init = super()._make_init_state(color, index)

        agent_id = f"{color.lower()}_{index}"
        offsets = self._init_offsets_for(agent_id)
        alt_offset_m = offsets["altitude_offset_m"]
        speed_offset_mps = offsets["speed_offset_mps"]

        if alt_offset_m != 0.0:
            alt_offset_ft = alt_offset_m * FT_PER_M
            if "ic\\h-sl-ft" in init:
                init["ic\\h-sl-ft"] = float(init["ic\\h-sl-ft"]) + alt_offset_ft
            elif "ic/h-sl-ft" in init:
                init["ic/h-sl-ft"] = float(init["ic/h-sl-ft"]) + alt_offset_ft

        if speed_offset_mps != 0.0:
            speed_offset_fps = speed_offset_mps * FPS_PER_MPS
            if "ic\\u-fps" in init:
                init["ic\\u-fps"] = float(init["ic\\u-fps"]) + speed_offset_fps
            elif "ic/u-fps" in init:
                init["ic/u-fps"] = float(init["ic/u-fps"]) + speed_offset_fps

        return init


__all__ = ["HeteroUavCombatEnv"]
