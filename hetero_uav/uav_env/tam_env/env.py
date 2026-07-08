"""TAM-HAPPO direct-FCS diagnostic environment.

This module reuses the maintained JSBSim heterogeneous combat environment for
reward, observation, missile dynamics, launch gate, termination,
initialization, Tacview, and diagnostics. It only replaces the inherited PID
target-control actor interface with TAM-style direct flight-control-surface
commands.

The phase-1 tam_env action path intentionally does not execute the inherited
``_parse_actions`` missile-evasion action override or blue GCAS action
override. This keeps direct-FCS actor actions diagnostically pure. Future
direct-FCS missile evasion or GCAS logic should be implemented separately and
must not reuse the PID target-control override path.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


class TamCombatEnv(HeteroUavCombatEnv):
    """JSBSim hetero combat env with direct-FCS Box(4) actions.

    Action order per agent:
    ``[throttle, aileron, elevator, rudder]``.

    ``throttle`` maps from actor ``[-1, 1]`` to JSBSim
    ``[tam_throttle_min, tam_throttle_max]``. Control surfaces are clipped to
    ``[-tam_surface_limit, tam_surface_limit]`` and written to JSBSim FCS
    properties. Reward, observation, missile, blue-rule, initialization,
    termination, and logging behavior are inherited unchanged.

    This environment does not inherit the parent PID target-control action
    path, parent missile-evasion action override, or blue GCAS action override.
    It is a TAM-HAPPO direct-FCS diagnostic environment, not a full TAM-HAPPO
    reproduction.
    """

    THROTTLE_MIN = 0.4
    THROTTLE_MAX = 0.9

    def __init__(self, *args, **kwargs):
        self.strict_action_shape = bool(kwargs.pop("strict_action_shape", True))
        self.tam_throttle_min = float(kwargs.pop("tam_throttle_min", self.THROTTLE_MIN))
        self.tam_throttle_max = float(kwargs.pop("tam_throttle_max", self.THROTTLE_MAX))
        self.tam_surface_limit = float(kwargs.pop("tam_surface_limit", 1.0))
        self.tam_nan_action_policy = str(kwargs.pop("tam_nan_action_policy", "zero_clip"))
        self.tam_record_control_diagnostics = bool(
            kwargs.pop("tam_record_control_diagnostics", True)
        )
        self.tam_allow_non_strict_padding = bool(
            kwargs.pop("tam_allow_non_strict_padding", False)
        )
        self.tam_parent_action_overrides_enabled = bool(
            kwargs.pop("tam_parent_action_overrides_enabled", False)
        )
        self._validate_tam_config()

        super().__init__(*args, **kwargs)
        self.action_space = gymnasium.spaces.Dict({
            aid: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for aid in self.agent_ids
        })
        self._reset_tam_control_logs()

    def _validate_tam_config(self):
        if not (0.0 <= self.tam_throttle_min < self.tam_throttle_max <= 1.0):
            raise ValueError("tam_throttle_min/max must satisfy 0 <= min < max <= 1")
        if not (0.0 < self.tam_surface_limit <= 1.0):
            raise ValueError("tam_surface_limit must be in (0, 1]")
        if self.tam_nan_action_policy != "zero_clip":
            raise ValueError("tam_nan_action_policy currently only supports 'zero_clip'")
        if self.tam_parent_action_overrides_enabled:
            raise NotImplementedError(
                "tam_env phase 1 does not implement parent PID action overrides"
            )

    def _reset_tam_control_logs(self):
        self._last_tam_raw_actions: dict[str, list[float | str] | None] = {}
        self._last_tam_sanitized_actions: dict[str, list[float] | None] = {}
        self._last_tam_fcs_commands: dict[str, list[float] | None] = {}
        self._last_tam_action_warnings: dict[str, list[str]] = {}
        self._last_tam_applied_fcs: dict[str, list[float] | None] = {}
        self._last_effective_actions = {}
        self._last_action_trim_applied = {}

    def _apply_action_trim(self, actions: dict) -> dict:
        """Bypass inherited 3-D PID action trim for TAM direct-FCS actions."""

        return dict(actions)

    def _agent_alive(self, aid: str) -> bool:
        sim = self._get_sim(aid)
        return bool(sim is not None and sim.is_alive)

    @staticmethod
    def _json_value(value) -> float | str:
        try:
            val = float(value)
        except Exception:
            return str(value)
        if np.isnan(val):
            return "nan"
        if np.isposinf(val):
            return "inf"
        if np.isneginf(val):
            return "-inf"
        return round(val, 6)

    @classmethod
    def _json_list(cls, values) -> list[float | str]:
        return [cls._json_value(v) for v in np.asarray(values).reshape(-1)]

    @staticmethod
    def _round_list(values) -> list[float]:
        return [round(float(v), 6) for v in np.asarray(values).reshape(-1)]

    def _validate_tam_action(self, aid: str, action, alive: bool) -> np.ndarray | None:
        warnings = self._last_tam_action_warnings.setdefault(aid, [])
        if action is None:
            self._last_tam_raw_actions[aid] = None
            if not alive:
                warnings.append("inactive_missing_action")
                return None
            if self.strict_action_shape:
                raise ValueError(f"tam_env alive agent {aid} missing action")
            if not self.tam_allow_non_strict_padding:
                raise ValueError(f"tam_env alive agent {aid} missing action")
            warnings.append("missing_action_padded")
            return np.zeros(4, dtype=np.float32)

        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise ValueError(f"tam_env action for {aid} cannot be converted to float") from exc
        self._last_tam_raw_actions[aid] = self._json_list(arr)

        if not alive:
            warnings.append("inactive_action_ignored")
            return None
        if arr.size == 4:
            return arr
        if self.strict_action_shape:
            raise ValueError(
                f"tam_env action for {aid} must have shape=(4,), got size={arr.size}"
            )
        if not self.tam_allow_non_strict_padding:
            raise ValueError(
                f"tam_env action for {aid} must have shape=(4,), got size={arr.size}"
            )
        if arr.size < 4:
            warnings.append(f"action_padded_from_{arr.size}_to_4")
            return np.pad(arr, (0, 4 - arr.size), mode="constant").astype(np.float32)
        warnings.append(f"action_truncated_from_{arr.size}_to_4")
        return arr[:4].astype(np.float32)

    def _sanitize_tam_action(self, aid: str, action: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(action)):
            self._last_tam_action_warnings.setdefault(aid, []).append(
                "non-finite_action_zero_clip"
            )
        sanitized = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        sanitized = np.clip(sanitized, -1.0, 1.0).astype(np.float32)
        return sanitized

    def _tam_action_to_fcs(self, action: np.ndarray) -> tuple[float, float, float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 4:
            raise ValueError(f"tam_env FCS action must have shape=(4,), got size={action.size}")
        throttle = self.tam_throttle_min + (float(action[0]) + 1.0) * 0.5 * (
            self.tam_throttle_max - self.tam_throttle_min
        )
        aileron = float(np.clip(action[1], -self.tam_surface_limit, self.tam_surface_limit))
        elevator = float(np.clip(action[2], -self.tam_surface_limit, self.tam_surface_limit))
        rudder = float(np.clip(action[3], -self.tam_surface_limit, self.tam_surface_limit))
        return (
            float(np.clip(throttle, self.tam_throttle_min, self.tam_throttle_max)),
            aileron,
            elevator,
            rudder,
        )

    def _parse_actions(self, actions: dict) -> dict:
        """Parse TAM direct-FCS Box(4) actions into JSBSim command tuples."""

        self._reset_tam_control_logs()
        targets = {}
        for aid in self.agent_ids:
            alive = self._agent_alive(aid)
            validated = self._validate_tam_action(aid, actions.get(aid), alive)
            self._last_action_trim_applied[aid] = [0.0, 0.0, 0.0, 0.0]
            if validated is None:
                self._last_tam_sanitized_actions[aid] = None
                self._last_tam_fcs_commands[aid] = None
                self._last_effective_actions[aid] = None
                targets[aid] = None
                continue
            sanitized = self._sanitize_tam_action(aid, validated)
            fcs = self._tam_action_to_fcs(sanitized)
            self._last_tam_sanitized_actions[aid] = self._round_list(sanitized)
            self._last_tam_fcs_commands[aid] = self._round_list(fcs)
            self._last_effective_actions[aid] = self._round_list(sanitized)
            targets[aid] = fcs
        return targets

    def _apply_direct_fcs_controls(self, targets: dict):
        """Write TAM direct-FCS command tuples to JSBSim properties."""

        self._last_tam_applied_fcs = {}
        for aid in self.agent_ids:
            target = targets.get(aid)
            if target is None:
                self._last_tam_applied_fcs[aid] = None
                continue
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                self._last_tam_applied_fcs[aid] = None
                continue
            throttle, aileron, elevator, rudder = target
            sim.set_property_value("fcs/throttle-cmd-norm", float(throttle))
            sim.set_property_value("fcs/aileron-cmd-norm", float(aileron))
            sim.set_property_value("fcs/elevator-cmd-norm", float(elevator))
            sim.set_property_value("fcs/rudder-cmd-norm", float(rudder))
            self._last_tam_applied_fcs[aid] = self._round_list(target)

    def _apply_pid_controls(self, targets: dict):
        """Compatibility hook: tam_env intentionally overrides inherited PID control."""

        return self._apply_direct_fcs_controls(targets)

    def _read_fcs_property(self, sim, prop: str) -> float | None:
        try:
            value = float(sim.get_property_value(prop))
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _tam_control_diagnostics(self) -> dict[str, dict]:
        diagnostics: dict[str, dict] = {}
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            alive = bool(sim is not None and sim.is_alive)
            diag = {
                "alive": alive,
                "altitude_m": None,
                "speed_mps": None,
                "roll_rad": None,
                "pitch_rad": None,
                "heading_rad": None,
                "fcs_throttle_cmd": None,
                "fcs_aileron_cmd": None,
                "fcs_elevator_cmd": None,
                "fcs_rudder_cmd": None,
                "g_load": None,
            }
            if sim is not None:
                try:
                    diag["altitude_m"] = float(np.asarray(sim.get_geodetic()).reshape(-1)[2])
                except Exception:
                    pass
                try:
                    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
                    diag["speed_mps"] = float(np.linalg.norm(velocity))
                except Exception:
                    pass
                try:
                    rpy = np.asarray(sim.get_rpy(), dtype=np.float64).reshape(-1)
                    diag["roll_rad"] = float(rpy[0])
                    diag["pitch_rad"] = float(rpy[1])
                    diag["heading_rad"] = float(rpy[2])
                except Exception:
                    pass
                diag["fcs_throttle_cmd"] = self._read_fcs_property(sim, "fcs/throttle-cmd-norm")
                diag["fcs_aileron_cmd"] = self._read_fcs_property(sim, "fcs/aileron-cmd-norm")
                diag["fcs_elevator_cmd"] = self._read_fcs_property(sim, "fcs/elevator-cmd-norm")
                diag["fcs_rudder_cmd"] = self._read_fcs_property(sim, "fcs/rudder-cmd-norm")
                diag["g_load"] = self._read_fcs_property(sim, "accelerations/n-pilot-z-norm")
            diagnostics[aid] = diag
        return diagnostics

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = super()._get_info(reward_components)
        info.update({
            "tam_control_mode": "direct_fcs_box4",
            "tam_action_order": ["throttle", "aileron", "elevator", "rudder"],
            "tam_throttle_range": [self.tam_throttle_min, self.tam_throttle_max],
            "tam_surface_limit": self.tam_surface_limit,
            "tam_strict_action_shape": self.strict_action_shape,
            "tam_parent_action_overrides_enabled": False,
            "tam_note": (
                "tam_env uses direct FCS actions and does not execute inherited "
                "PID target-control, missile-evasion action override, or GCAS "
                "action override in this phase."
            ),
            "tam_raw_actions": dict(self._last_tam_raw_actions),
            "tam_sanitized_actions": dict(self._last_tam_sanitized_actions),
            "tam_fcs_commands": dict(self._last_tam_fcs_commands),
            "tam_applied_fcs": dict(self._last_tam_applied_fcs),
            "tam_action_warnings": dict(self._last_tam_action_warnings),
            "tam_control_diagnostics": (
                self._tam_control_diagnostics()
                if self.tam_record_control_diagnostics else {}
            ),
        })
        return info
