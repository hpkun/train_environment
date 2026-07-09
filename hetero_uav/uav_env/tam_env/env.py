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

from copy import deepcopy

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
        self.tam_control_mode = str(kwargs.pop("tam_control_mode", "raw_direct_fcs"))
        self.tam_throttle_min = float(kwargs.pop("tam_throttle_min", self.THROTTLE_MIN))
        self.tam_throttle_max = float(kwargs.pop("tam_throttle_max", self.THROTTLE_MAX))
        self.tam_surface_limit = float(kwargs.pop("tam_surface_limit", 1.0))
        self.tam_bank_limit_rad = np.deg2rad(float(kwargs.pop("tam_bank_limit_deg", 70.0)))
        self.tam_neutral_nz_g = float(kwargs.pop("tam_neutral_nz_g", 1.0))
        self.tam_nz_delta_g = float(kwargs.pop("tam_nz_delta_g", 4.0))
        self.tam_nz_min_g = float(kwargs.pop("tam_nz_min_g", 0.0))
        self.tam_nz_max_g = float(kwargs.pop("tam_nz_max_g", 6.0))
        self.tam_yaw_rate_limit_rad_s = np.deg2rad(float(
            kwargs.pop("tam_yaw_rate_limit_deg_s", 20.0)
        ))
        self.tam_nz_sensor_sign = float(kwargs.pop("tam_nz_sensor_sign", -1.0))
        self.tam_nz_feedback_abs = bool(kwargs.pop("tam_nz_feedback_abs", False))
        self.tam_nz_error_clip_g = float(kwargs.pop("tam_nz_error_clip_g", 3.0))
        self.tam_aileron_sign = float(kwargs.pop("tam_aileron_sign", 1.0))
        self.tam_elevator_sign = float(kwargs.pop("tam_elevator_sign", -1.0))
        self.tam_rudder_sign = float(kwargs.pop("tam_rudder_sign", 1.0))
        self.tam_bank_kp = float(kwargs.pop("tam_bank_kp", 1.2))
        self.tam_roll_rate_kd = float(kwargs.pop("tam_roll_rate_kd", 0.3))
        self.tam_nz_kp = float(kwargs.pop("tam_nz_kp", 0.18))
        self.tam_pitch_rate_kd = float(kwargs.pop("tam_pitch_rate_kd", 0.05))
        self.tam_yaw_rate_kp = float(kwargs.pop("tam_yaw_rate_kp", 0.8))
        self.tam_yaw_rate_kd = float(kwargs.pop("tam_yaw_rate_kd", 0.2))
        self.tam_low_speed_warn_mps = float(kwargs.pop("tam_low_speed_warn_mps", 120.0))
        self.tam_high_altitude_warn_m = float(kwargs.pop("tam_high_altitude_warn_m", 10000.0))
        self.tam_low_altitude_warn_m = float(kwargs.pop("tam_low_altitude_warn_m", 2500.0))
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
        self._apply_tam_missile_count_override(kwargs)

        super().__init__(*args, **kwargs)
        self.action_space = gymnasium.spaces.Dict({
            aid: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for aid in self.agent_ids
        })
        self._reset_tam_control_logs()

    def _validate_tam_config(self):
        if not (0.0 <= self.tam_throttle_min < self.tam_throttle_max <= 1.0):
            raise ValueError("tam_throttle_min/max must satisfy 0 <= min < max <= 1")
        if self.tam_control_mode not in {"raw_direct_fcs", "maneuver_fcs"}:
            raise ValueError("tam_control_mode must be 'raw_direct_fcs' or 'maneuver_fcs'")
        if not (0.0 < self.tam_surface_limit <= 1.0):
            raise ValueError("tam_surface_limit must be in (0, 1]")
        if self.tam_bank_limit_rad <= 0.0:
            raise ValueError("tam_bank_limit_deg must be positive")
        if not (self.tam_nz_min_g < self.tam_nz_max_g):
            raise ValueError("tam_nz_min_g must be less than tam_nz_max_g")
        if self.tam_yaw_rate_limit_rad_s <= 0.0:
            raise ValueError("tam_yaw_rate_limit_deg_s must be positive")
        if self.tam_nz_error_clip_g <= 0.0:
            raise ValueError("tam_nz_error_clip_g must be positive")
        if self.tam_nan_action_policy != "zero_clip":
            raise ValueError("tam_nan_action_policy currently only supports 'zero_clip'")
        if self.tam_parent_action_overrides_enabled:
            raise NotImplementedError(
                "tam_env phase 1 does not implement parent PID action overrides"
            )

    @staticmethod
    def _apply_tam_missile_count_override(kwargs: dict):
        if "num_missiles_per_plane" not in kwargs or "aircraft_type_params" not in kwargs:
            return
        count = int(kwargs["num_missiles_per_plane"])
        params = deepcopy(kwargs.get("aircraft_type_params") or {})
        for value in params.values():
            if isinstance(value, dict):
                value["num_missiles"] = count
        kwargs["aircraft_type_params"] = params

    def _reset_tam_control_logs(self):
        self._last_tam_raw_actions: dict[str, list[float | str] | None] = {}
        self._last_tam_sanitized_actions: dict[str, list[float] | None] = {}
        self._last_tam_fcs_commands: dict[str, list[float] | None] = {}
        self._last_tam_control_commands: dict[str, list[float] | None] = {}
        self._last_tam_controller_terms: dict[str, dict] = {}
        self._last_tam_action_warnings: dict[str, list[str]] = {}
        self._last_tam_blue_legacy_pid_rule_adapter: dict[str, bool | str] = {}
        self._last_tam_blue_legacy_pid_applied: dict[str, bool] = {}
        self._last_tam_blue_legacy_pid_targets: dict[str, list[float] | None] = {}
        self._last_tam_applied_fcs: dict[str, list[float] | None] = {}
        self._last_effective_actions = {}
        self._last_action_trim_applied = {}

    def reset(self, *args, **kwargs):
        self._reset_tam_control_logs()
        obs, _info = super().reset(*args, **kwargs)
        self._reset_tam_control_logs()
        return obs, self._get_info()

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
            raw_arr = np.asarray(action, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"tam_env action for {aid} cannot be converted to float") from exc
        self._last_tam_raw_actions[aid] = self._json_list(raw_arr)

        if self.strict_action_shape and raw_arr.shape != (4,):
            raise ValueError(
                f"tam_env action for {aid} must have shape=(4,), got shape={raw_arr.shape}"
            )
        if (not self.strict_action_shape
                and not self.tam_allow_non_strict_padding
                and raw_arr.shape != (4,)):
            raise ValueError(
                f"tam_env action for {aid} must have shape=(4,), got shape={raw_arr.shape}"
            )
        arr = raw_arr.reshape(-1)

        if not alive:
            warnings.append("inactive_action_ignored")
            return None
        if raw_arr.shape != (4,):
            warnings.append(f"non_strict_reshaped_from_{raw_arr.shape}_to_flat")
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

    def _tam_maneuver_action_to_commands(
        self, action: np.ndarray
    ) -> tuple[float, float, float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 4:
            raise ValueError(f"tam_env maneuver action must have shape=(4,), got size={action.size}")
        throttle = self.tam_throttle_min + (float(action[0]) + 1.0) * 0.5 * (
            self.tam_throttle_max - self.tam_throttle_min
        )
        bank_cmd = float(action[1]) * self.tam_bank_limit_rad
        nz_cmd = self.tam_neutral_nz_g + float(action[2]) * self.tam_nz_delta_g
        nz_cmd = float(np.clip(nz_cmd, self.tam_nz_min_g, self.tam_nz_max_g))
        yaw_rate_cmd = float(action[3]) * self.tam_yaw_rate_limit_rad_s
        return (
            float(np.clip(throttle, self.tam_throttle_min, self.tam_throttle_max)),
            float(bank_cmd),
            float(nz_cmd),
            float(yaw_rate_cmd),
        )

    def _tam_action_to_control_commands(
        self, action: np.ndarray
    ) -> tuple[float, float, float, float]:
        if self.tam_control_mode == "maneuver_fcs":
            return self._tam_maneuver_action_to_commands(action)
        return self._tam_action_to_fcs(action)

    def _legacy_pid_rule_action_to_target(self, aid: str, action) -> tuple[float, float, float]:
        """Convert a blue rule 3-D PID action to inherited PID targets.

        This adapter is explicit and blue-only.  It exists so legacy rule
        opponents can be used for tam_env smoke tests without padding a 3-D
        action into the red policy's 4-D maneuver-FCS action contract.
        """

        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size != 3:
            raise ValueError(f"legacy blue PID action for {aid} must have size=3, got {act.size}")
        act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
        act = np.clip(act, -1.0, 1.0)
        target_velocity = self.VELOCITY_MIN + (float(act[2]) + 1.0) / 2.0 * (
            self.VELOCITY_MAX - self.VELOCITY_MIN
        )
        target_pitch = float(act[0]) * np.deg2rad(self.PITCH_DEG)
        target_heading = float(act[1]) * np.pi
        return (target_pitch, target_heading, target_velocity)

    def _parse_actions(self, actions: dict) -> dict:
        """Parse TAM direct-FCS Box(4) actions into JSBSim command tuples."""

        self._reset_tam_control_logs()
        targets = {}
        for aid in self.agent_ids:
            alive = self._agent_alive(aid)
            self._last_action_trim_applied[aid] = [0.0, 0.0, 0.0, 0.0]
            raw_action = actions.get(aid)
            if aid.startswith("blue_") and raw_action is not None:
                raw_arr = np.asarray(raw_action, dtype=np.float32)
                if raw_arr.shape == (3,):
                    self._last_tam_raw_actions[aid] = self._json_list(raw_arr)
                    self._last_tam_sanitized_actions[aid] = None
                    self._last_tam_fcs_commands[aid] = None
                    self._last_effective_actions[aid] = None
                    self._last_tam_control_commands[aid] = None
                    if not alive:
                        self._last_tam_action_warnings.setdefault(aid, []).append(
                            "blue_legacy_pid_rule_adapter_inactive_ignored"
                        )
                        self._last_tam_blue_legacy_pid_rule_adapter[aid] = "inactive_ignored"
                        self._last_tam_blue_legacy_pid_applied[aid] = False
                        self._last_tam_blue_legacy_pid_targets[aid] = None
                        targets[aid] = None
                        continue
                    self._last_tam_action_warnings.setdefault(aid, []).append(
                        "blue_legacy_pid_rule_adapter"
                    )
                    pid_target = self._legacy_pid_rule_action_to_target(aid, raw_arr)
                    self._last_tam_blue_legacy_pid_rule_adapter[aid] = True
                    self._last_tam_blue_legacy_pid_applied[aid] = True
                    self._last_tam_blue_legacy_pid_targets[aid] = self._round_list(pid_target)
                    targets[aid] = ("legacy_pid_rule_adapter", pid_target)
                    continue
            validated = self._validate_tam_action(aid, raw_action, alive)
            if validated is None:
                self._last_tam_sanitized_actions[aid] = None
                self._last_tam_fcs_commands[aid] = None
                self._last_effective_actions[aid] = None
                self._last_tam_blue_legacy_pid_rule_adapter[aid] = False
                self._last_tam_blue_legacy_pid_applied[aid] = False
                targets[aid] = None
                continue
            sanitized = self._sanitize_tam_action(aid, validated)
            fcs = self._tam_action_to_control_commands(sanitized)
            self._last_tam_sanitized_actions[aid] = self._round_list(sanitized)
            self._last_tam_fcs_commands[aid] = self._round_list(fcs)
            self._last_tam_control_commands[aid] = self._round_list(fcs)
            self._last_effective_actions[aid] = self._round_list(sanitized)
            self._last_tam_blue_legacy_pid_rule_adapter[aid] = False
            self._last_tam_blue_legacy_pid_applied[aid] = False
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

    @staticmethod
    def _wrap_angle_rad(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    def _read_property_or_zero(self, sim, prop: str) -> float:
        value = self._read_fcs_property(sim, prop)
        return 0.0 if value is None else float(value)

    def _effective_nz_g(self, raw_nz_sensor_g: float) -> float:
        raw = float(raw_nz_sensor_g)
        if self.tam_nz_feedback_abs:
            return abs(raw)
        return self.tam_nz_sensor_sign * raw

    def _apply_maneuver_fcs_controls(self, targets: dict):
        """Convert maneuver commands to direct FCS outputs using ownship state."""

        self._last_tam_applied_fcs = {}
        self._last_tam_controller_terms = {}
        for aid in self.agent_ids:
            command = targets.get(aid)
            if command is None:
                self._last_tam_applied_fcs[aid] = None
                self._last_tam_controller_terms[aid] = {}
                continue
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                self._last_tam_applied_fcs[aid] = None
                self._last_tam_controller_terms[aid] = {}
                continue

            throttle, bank_cmd, nz_cmd, yaw_rate_cmd = command
            try:
                roll = float(np.asarray(sim.get_rpy(), dtype=np.float64).reshape(-1)[0])
            except Exception:
                roll = 0.0
            p_rate = self._read_property_or_zero(sim, "velocities/p-rad_sec")
            q_rate = self._read_property_or_zero(sim, "velocities/q-rad_sec")
            r_rate = self._read_property_or_zero(sim, "velocities/r-rad_sec")
            raw_nz_sensor = self._read_property_or_zero(sim, "accelerations/n-pilot-z-norm")
            effective_nz = self._effective_nz_g(raw_nz_sensor)

            bank_error = self._wrap_angle_rad(float(bank_cmd) - roll)
            raw_aileron = self.tam_aileron_sign * (
                self.tam_bank_kp * bank_error - self.tam_roll_rate_kd * p_rate
            )
            nz_error = float(nz_cmd) - effective_nz
            nz_error_clipped = float(np.clip(
                nz_error, -self.tam_nz_error_clip_g, self.tam_nz_error_clip_g
            ))
            raw_elevator = self.tam_elevator_sign * (
                self.tam_nz_kp * nz_error_clipped - self.tam_pitch_rate_kd * q_rate
            )
            raw_rudder = self.tam_rudder_sign * (
                self.tam_yaw_rate_kp * (float(yaw_rate_cmd) - r_rate)
                - self.tam_yaw_rate_kd * r_rate
            )

            aileron = float(np.clip(raw_aileron, -self.tam_surface_limit, self.tam_surface_limit))
            elevator = float(np.clip(raw_elevator, -self.tam_surface_limit, self.tam_surface_limit))
            rudder = float(np.clip(raw_rudder, -self.tam_surface_limit, self.tam_surface_limit))
            sim.set_property_value("fcs/throttle-cmd-norm", float(throttle))
            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            self._last_tam_applied_fcs[aid] = self._round_list((throttle, aileron, elevator, rudder))
            self._last_tam_controller_terms[aid] = {
                "bank_error_rad": float(bank_error),
                "roll_rate_rad_s": float(p_rate),
                "nz_cmd_g": float(nz_cmd),
                "raw_nz_sensor_g": float(raw_nz_sensor),
                "effective_nz_g": float(effective_nz),
                "nz_current_g": float(effective_nz),
                "nz_error_g": float(nz_error),
                "nz_error_clipped_g": float(nz_error_clipped),
                "pitch_rate_rad_s": float(q_rate),
                "yaw_rate_cmd_rad_s": float(yaw_rate_cmd),
                "yaw_rate_rad_s": float(r_rate),
                "tam_aileron_sign": float(self.tam_aileron_sign),
                "tam_elevator_sign": float(self.tam_elevator_sign),
                "tam_rudder_sign": float(self.tam_rudder_sign),
                "raw_aileron": float(raw_aileron),
                "raw_elevator": float(raw_elevator),
                "raw_rudder": float(raw_rudder),
                "clipped_aileron": float(aileron),
                "clipped_elevator": float(elevator),
                "clipped_rudder": float(rudder),
            }

    def _apply_tam_controls(self, targets: dict):
        pid_targets = {
            aid: target[1]
            for aid, target in targets.items()
            if isinstance(target, tuple)
            and len(target) == 2
            and target[0] == "legacy_pid_rule_adapter"
        }
        if pid_targets:
            HeteroUavCombatEnv._apply_pid_controls(self, pid_targets)
        targets = {
            aid: target
            for aid, target in targets.items()
            if not (
                isinstance(target, tuple)
                and len(target) == 2
                and target[0] == "legacy_pid_rule_adapter"
            )
        }
        if self.tam_control_mode == "maneuver_fcs":
            return self._apply_maneuver_fcs_controls(targets)
        return self._apply_direct_fcs_controls(targets)

    def _apply_pid_controls(self, targets: dict):
        """Compatibility hook: tam_env intentionally overrides inherited PID control."""

        return self._apply_tam_controls(targets)

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
                "true_speed_mps": None,
                "airspeed_mps": None,
                "alpha_rad": None,
                "beta_rad": None,
                "p_rad_s": None,
                "q_rad_s": None,
                "r_rad_s": None,
                "low_speed_flag": None,
                "high_altitude_flag": None,
                "low_altitude_flag": None,
                "roll_rad": None,
                "pitch_rad": None,
                "heading_rad": None,
                "fcs_throttle_cmd": None,
                "fcs_aileron_cmd": None,
                "fcs_elevator_cmd": None,
            "fcs_rudder_cmd": None,
            "g_load": None,
            "g_load_total": None,
            "g_load_x": None,
            "g_load_y": None,
            "g_load_z": None,
        }
            if sim is not None:
                try:
                    diag["altitude_m"] = float(np.asarray(sim.get_geodetic()).reshape(-1)[2])
                except Exception:
                    pass
                try:
                    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
                    true_speed = float(np.linalg.norm(velocity))
                    diag["speed_mps"] = true_speed
                    diag["true_speed_mps"] = true_speed
                except Exception:
                    pass
                diag["airspeed_mps"] = self._read_fcs_property(sim, "velocities/vc-mps")
                diag["alpha_rad"] = self._read_fcs_property(sim, "aero/alpha-rad")
                diag["beta_rad"] = self._read_fcs_property(sim, "aero/beta-rad")
                diag["p_rad_s"] = self._read_fcs_property(sim, "velocities/p-rad_sec")
                diag["q_rad_s"] = self._read_fcs_property(sim, "velocities/q-rad_sec")
                diag["r_rad_s"] = self._read_fcs_property(sim, "velocities/r-rad_sec")
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
                gx = self._read_fcs_property(sim, "accelerations/n-pilot-x-norm")
                gy = self._read_fcs_property(sim, "accelerations/n-pilot-y-norm")
                gz = self._read_fcs_property(sim, "accelerations/n-pilot-z-norm")
                diag["g_load_x"] = gx
                diag["g_load_y"] = gy
                diag["g_load_z"] = gz
                if gx is not None and gy is not None and gz is not None:
                    total = float(np.sqrt(gx * gx + gy * gy + gz * gz))
                    diag["g_load_total"] = total
                    diag["g_load"] = total
                speed_for_flag = diag["true_speed_mps"]
                alt_for_flag = diag["altitude_m"]
                if speed_for_flag is not None:
                    diag["low_speed_flag"] = bool(speed_for_flag < self.tam_low_speed_warn_mps)
                if alt_for_flag is not None:
                    diag["high_altitude_flag"] = bool(alt_for_flag > self.tam_high_altitude_warn_m)
                    diag["low_altitude_flag"] = bool(alt_for_flag < self.tam_low_altitude_warn_m)
            diagnostics[aid] = diag
        return diagnostics

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = super()._get_info(reward_components)
        info.update({
            "tam_control_mode": self.tam_control_mode,
            "tam_action_order": (
                ["throttle", "roll_cmd", "pitch_or_load_cmd", "yaw_cmd"]
                if self.tam_control_mode == "maneuver_fcs"
                else ["throttle", "aileron", "elevator", "rudder"]
            ),
            "tam_command_semantics": (
                ["throttle", "bank_cmd_rad", "nz_cmd_g", "yaw_rate_cmd_rad_s"]
                if self.tam_control_mode == "maneuver_fcs"
                else ["throttle", "aileron", "elevator", "rudder"]
            ),
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
            "tam_control_commands": dict(self._last_tam_control_commands),
            "tam_fcs_commands": dict(self._last_tam_fcs_commands),
            "tam_applied_fcs": dict(self._last_tam_applied_fcs),
            "tam_action_warnings": dict(self._last_tam_action_warnings),
            "tam_blue_legacy_pid_rule_adapter": dict(
                self._last_tam_blue_legacy_pid_rule_adapter
            ),
            "tam_blue_legacy_pid_applied": dict(self._last_tam_blue_legacy_pid_applied),
            "tam_blue_legacy_pid_targets": dict(self._last_tam_blue_legacy_pid_targets),
            "tam_controller_terms": dict(self._last_tam_controller_terms),
            "tam_control_diagnostics": (
                self._tam_control_diagnostics()
                if self.tam_record_control_diagnostics else {}
            ),
        })
        return info
