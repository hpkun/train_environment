"""Per-environment diagnostic blue-policy profiles.

The paper baseline remains delegated to ``blue_coordinated_actions``.  The
additional profiles are controlled counterfactuals for diagnosis; they are not
claims about the paper's original scripted opponent.
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from rule_based_agent import (
    _blue_simple_pursuit_action_impl,
    _paper_absolute_action,
    _simple_nearest_target,
    _simple_target_by_index,
    _wrap_pi,
    blue_coordinated_actions,
)


BLUE_POLICY_PROFILES = (
    "paper_pursuit",
    "fixed_pair_pursuit_v1",
    "fixed_pair_no_mws_v1",
    "fixed_pair_hold_after_kill_v1",
    "frozen_route_blue_v1",
    "paper_minimal_fixed_pair_v1",
    "paper_minimal_straight_patrol_v1",
)


def validate_blue_policy_profile(profile: str) -> str:
    profile = str(profile)
    if profile not in BLUE_POLICY_PROFILES:
        raise ValueError(
            f"blue_policy_profile must be one of {BLUE_POLICY_PROFILES}, got {profile!r}")
    return profile


class BluePolicyController:
    """Stateful diagnostic policy owned by exactly one environment instance."""

    _HEADING_DISCONTINUITY_RAD = np.deg2rad(30.0)

    def __init__(self, profile: str = "paper_pursuit"):
        self.profile = validate_blue_policy_profile(profile)
        self.episode_generation = 0
        self.clear()

    @property
    def blue_mws_override_enabled(self) -> bool:
        return self.profile not in (
            "fixed_pair_no_mws_v1", "frozen_route_blue_v1",
            "paper_minimal_straight_patrol_v1")

    def clear(self) -> None:
        self.initial_targets: dict[str, str | None] = {}
        self.current_targets: dict[str, str | None] = {}
        self.target_switch_counts: Counter = Counter()
        self.switch_reason_counts: Counter = Counter()
        self.last_switch_reasons: dict[str, str] = {}
        self.initial_headings: dict[str, float] = {}
        self.initial_altitudes: dict[str, float] = {}
        self.hold_headings: dict[str, float] = {}
        self.route_phases: dict[str, str] = {}
        self.last_base_commands: dict[str, tuple[float, float, float]] = {}
        self.last_executed_headings: dict[str, float] = {}
        self.latest_per_blue: list[dict] = []
        self.mws_detected_agent_decisions = 0
        self.mws_override_agent_decisions = 0
        self.route_phase_changes = 0
        self.base_heading_command_discontinuities = 0
        self.executed_heading_command_discontinuities = 0
        self.altitude_recovery_frames = 0

    def reset(self, blue_ids: list[str], red_ids: list[str],
              initial_headings: dict[str, float],
              initial_altitudes: dict[str, float]) -> None:
        self.clear()
        self.episode_generation += 1
        for index, blue_id in enumerate(blue_ids):
            target = red_ids[index % len(red_ids)] if red_ids else None
            if self.profile in (
                    "paper_pursuit", "frozen_route_blue_v1",
                    "paper_minimal_straight_patrol_v1"):
                target = None
            self.initial_targets[blue_id] = target
            self.current_targets[blue_id] = target
            self.initial_headings[blue_id] = _wrap_pi(
                float(initial_headings.get(blue_id, 0.0)))
            self.initial_altitudes[blue_id] = float(
                initial_altitudes.get(blue_id, 0.0))
            self.last_switch_reasons[blue_id] = ""

    @staticmethod
    def _target_alive(obs: dict, target_id: str | None,
                      num_blue: int, num_red: int) -> bool:
        if target_id is None or not target_id.startswith("red_"):
            return False
        try:
            target_index = int(target_id.split("_", 1)[1])
        except (ValueError, IndexError):
            return False
        if not 0 <= target_index < num_red:
            return False
        mask = np.asarray(obs.get("alive_mask", obs.get("death_mask", [])))
        mask_index = num_blue + target_index
        return mask_index < mask.size and bool(mask[mask_index] > 0.5)

    @staticmethod
    def _target_index(target_id: str | None) -> int | None:
        if not target_id:
            return None
        try:
            return int(target_id.split("_", 1)[1])
        except (ValueError, IndexError):
            return None

    def _next_alive_target(self, blue_index: int, obs: dict,
                           num_blue: int, num_red: int) -> str | None:
        if num_red <= 0:
            return None
        start = blue_index % num_red
        for offset in range(1, num_red + 1):
            candidate = f"red_{(start + offset) % num_red}"
            if self._target_alive(obs, candidate, num_blue, num_red):
                return candidate
        return None

    @staticmethod
    def _paper_assignments(blue_obs: dict[str, dict], num_blue: int,
                           num_red: int) -> dict[str, str | None]:
        taken: set[int] = set()
        assignments: dict[str, str | None] = {}
        for index in range(num_blue):
            blue_id = f"blue_{index}"
            target_index, _state, _range = _simple_nearest_target(
                blue_obs.get(blue_id, {}), num_blue, num_red, excluded=taken)
            if target_index is not None:
                taken.add(target_index)
            assignments[blue_id] = (
                None if target_index is None else f"red_{target_index}")
        for index in range(num_blue):
            blue_id = f"blue_{index}"
            if assignments[blue_id] is None:
                target_index, _state, _range = _simple_nearest_target(
                    blue_obs.get(blue_id, {}), num_blue, num_red, excluded=set())
                assignments[blue_id] = (
                    None if target_index is None else f"red_{target_index}")
        return assignments

    @staticmethod
    def _altitude_recovery(obs: dict, pitch_rad: float) -> tuple[float, bool]:
        altitude = np.asarray(obs.get("altitude", [0.0])).reshape(-1)
        velocity = np.asarray(obs.get("velocity", np.zeros(3))).reshape(-1)
        alt_m = float(altitude[0]) if altitude.size else 0.0
        v_up = float(velocity[2]) if velocity.size > 2 else 0.0
        recovered = False
        if alt_m > 8500.0:
            recovery = np.deg2rad(5.0 if alt_m < 9500.0 else 10.0)
            limited = min(pitch_rad, -recovery if v_up > 5.0 else 0.0)
            recovered = limited != pitch_rad
            pitch_rad = limited
        elif alt_m < 1200.0:
            recovery = np.deg2rad(5.0 if alt_m > 500.0 else 10.0)
            limited = max(pitch_rad, recovery if v_up < -5.0 else 0.0)
            recovered = limited != pitch_rad
            pitch_rad = limited
        return float(pitch_rad), recovered

    @staticmethod
    def _route_phase(step: int) -> str:
        if step <= 75:
            return "A"
        if step <= 175:
            return "B"
        if step <= 275:
            return "C"
        return "D"

    def _frozen_action(self, blue_id: str, blue_index: int, obs: dict,
                       current_step: int) -> tuple[np.ndarray, dict]:
        phase = self._route_phase(current_step)
        offsets = {
            "A": (0.0, 0.0, 0.0),
            "B": (0.0, np.deg2rad(45.0), -np.deg2rad(45.0)),
            "C": (0.0, -np.deg2rad(45.0), np.deg2rad(45.0)),
            "D": (0.0, 0.0, 0.0),
        }
        offset = offsets[phase][blue_index] if blue_index < 3 else 0.0
        heading = _wrap_pi(self.initial_headings.get(blue_id, 0.0) + offset)
        pitch_rad, recovery = self._altitude_recovery(obs, 0.0)
        return _paper_absolute_action(pitch_rad / (np.pi / 2.0), heading), {
            "heading": heading, "pitch": pitch_rad, "phase": phase,
            "altitude_recovery": recovery, "assignment_reason": "open_loop_route",
        }

    def _record_command(self, blue_id: str, heading: float, pitch: float,
                        speed: float, phase: str, recovery: bool) -> float:
        previous = self.last_base_commands.get(blue_id)
        delta = 0.0 if previous is None else _wrap_pi(heading - previous[0])
        if previous is not None and abs(delta) > self._HEADING_DISCONTINUITY_RAD:
            self.base_heading_command_discontinuities += 1
        previous_phase = self.route_phases.get(blue_id)
        if previous_phase is not None and phase != previous_phase:
            self.route_phase_changes += 1
        self.route_phases[blue_id] = phase
        self.last_base_commands[blue_id] = (heading, pitch, speed)
        if recovery:
            self.altitude_recovery_frames += 1
        return float(delta)

    def record_executed_heading(self, blue_id: str, heading: float,
                                source: str) -> None:
        """Record the final heading target emitted by environment priority logic."""
        row = next((item for item in self.latest_per_blue
                    if item.get("blue_id") == blue_id), None)
        if row is None or not bool(row.get("alive", False)):
            return
        heading = _wrap_pi(float(heading))
        previous = self.last_executed_headings.get(blue_id)
        delta = 0.0 if previous is None else _wrap_pi(heading - previous)
        if previous is not None and abs(delta) > self._HEADING_DISCONTINUITY_RAD:
            self.executed_heading_command_discontinuities += 1
        self.last_executed_headings[blue_id] = heading
        row["executed_heading_command_rad"] = heading
        row["executed_heading_delta_rad"] = float(delta)
        row["executed_command_source"] = str(source)
        if source == "mws_override" and not row.get("mws_action_override", False):
            row["mws_action_override"] = True
            self.mws_override_agent_decisions += 1

    @staticmethod
    def _ownship_alive_from_obs(obs: dict) -> bool:
        mask = np.asarray(obs.get("alive_mask", obs.get("death_mask", []))).reshape(-1)
        if mask.size:
            return bool(mask[0] > 0.5)
        ego = np.asarray(obs.get("ego_state", []))
        return bool(ego.size and not np.allclose(ego, 0.0))

    def _dead_diagnostic_row(self, blue_id: str) -> dict:
        return {
            "blue_id": blue_id,
            "alive": False,
            "assigned_target_id": self.current_targets.get(blue_id),
            "initial_target_id": self.initial_targets.get(blue_id),
            "assignment_reason": "dead_ownship_ignored",
            "target_switch_count": int(self.target_switch_counts[blue_id]),
            "last_switch_reason": self.last_switch_reasons.get(blue_id, ""),
            "base_heading_command_rad": None,
            "pursuit_heading_cmd_rad": None,
            "pursuit_pitch_cmd_rad": None,
            "target_speed_cmd_mps": None,
            "mws_detected": False,
            "mws_action_override": False,
            "selected_missile_id": None,
            "route_phase": self.route_phases.get(blue_id, ""),
            "altitude_recovery_active": False,
            "executed_heading_command_rad": None,
            "executed_heading_delta_rad": None,
            "executed_command_source": "not_executed_dead",
            "command_heading_delta_rad": None,
        }

    def act(self, blue_obs: dict[str, dict], num_blue: int, num_red: int,
            engaged_targets: set[str] | None,
            own_positions: dict[str, np.ndarray] | None,
            own_headings: dict[str, float] | None,
            current_step: int,
            selected_missiles: dict[str, str | None] | None = None,
            mws_detected: dict[str, bool] | None = None,
            own_alive: dict[str, bool] | None = None) -> dict[str, np.ndarray]:
        own_positions = own_positions or {}
        own_headings = own_headings or {}
        selected_missiles = selected_missiles or {}
        mws_detected = mws_detected or {}
        if own_alive is None:
            own_alive = {
                f"blue_{index}": self._ownship_alive_from_obs(
                    blue_obs.get(f"blue_{index}", {}))
                for index in range(num_blue)
            }
        else:
            own_alive = {key: bool(value) for key, value in own_alive.items()}

        if self.profile == "paper_pursuit":
            assignments = self._paper_assignments(blue_obs, num_blue, num_red)
            actions = blue_coordinated_actions(
                blue_obs, num_blue, num_red, engaged_targets=engaged_targets,
                own_positions=own_positions, own_headings=own_headings,
                pursuit_mode="paper_pursuit")
        else:
            assignments = {}
            actions = {}
            assignment_reasons: dict[str, str] = {}
            for blue_index in range(num_blue):
                blue_id = f"blue_{blue_index}"
                if not own_alive.get(blue_id, False):
                    actions[blue_id] = np.zeros(3, dtype=np.float32)
                    assignments[blue_id] = self.current_targets.get(blue_id)
                    assignment_reasons[blue_id] = "dead_ownship_ignored"
                    continue
                obs = blue_obs.get(blue_id, {})
                if self.profile == "paper_minimal_straight_patrol_v1":
                    heading = self.initial_headings.get(blue_id, 0.0)
                    actions[blue_id] = _paper_absolute_action(
                        0.0, heading, speed_mps=300.0)
                    assignments[blue_id] = None
                    assignment_reasons[blue_id] = "straight_patrol"
                    continue
                if self.profile == "frozen_route_blue_v1":
                    action, _meta = self._frozen_action(
                        blue_id, blue_index, obs, current_step)
                    actions[blue_id] = action
                    assignments[blue_id] = None
                    assignment_reasons[blue_id] = "open_loop_route"
                    continue

                target_id = self.current_targets.get(blue_id)
                if self.profile == "paper_minimal_fixed_pair_v1":
                    assignments[blue_id] = target_id
                    target_index = self._target_index(target_id)
                    if not self._target_alive(
                            obs, target_id, num_blue, num_red):
                        heading = _wrap_pi(float(own_headings.get(blue_id, 0.0)))
                        actions[blue_id] = _paper_absolute_action(
                            0.0, heading, speed_mps=300.0)
                        assignment_reasons[blue_id] = "target_dead_hold"
                    else:
                        _idx, target_state, _range = _simple_target_by_index(
                            obs, num_blue, num_red, int(target_index))
                        action = _blue_simple_pursuit_action_impl(
                            obs, num_blue, num_red, blue_index,
                            forced_target_idx=target_index,
                            own_position=own_positions.get(blue_id),
                            own_heading=own_headings.get(blue_id),
                            paper_profile=True)
                        action[0] = np.clip(
                            action[0], -10.0 / 90.0, 10.0 / 90.0)
                        action[2] = _paper_absolute_action(
                            0.0, 0.0, speed_mps=300.0)[2]
                        actions[blue_id] = action
                        assignment_reasons[blue_id] = (
                            "fixed_pair" if target_state is not None
                            else "track_unavailable_hold")
                    continue
                if not self._target_alive(obs, target_id, num_blue, num_red):
                    if self.profile == "fixed_pair_hold_after_kill_v1":
                        self.current_targets[blue_id] = None
                        self.hold_headings.setdefault(
                            blue_id, _wrap_pi(float(own_headings.get(blue_id, 0.0))))
                    else:
                        replacement = self._next_alive_target(
                            blue_index, obs, num_blue, num_red)
                        if replacement != target_id:
                            self.current_targets[blue_id] = replacement
                            if target_id is not None:
                                self.target_switch_counts[blue_id] += 1
                                self.switch_reason_counts["target_dead"] += 1
                                self.last_switch_reasons[blue_id] = "target_dead"
                    target_id = self.current_targets.get(blue_id)
                assignments[blue_id] = target_id
                target_index = self._target_index(target_id)
                if target_index is None:
                    heading = self.hold_headings.setdefault(
                        blue_id, _wrap_pi(float(own_headings.get(blue_id, 0.0))))
                    actions[blue_id] = _paper_absolute_action(0.0, heading)
                    assignment_reasons[blue_id] = "hold_after_kill"
                else:
                    _idx, target_state, _range = _simple_target_by_index(
                        obs, num_blue, num_red, target_index)
                    actions[blue_id] = _blue_simple_pursuit_action_impl(
                        obs, num_blue, num_red, blue_index,
                        forced_target_idx=target_index,
                        own_position=own_positions.get(blue_id),
                        own_heading=own_headings.get(blue_id), paper_profile=True)
                    assignment_reasons[blue_id] = (
                        "fixed_pair" if target_state is not None
                        else "track_unavailable_hold")

        if self.profile == "paper_pursuit":
            assignment_reasons = {
                blue_id: "per_step_nearest" for blue_id in assignments}

        self.latest_per_blue = []
        for blue_index in range(num_blue):
            blue_id = f"blue_{blue_index}"
            if not own_alive.get(blue_id, False):
                self.latest_per_blue.append(self._dead_diagnostic_row(blue_id))
                continue
            obs = blue_obs.get(blue_id, {})
            action = np.asarray(actions.get(blue_id, np.zeros(3)), dtype=np.float32)
            detected = False
            if self.profile not in (
                    "frozen_route_blue_v1", "paper_minimal_straight_patrol_v1"):
                warning = np.asarray(obs.get("missile_warning", [0.0])).reshape(-1)
                detected = bool(mws_detected.get(
                    blue_id, warning[0] > 0.5 if warning.size else False))
            override = False
            self.mws_detected_agent_decisions += int(detected)
            heading = _wrap_pi(float(action[1]) * np.pi)
            pitch = float(action[0]) * (np.pi / 2.0)
            phase = (self._route_phase(current_step)
                     if self.profile == "frozen_route_blue_v1"
                     else "straight" if self.profile == "paper_minimal_straight_patrol_v1"
                     else "pursuit")
            recovery = False
            if self.profile == "frozen_route_blue_v1":
                _unused, meta = self._frozen_action(blue_id, blue_index, obs, current_step)
                heading, pitch = meta["heading"], meta["pitch"]
                recovery = bool(meta["altitude_recovery"])
            commanded_speed = (300.0 if self.profile in (
                "paper_minimal_fixed_pair_v1",
                "paper_minimal_straight_patrol_v1") else 250.0)
            delta = self._record_command(
                blue_id, heading, pitch, commanded_speed, phase, recovery)
            assigned = assignments.get(blue_id)
            previous = self.current_targets.get(blue_id)
            assignment_reason = assignment_reasons.get(blue_id, "fixed_pair")
            if self.profile == "paper_pursuit":
                if previous is not None and assigned != previous:
                    reason = ("target_dead" if not self._target_alive(
                        obs, previous, num_blue, num_red) else "distance")
                    self.target_switch_counts[blue_id] += 1
                    self.switch_reason_counts[reason] += 1
                    self.last_switch_reasons[blue_id] = reason
                self.current_targets[blue_id] = assigned
            self.latest_per_blue.append({
                "blue_id": blue_id,
                "alive": True,
                "assigned_target_id": assigned,
                "initial_target_id": self.initial_targets.get(blue_id),
                "assignment_reason": assignment_reason,
                "target_switch_count": int(self.target_switch_counts[blue_id]),
                "last_switch_reason": self.last_switch_reasons.get(blue_id, ""),
                "base_heading_command_rad": float(heading),
                "pursuit_heading_cmd_rad": float(heading),
                "pursuit_pitch_cmd_rad": float(pitch),
                "target_speed_cmd_mps": commanded_speed,
                "mws_detected": detected,
                "mws_action_override": override,
                "selected_missile_id": (
                    None if self.profile == "frozen_route_blue_v1"
                    else selected_missiles.get(blue_id)),
                "route_phase": phase,
                "altitude_recovery_active": recovery,
                "executed_heading_command_rad": None,
                "executed_heading_delta_rad": None,
                "executed_command_source": "pending_parse",
                "command_heading_delta_rad": delta,
            })
        return actions

    def snapshot_episode_diagnostics(self) -> dict:
        return copy.deepcopy({
            "blue_policy_profile": self.profile,
            "episode_generation": int(self.episode_generation),
            "per_blue": self.latest_per_blue,
            "blue_target_switches_total": int(sum(self.target_switch_counts.values())),
            "blue_target_dead_switches": int(self.switch_reason_counts["target_dead"]),
            "blue_distance_triggered_switches": int(self.switch_reason_counts["distance"]),
            "blue_engaged_triggered_switches": int(self.switch_reason_counts["engaged"]),
            "blue_mws_detected_agent_decisions": int(
                self.mws_detected_agent_decisions),
            "blue_mws_override_agent_decisions": int(
                self.mws_override_agent_decisions),
            # Compatibility aliases: units are agent decisions, not physics frames.
            "blue_mws_detected_frames": int(self.mws_detected_agent_decisions),
            "blue_mws_override_frames": int(self.mws_override_agent_decisions),
            "blue_route_phase_changes": int(self.route_phase_changes),
            "blue_base_heading_command_discontinuities": int(
                self.base_heading_command_discontinuities),
            "blue_executed_heading_command_discontinuities": int(
                self.executed_heading_command_discontinuities),
            "blue_heading_command_discontinuities": int(
                self.base_heading_command_discontinuities),
            "blue_altitude_recovery_frames": int(self.altitude_recovery_frames),
        })
