"""Single formal Blue policy for the paper-derived 3V3 environment."""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from rule_based_agent import _paper_absolute_action, _wrap_pi


PAPER_BLUE_POLICY_PROFILE = "simple_dynamic_pursuit_with_mws"
BLUE_POLICY_PROFILES = (PAPER_BLUE_POLICY_PROFILE,)


def validate_blue_policy_profile(profile: str) -> str:
    profile = str(profile)
    if profile != PAPER_BLUE_POLICY_PROFILE:
        raise ValueError(
            f"blue_policy_profile must be {PAPER_BLUE_POLICY_PROFILE!r}")
    return profile


class BluePolicyController:
    """Per-environment dynamic nearest-target pursuit controller."""

    _HEADING_DISCONTINUITY_RAD = np.deg2rad(30.0)

    def __init__(self, profile: str = PAPER_BLUE_POLICY_PROFILE):
        self.profile = validate_blue_policy_profile(profile)
        self.episode_generation = 0
        self.clear()

    @property
    def blue_mws_override_enabled(self) -> bool:
        return True

    def clear(self) -> None:
        self.current_targets: dict[str, str | None] = {}
        self.target_switch_counts: Counter = Counter()
        self.last_switch_reasons: dict[str, str] = {}
        self.last_base_commands: dict[str, tuple[float, float, float]] = {}
        self.last_executed_headings: dict[str, float] = {}
        self.latest_per_blue: list[dict] = []
        self.mws_detected_agent_decisions = 0
        self.mws_override_agent_decisions = 0
        self.base_heading_command_discontinuities = 0
        self.executed_heading_command_discontinuities = 0

    def reset(self, blue_ids: list[str], red_ids: list[str],
              initial_headings: dict[str, float],
              initial_altitudes: dict[str, float]) -> None:
        del red_ids, initial_headings, initial_altitudes
        self.clear()
        self.episode_generation += 1
        self.current_targets = {blue_id: None for blue_id in blue_ids}

    @classmethod
    def _assign_targets(cls, num_blue: int, num_red: int,
                        own_positions: dict[str, np.ndarray],
                        own_alive: dict[str, bool],
                        enemy_positions: dict[str, np.ndarray],
                        enemy_alive: dict[str, bool]):
        assignments: dict[str, int | None] = {}
        taken: set[int] = set()
        for blue_index in range(num_blue):
            blue_id = f"blue_{blue_index}"
            if not own_alive.get(blue_id, False):
                assignments[blue_id] = None
                continue
            own_position = np.asarray(
                own_positions.get(blue_id, np.full(3, np.nan)),
                dtype=np.float64)
            candidates = []
            for target_index in range(num_red):
                target_id = f"red_{target_index}"
                target_position = np.asarray(
                    enemy_positions.get(target_id, np.full(3, np.nan)),
                    dtype=np.float64)
                distance = float(np.linalg.norm(target_position - own_position))
                if (enemy_alive.get(target_id, False)
                        and np.all(np.isfinite(own_position))
                        and np.all(np.isfinite(target_position))
                        and distance > 0.0):
                    candidates.append((target_index, distance))
            candidates.sort(key=lambda item: (item[1], item[0]))
            unassigned = [item for item in candidates if item[0] not in taken]
            selected = (unassigned or candidates)
            target_index = selected[0][0] if selected else None
            assignments[blue_id] = target_index
            if target_index is not None:
                taken.add(target_index)
        return assignments

    def _record_base_heading(self, blue_id: str, heading: float) -> float:
        previous = self.last_base_commands.get(blue_id)
        delta = 0.0 if previous is None else _wrap_pi(heading - previous[0])
        if previous is not None and abs(delta) > self._HEADING_DISCONTINUITY_RAD:
            self.base_heading_command_discontinuities += 1
        self.last_base_commands[blue_id] = (heading, 0.0, 300.0)
        return float(delta)

    def record_executed_heading(self, blue_id: str, heading: float,
                                source: str) -> None:
        row = next((item for item in self.latest_per_blue
                    if item.get("blue_id") == blue_id), None)
        if row is None or not row.get("alive", False):
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
        if source == "mws_override" and not row["mws_action_override"]:
            row["mws_action_override"] = True
            self.mws_override_agent_decisions += 1

    def act(self, blue_obs: dict[str, dict], num_blue: int, num_red: int,
            engaged_targets: set[str] | None,
            own_positions: dict[str, np.ndarray] | None,
            own_headings: dict[str, float] | None,
            current_step: int,
            selected_missiles: dict[str, str | None] | None = None,
            mws_detected: dict[str, bool] | None = None,
            own_alive: dict[str, bool] | None = None,
            enemy_positions: dict[str, np.ndarray] | None = None,
            enemy_alive: dict[str, bool] | None = None,
            assigned_targets: dict[str, str | None] | None = None,
            ) -> dict[str, np.ndarray]:
        del engaged_targets, current_step, assigned_targets
        own_positions = own_positions or {}
        own_headings = own_headings or {}
        selected_missiles = selected_missiles or {}
        mws_detected = mws_detected or {}
        own_alive = own_alive or {
            f"blue_{index}": True for index in range(num_blue)}
        enemy_positions = enemy_positions or {}
        enemy_alive = enemy_alive or {
            f"red_{index}": True for index in range(num_red)}
        assignments = self._assign_targets(
            num_blue, num_red, own_positions, own_alive,
            enemy_positions, enemy_alive)

        actions: dict[str, np.ndarray] = {}
        self.latest_per_blue = []
        for blue_index in range(num_blue):
            blue_id = f"blue_{blue_index}"
            alive = bool(own_alive.get(blue_id, False))
            target_index = assignments[blue_id]
            target_id = None if target_index is None else f"red_{target_index}"
            previous_target = self.current_targets.get(blue_id)
            if alive and previous_target is not None and target_id != previous_target:
                self.target_switch_counts[blue_id] += 1
                self.last_switch_reasons[blue_id] = "dynamic_nearest"
            self.current_targets[blue_id] = target_id

            if not alive:
                actions[blue_id] = np.zeros(3, dtype=np.float32)
                self.latest_per_blue.append({
                    "blue_id": blue_id, "alive": False,
                    "assigned_target_id": target_id,
                    "assignment_reason": "dead_ownship_ignored",
                    "executed_command_source": "not_executed_dead"})
                continue

            if target_index is None:
                heading = _wrap_pi(float(own_headings.get(blue_id, 0.0)))
                action = _paper_absolute_action(0.0, heading, speed_mps=300.0)
                reason = "no_valid_target_hold"
            else:
                relative = (np.asarray(enemy_positions[target_id], dtype=np.float64)
                            - np.asarray(own_positions[blue_id], dtype=np.float64))
                heading = _wrap_pi(float(np.arctan2(relative[1], relative[0])))
                pitch = float(np.arctan2(
                    relative[2], max(np.hypot(relative[0], relative[1]), 1e-9)))
                action = _paper_absolute_action(
                    pitch, heading, speed_mps=300.0)
                reason = "dynamic_nearest_unique_first"

            actions[blue_id] = np.asarray(action, dtype=np.float32)
            detected = bool(mws_detected.get(blue_id, False))
            self.mws_detected_agent_decisions += int(detected)
            delta = self._record_base_heading(blue_id, heading)
            self.latest_per_blue.append({
                "blue_id": blue_id,
                "alive": True,
                "assigned_target_id": target_id,
                "assignment_reason": reason,
                "target_switch_count": int(self.target_switch_counts[blue_id]),
                "last_switch_reason": self.last_switch_reasons.get(blue_id, ""),
                "base_heading_command_rad": float(heading),
                "pursuit_heading_cmd_rad": float(heading),
                "pursuit_pitch_cmd_rad": float(action[0]) * np.pi / 2.0,
                "target_speed_cmd_mps": 300.0,
                "mws_detected": detected,
                "mws_action_override": False,
                "selected_missile_id": selected_missiles.get(blue_id),
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
            "blue_mws_detected_agent_decisions": int(
                self.mws_detected_agent_decisions),
            "blue_mws_override_agent_decisions": int(
                self.mws_override_agent_decisions),
            "blue_base_heading_command_discontinuities": int(
                self.base_heading_command_discontinuities),
            "blue_executed_heading_command_discontinuities": int(
                self.executed_heading_command_discontinuities),
        })
