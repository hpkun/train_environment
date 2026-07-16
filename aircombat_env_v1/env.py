"""Minimal single-agent red-vs-rule-blue JSBSim Gymnasium environment."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .aircraft import AircraftSimulator
from .combat import (action_to_targets, hit_event, in_attack_zone,
                     pursuit_action, relative_geometry, update_attack_dwell)
from .config import load_config
from .observation import build_observation
from .pid import PaperAutopilot
from .reward import potential, step_reward, terminal_reward
from .scenario import SCENARIO_MODES, make_scenario


class AirCombat1v1Env(gym.Env):
    """One externally controlled red F-16 against a rule-controlled blue F-16."""

    metadata = {"render_modes": []}

    def __init__(self, opponent_policy="straight",
                 scenario_mode="fixed_tail_chase",
                 opponent_mix_probability=0.2, randomize=None,
                 max_steps=1000, config_path=None):
        if opponent_policy not in ("straight", "pursuit", "mixed"):
            raise ValueError("opponent_policy must be straight, pursuit, or mixed")
        if randomize is not None:
            scenario_mode = (
                "randomized_tail_chase" if randomize
                else "fixed_tail_chase")
        if scenario_mode not in SCENARIO_MODES:
            raise ValueError(f"unsupported scenario mode: {scenario_mode}")
        self.opponent_policy = opponent_policy
        self.scenario_mode = scenario_mode
        self.opponent_mix_probability = float(opponent_mix_probability)
        self.max_steps = int(max_steps)
        self.config = load_config(config_path) if config_path else load_config()
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(20,), dtype=np.float32)
        frequency = self.config["timing"]["sim_frequency_hz"]
        self.red_sim = AircraftSimulator(frequency)
        self.blue_sim = AircraftSimulator(frequency)
        self.red_controller = PaperAutopilot.from_config(self.config)
        self.blue_controller = PaperAutopilot.from_config(self.config)
        self.render_mode = None
        self._red_state = None
        self._blue_state = None
        self._blue_heading = 0.0
        self._step_count = 0
        self._red_dwell = 0.0
        self._blue_dwell = 0.0
        self._potential = 0.0
        self._event = None
        self._actual_opponent_policy = opponent_policy
        self._previous_action = np.zeros(3, dtype=np.float32)
        self._numerical_invalid = False
        self._physics_exception = False

    def set_curriculum_stage(self, stage):
        if stage == 1:
            self.scenario_mode = "fixed_tail_chase"
            self.opponent_policy = "straight"
        elif stage == 2:
            self.scenario_mode = "randomized_tail_chase"
            self.opponent_policy = "straight"
        elif stage == 3:
            self.scenario_mode = "randomized_tail_chase"
            self.opponent_policy = "mixed"
        else:
            raise ValueError("curriculum stage must be 1, 2, or 3")

    @staticmethod
    def action_to_targets(action, current_heading):
        return action_to_targets(action, current_heading)

    def _reset_aircraft(self, simulator, initial):
        trim = self.config["trim"]
        return simulator.reset(
            altitude_m=initial.altitude_m, speed_mps=initial.speed_mps,
            heading_deg=initial.heading_deg, roll_deg=initial.roll_deg,
            pitch_deg=initial.pitch_deg, latitude_deg=initial.latitude_deg,
            longitude_deg=initial.longitude_deg,
            elevator_trim=trim["elevator_trim"],
            throttle_base=trim["throttle_base"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mode = options.get("scenario_mode", self.scenario_mode)
        red_initial, blue_initial = make_scenario(mode, self.np_random)
        if self.opponent_policy == "mixed":
            self._actual_opponent_policy = (
                "pursuit" if self.np_random.random()
                < self.opponent_mix_probability else "straight")
        else:
            self._actual_opponent_policy = self.opponent_policy
        self._red_state = self._reset_aircraft(self.red_sim, red_initial)
        self._blue_state = self._reset_aircraft(self.blue_sim, blue_initial)
        self.red_controller.reset()
        self.blue_controller.reset()
        self._blue_heading = float(self._blue_state["heading"])
        self._step_count = 0
        self._red_dwell = self._blue_dwell = 0.0
        self._event = None
        self._previous_action = np.zeros(3, dtype=np.float32)
        self._numerical_invalid = False
        self._physics_exception = False
        self._potential = potential(
            relative_geometry(self._red_state, self._blue_state))
        observation = build_observation(
            self._red_state, self._blue_state, 0.0, 0.0)
        return observation, self._info()

    def _blue_action(self):
        if self._actual_opponent_policy == "pursuit":
            return pursuit_action(self._blue_state, self._red_state)
        pitch = 0.0
        heading_offset = (
            self._blue_heading - self._blue_state["heading"]) / np.deg2rad(60.0)
        speed = (240.0 - 250.0) / 50.0
        return np.clip(
            np.array([pitch, heading_offset, speed], dtype=np.float32),
            -1.0, 1.0)

    @staticmethod
    def _finite_state(state):
        return bool(np.all(np.isfinite(tuple(state.values()))))

    @staticmethod
    def _crashed_state(state):
        return state["altitude"] < 1000.0 or state["true_airspeed"] < 120.0

    def _advance_pair(self, red_targets, blue_targets):
        for _ in range(self.config["timing"]["physics_steps_per_decision"]):
            try:
                red_controls = self.red_controller.step(
                    self._red_state["roll"], self._red_state["pitch"],
                    self._red_state["heading"], self._red_state["true_airspeed"],
                    *red_targets, self.red_sim.dt)
                blue_controls = self.blue_controller.step(
                    self._blue_state["roll"], self._blue_state["pitch"],
                    self._blue_state["heading"], self._blue_state["true_airspeed"],
                    *blue_targets, self.blue_sim.dt)
                self.red_sim.set_controls(*red_controls)
                self.blue_sim.set_controls(*blue_controls)
                self._red_state = self.red_sim.run()
            except (RuntimeError, ValueError, FloatingPointError):
                return "red_numerical_invalid", True
            try:
                self._blue_state = self.blue_sim.run()
            except (RuntimeError, ValueError, FloatingPointError):
                return "blue_numerical_invalid", True
            red_finite = self._finite_state(self._red_state)
            blue_finite = self._finite_state(self._blue_state)
            if not red_finite and not blue_finite:
                return "draw_both_numerical_invalid", False
            if not red_finite:
                return "red_numerical_invalid", False
            if not blue_finite:
                return "blue_numerical_invalid", False
            red_crash = self._crashed_state(self._red_state)
            blue_crash = self._crashed_state(self._blue_state)
            if red_crash and blue_crash:
                return "draw_both_crash", False
            if red_crash:
                return "red_crash", False
            if blue_crash:
                return "blue_crash", False
        return None, False

    def step(self, action):
        if self._red_state is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float32)
        previous_red_state = copy.copy(self._red_state)
        previous_blue_state = copy.copy(self._blue_state)
        previous_observation = build_observation(
            previous_red_state, previous_blue_state,
            self._red_dwell, self._blue_dwell)
        previous_potential = self._potential
        red_targets = action_to_targets(action, self._red_state["heading"])
        blue_targets = action_to_targets(
            self._blue_action(), self._blue_state["heading"])
        previous_red_dwell, previous_blue_dwell = self._red_dwell, self._blue_dwell
        event, physics_exception = self._advance_pair(red_targets, blue_targets)
        self._step_count += 1
        numerical_invalid = event is not None and "numerical_invalid" in event
        self._numerical_invalid = numerical_invalid
        self._physics_exception = physics_exception
        if numerical_invalid:
            self._red_state = previous_red_state
            self._blue_state = previous_blue_state
            self._event = event
            reward = terminal_reward(event)
            if not np.isfinite(reward):
                reward = 0.0
            return (previous_observation.copy(), float(reward), True, False,
                    self._info())
        if event is None:
            red_geometry = relative_geometry(self._red_state, self._blue_state)
            blue_geometry = relative_geometry(self._blue_state, self._red_state)
            self._red_dwell = update_attack_dwell(
                self._red_dwell, in_attack_zone(red_geometry))
            self._blue_dwell = update_attack_dwell(
                self._blue_dwell, in_attack_zone(blue_geometry))
            event = hit_event(self._red_dwell, self._blue_dwell)
        next_potential = potential(
            relative_geometry(self._red_state, self._blue_state))
        reward = step_reward(
            previous_potential, next_potential,
            self._red_dwell - previous_red_dwell,
            self._blue_dwell - previous_blue_dwell, event,
            action, self._previous_action)
        self._previous_action = action.copy()
        self._potential = next_potential
        terminated = event is not None
        truncated = not terminated and self._step_count >= self.max_steps
        if truncated:
            event = "timeout"
        self._event = event
        observation = build_observation(
            self._red_state, self._blue_state,
            self._red_dwell, self._blue_dwell)
        return observation, reward, terminated, truncated, self._info()

    def _info(self):
        geometry = relative_geometry(self._red_state, self._blue_state)
        reverse = relative_geometry(self._blue_state, self._red_state)
        info = {
            "step": self._step_count, "event": self._event,
            "distance_m": geometry["distance_m"],
            "closure_mps": geometry["closure_mps"],
            "red_boresight_deg": float(np.rad2deg(geometry["boresight_angle"])),
            "blue_boresight_deg": float(np.rad2deg(reverse["boresight_angle"])),
            "red_attack_dwell": self._red_dwell,
            "blue_attack_dwell": self._blue_dwell,
            "red_altitude_m": self._red_state["altitude"],
            "blue_altitude_m": self._blue_state["altitude"],
            "red_speed_mps": self._red_state["true_airspeed"],
            "blue_speed_mps": self._blue_state["true_airspeed"],
            "opponent_policy": self._actual_opponent_policy,
            "numerical_invalid": self._numerical_invalid,
            "physics_exception": self._physics_exception,
        }
        if self._event is not None:
            winners = {
                "red_hit": "red", "blue_crash": "red",
                "blue_hit": "blue", "red_crash": "blue",
                "draw_simultaneous_hit": "draw", "draw_both_crash": "draw",
                "red_numerical_invalid": "blue",
                "blue_numerical_invalid": "red",
                "draw_both_numerical_invalid": "draw",
                "timeout": None,
            }
            info.update(
                winner=winners[self._event],
                termination_reason=self._event)
        return info

    @property
    def red_state(self):
        return copy.copy(self._red_state)

    @property
    def blue_state(self):
        return copy.copy(self._blue_state)

    def close(self):
        self.red_sim = None
        self.blue_sim = None
