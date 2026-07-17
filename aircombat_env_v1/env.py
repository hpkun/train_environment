"""Single-agent red versus rule-blue JSBSim missile combat environment."""

from __future__ import annotations
import copy
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .aircraft import AircraftSimulator
from .combat import action_to_targets, local_neu, pursuit_action, relative_geometry
from .config import load_config
from .missile import Missile, PAPER_MISSILE_PARAMETERS, PROJECT_MISSILE_ASSUMPTIONS
from .observation import build_observation
from .opponent import paper_greedy_action
from .pid import PaperAutopilot
from .reward import step_reward, terminal_reward
from .scenario import SCENARIO_MODES, curriculum_scenario, make_scenario


def event_semantics(event):
    winners = {"red_hit": "red", "blue_hit": "blue", "red_crash": "blue",
               "blue_crash": "red", "draw_simultaneous_hit": "draw",
               "draw_both_crash": "draw", "timeout": None,
               "red_numerical_invalid": None, "blue_numerical_invalid": None,
               "draw_both_numerical_invalid": None, "physics_exception": None}
    return {"winner": winners.get(event), "termination_reason": event,
            "opponent_failure": event == "blue_crash",
            "invalid_episode": "numerical_invalid" in str(event) or event == "physics_exception",
            "valid_combat_outcome": event in {"red_hit", "blue_hit", "red_crash",
                "blue_crash", "draw_simultaneous_hit", "draw_both_crash", "timeout"}}


class AirCombat1v1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_policy="paper_greedy", scenario_mode="paper_nominal_1v1",
                 opponent_mix_probability=0.2, randomize=None, max_steps=1000,
                 config_path=None):
        if opponent_policy not in ("paper_greedy", "straight", "pursuit", "mixed"):
            raise ValueError("unsupported opponent policy")
        if randomize is not None:
            scenario_mode = "randomized_tail_chase" if randomize else "fixed_tail_chase"
        if scenario_mode not in SCENARIO_MODES:
            raise ValueError(f"unsupported scenario mode: {scenario_mode}")
        self.opponent_policy, self.scenario_mode = opponent_policy, scenario_mode
        self.opponent_mix_probability, self.max_steps = float(opponent_mix_probability), int(max_steps)
        self.config = load_config(config_path) if config_path else load_config()
        self.action_space = spaces.Dict({
            "maneuver": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "fire": spaces.Discrete(2)})
        self.observation_space = spaces.Box(-1.0, 1.0, (26,), np.float32)
        frequency = self.config["timing"]["sim_frequency_hz"]
        self.red_sim, self.blue_sim = AircraftSimulator(frequency), AircraftSimulator(frequency)
        self.red_controller = PaperAutopilot.from_config(self.config)
        self.blue_controller = PaperAutopilot.from_config(self.config)
        self.render_mode = None
        self._red_state = self._blue_state = None

    def set_curriculum_stage(self, stage):
        del stage
        self.scenario_mode = "paper_nominal_1v1"

    @staticmethod
    def action_to_targets(action, current_heading):
        return action_to_targets(action, current_heading)

    def _reset_aircraft(self, simulator, initial):
        trim = self.config["trim"]
        return simulator.reset(altitude_m=initial.altitude_m, speed_mps=initial.speed_mps,
            heading_deg=initial.heading_deg, roll_deg=initial.roll_deg,
            pitch_deg=initial.pitch_deg, latitude_deg=initial.latitude_deg,
            longitude_deg=initial.longitude_deg, elevator_trim=trim["elevator_trim"],
            throttle_base=trim["throttle_base"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); options = options or {}
        mode = options.get("scenario_mode", self.scenario_mode)
        if mode == "curriculum_mixed_tail_chase": mode = curriculum_scenario(self.np_random)
        self._actual_scenario_mode = mode
        red_initial, blue_initial = make_scenario(mode, self.np_random)
        self._actual_opponent_policy = self.opponent_policy
        if self.opponent_policy == "mixed":
            self._actual_opponent_policy = "pursuit" if self.np_random.random() < self.opponent_mix_probability else "straight"
        self._red_state = self._reset_aircraft(self.red_sim, red_initial)
        self._blue_state = self._reset_aircraft(self.blue_sim, blue_initial)
        self.red_controller.reset(); self.blue_controller.reset()
        self._blue_heading = float(self._blue_state["heading"])
        self._step_count = 0; self._event = None; self._sim_time = 0.0
        self._missiles = []; self._inventory = {"red": 2, "blue": 2}
        self._last_launch = {"red": -25.0, "blue": -25.0}
        self._launch_count = {"red": 0, "blue": 0}
        self._invalid_fire_commands = 0; self._numerical_invalid = False
        self._physics_exception = False; self._reward_components = {}
        self._flight_envelope_violation = False; self._last_hit_time = None
        return self._observation(), self._info()

    def _blue_action(self):
        if self._actual_opponent_policy == "paper_greedy":
            return paper_greedy_action(self._blue_state, self._red_state)
        if self._actual_opponent_policy == "pursuit":
            return pursuit_action(self._blue_state, self._red_state)
        return np.array([0.0, (self._blue_heading-self._blue_state["heading"])/np.deg2rad(60), -0.2], np.float32)

    @staticmethod
    def _finite_state(state): return bool(np.all(np.isfinite(tuple(state.values()))))
    @staticmethod
    def _crashed_state(state):
        # Engineering safety termination, deliberately above paper minimum 750 m.
        return state["altitude"] < 1000.0 or state["true_airspeed"] < 120.0
    @staticmethod
    def _velocity_neu(state):
        return np.array([state["v_north"], state["v_east"], -state["v_down"]], float)

    def _launch_allowed(self, side):
        own, target = ((self._red_state, self._blue_state) if side == "red"
                       else (self._blue_state, self._red_state))
        geometry = relative_geometry(own, target)
        distance = geometry["distance_m"]
        already_chasing = any(m.alive and m.shooter == side for m in self._missiles)
        return (self._inventory[side] > 0 and self._sim_time-self._last_launch[side] >= 25.0
                and distance <= 14000.0
                and geometry["boresight_angle"] <= np.deg2rad(
                    PROJECT_MISSILE_ASSUMPTIONS["lock_angle_deg"])
                and not already_chasing)

    def _launch(self, side):
        if not self._launch_allowed(side): return False
        state = self._red_state if side == "red" else self._blue_state
        target = "blue" if side == "red" else "red"
        velocity = self._velocity_neu(state)
        self._missiles.append(Missile(local_neu(state), velocity, side, target, self._sim_time))
        self._inventory[side] -= 1; self._last_launch[side] = self._sim_time
        self._launch_count[side] += 1
        return True

    def _advance_pair(self, red_targets, blue_targets):
        for _ in range(self.config["timing"]["physics_steps_per_decision"]):
            try:
                rc = self.red_controller.step(self._red_state["roll"], self._red_state["pitch"],
                    self._red_state["heading"], self._red_state["true_airspeed"], *red_targets, self.red_sim.dt)
                bc = self.blue_controller.step(self._blue_state["roll"], self._blue_state["pitch"],
                    self._blue_state["heading"], self._blue_state["true_airspeed"], *blue_targets, self.blue_sim.dt)
                self.red_sim.set_controls(*rc); self.blue_sim.set_controls(*bc)
                self._red_state = self.red_sim.run(); self._blue_state = self.blue_sim.run()
            except (RuntimeError, ValueError, FloatingPointError): return "physics_exception", True
            self._sim_time += self.red_sim.dt
            if not self._finite_state(self._red_state): return "red_numerical_invalid", False
            if not self._finite_state(self._blue_state): return "blue_numerical_invalid", False
            red_crash, blue_crash = self._crashed_state(self._red_state), self._crashed_state(self._blue_state)
            if red_crash or blue_crash:
                return ("draw_both_crash" if red_crash and blue_crash else
                        "red_crash" if red_crash else "blue_crash"), False
            hits = []
            for missile in self._missiles:
                target_state = self._blue_state if missile.target == "blue" else self._red_state
                reason = missile.step(local_neu(target_state), self._velocity_neu(target_state), True, self.red_sim.dt)
                if reason == "hit": hits.append(missile.shooter)
            if hits:
                self._last_hit_time = self._sim_time
                return ("draw_simultaneous_hit" if len(set(hits)) > 1 else
                        "red_hit" if hits[0] == "red" else "blue_hit"), False
        return None, False

    def _incoming(self, side="red"):
        items = [m for m in self._missiles if m.alive and m.target == side]
        return min(items, key=lambda m:m.distance) if items else None

    def _weapon_observation(self):
        incoming = self._incoming("red")
        cooldown = max(0.0, 25.0-(self._sim_time-self._last_launch["red"]))/25.0
        if incoming:
            rel = local_neu(self._red_state)-incoming.position_neu
            rv = self._velocity_neu(self._red_state)-incoming.velocity_neu
            distance = max(float(np.linalg.norm(rel)), 1e-6)
            closing = -float(np.dot(rel, rv))/distance
            los_angle = np.arccos(np.clip(np.dot(incoming.velocity_neu, rel)/
                max(np.linalg.norm(incoming.velocity_neu)*distance,1e-8),-1,1))
            speed = max(float(np.linalg.norm(incoming.velocity_neu)),1.0)
        else: distance, closing, los_angle, speed = 14000.0, 0.0, np.pi, 1.0
        return {"own_inventory_ratio":self._inventory["red"]/2,
                "cooldown_ratio":cooldown, "enemy_inventory_ratio":self._inventory["blue"]/2,
                "incoming":float(incoming is not None), "incoming_range_ratio":distance/14000,
                "incoming_closing_ratio":closing/1000, "incoming_los_angle_ratio":los_angle/np.pi,
                "incoming_ttg_ratio":min(distance/speed/56.0,1.0)}

    def _observation(self): return build_observation(self._red_state, self._blue_state, self._weapon_observation())

    def _dodge_reward(self):
        incoming = self._incoming("red")
        if not incoming: return 0.0
        speed_loss = (incoming.decision_start_speed-float(np.linalg.norm(incoming.velocity_neu)))/1000.0
        return float(np.clip(self._weapon_observation()["incoming_los_angle_ratio"] + speed_loss, -1, 1))

    def step(self, action):
        if self._red_state is None: raise RuntimeError("reset must be called before step")
        if isinstance(action, dict):
            maneuver = np.asarray(action["maneuver"], np.float32); fire = int(action["fire"])
        else:  # historical API compatibility only
            maneuver = np.asarray(action, np.float32); fire = 0
        previous_red, previous_blue, previous_obs = copy.copy(self._red_state), copy.copy(self._blue_state), self._observation()
        if fire and not self._launch("red"): self._invalid_fire_commands += 1
        # Official-project engineering rule: fire at first valid gate.
        if self._launch_allowed("blue"): self._launch("blue")
        event, physics_exception = self._advance_pair(
            action_to_targets(maneuver, self._red_state["heading"]),
            action_to_targets(self._blue_action(), self._blue_state["heading"]))
        self._step_count += 1; self._physics_exception = physics_exception
        self._numerical_invalid = event is not None and ("numerical_invalid" in event or event == "physics_exception")
        if self._numerical_invalid:
            self._red_state, self._blue_state, self._event = previous_red, previous_blue, event
            return previous_obs, terminal_reward(event), True, False, self._info()
        for state in (self._red_state, self._blue_state):
            self._flight_envelope_violation |= bool(state["altitude"] < 750.0 or
                state["true_airspeed"] > 400.0 or abs(state.get("load_factor",1.0)) > 9.0)
        geometry = relative_geometry(self._red_state, self._blue_state)
        reward, self._reward_components = step_reward(self._red_state, self._blue_state,
            geometry, event, self._dodge_reward())
        terminated = event is not None; truncated = not terminated and self._step_count >= self.max_steps
        if truncated: event = "timeout"
        self._event = event
        return self._observation(), reward, terminated, truncated, self._info()

    def _info(self):
        geometry = relative_geometry(self._red_state, self._blue_state)
        info = {"step":self._step_count, "event":self._event, "distance_m":geometry["distance_m"],
            "closure_mps":geometry["closure_mps"], "red_boresight_deg":float(np.rad2deg(geometry["boresight_angle"])),
            "blue_boresight_deg":float(np.rad2deg(relative_geometry(self._blue_state,self._red_state)["boresight_angle"])),
            "red_altitude_m":self._red_state["altitude"], "blue_altitude_m":self._blue_state["altitude"],
            "red_speed_mps":self._red_state["true_airspeed"], "blue_speed_mps":self._blue_state["true_airspeed"],
            "opponent_policy":self._actual_opponent_policy, "scenario_mode":self._actual_scenario_mode,
            "numerical_invalid":self._numerical_invalid, "physics_exception":self._physics_exception,
            "reward_components":dict(self._reward_components), "red_launch_count":self._launch_count["red"],
            "blue_launch_count":self._launch_count["blue"], "invalid_fire_commands":self._invalid_fire_commands,
            "flight_envelope_violation":self._flight_envelope_violation, "hit_time_s":self._last_hit_time,
            "active_missiles":sum(m.alive for m in self._missiles)}
        if self._event is not None: info.update(event_semantics(self._event))
        return info

    @property
    def red_state(self): return copy.copy(self._red_state)
    @property
    def blue_state(self): return copy.copy(self._blue_state)
    def close(self): self.red_sim = self.blue_sim = None
