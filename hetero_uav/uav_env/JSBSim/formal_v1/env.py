"""Isolated formal 3v2 heterogeneous air-combat environment."""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from ..pid_controller import PIDController
from ..simulator import AircraftSimulator
from .contract import (
    ACTION_DIM, ACTOR_OBS_DIM, BLUE_IDS, CRITIC_STATE_DIM, MAX_STEPS,
    RED_IDS, require_action, validate_formal_config,
)
from .missile import FormalMissile
from .observation import build_team_observations
from .opponent import PaperGreedyOpponent
from .reward import compute_role_rewards
from .scenario import MISSILE_COUNTS, ROLES, jsbsim_initial_state
from .targeting import fire_gate, select_target


class Hetero3v2PureHAPPOEnv(gym.Env):
    """Fixed 1 MAV + 2 attack UAV versus 2 rule UAV environment."""

    metadata = {"render_modes": []}

    def __init__(self, **config):
        env_type = str(config.get("env_type", ""))
        if env_type == "hetero_3v2_pure_happo_v2":
            from ..formal_v2.contract import (
                ACTOR_OBS_DIM as contract_actor_obs_dim,
                CRITIC_STATE_DIM as contract_critic_state_dim,
                ENV_TYPE as formal_contract,
                OBSERVATION_CONTRACT as observation_contract,
                REWARD_CONTRACT_VERSION as reward_contract,
                validate_formal_config as validate_config,
            )
            from ..formal_v2.observation import build_team_observations as observation_builder
            from ..formal_v2.reward import compute_role_rewards as reward_builder
        else:
            contract_actor_obs_dim = ACTOR_OBS_DIM
            contract_critic_state_dim = CRITIC_STATE_DIM
            formal_contract = "hetero_3v2_pure_happo_v1"
            observation_contract = "formal_entity_v1"
            from .reward import REWARD_CONTRACT_VERSION as reward_contract
            validate_config = validate_formal_config
            observation_builder = build_team_observations
            reward_builder = compute_role_rewards
        validate_config(config)
        self.config = dict(config)
        self.formal_contract = formal_contract
        self.observation_contract = observation_contract
        self.reward_contract = reward_contract
        self._observation_builder = observation_builder
        self._reward_builder = reward_builder
        self.red_ids = list(RED_IDS)
        self.blue_ids = list(BLUE_IDS)
        self.agent_ids = self.red_ids
        self.roles = dict(ROLES)
        self.sim_freq = int(config["sim_freq"])
        self.agent_interaction_steps = int(config["agent_interaction_steps"])
        self.max_steps = int(config["max_steps"])
        self.uav_detection_range_m = float(config["sensing"]["uav_direct_range_m"])
        self.mav_detection_range_m = float(config["sensing"]["mav_range_m"])
        missile = config["missile"]
        self.attack_range_m = float(missile["attack_range_m"])
        self.attack_interval_sec = float(missile["attack_interval_sec"])
        self.launch_ata_rad = math.radians(float(missile["launch_ata_deg"]))
        self.launch_ta_rad = math.radians(float(missile["launch_ta_deg"]))
        self.missile_config = dict(missile)
        self.origin = (120.02, 60.0, 0.0)
        self.action_dim = ACTION_DIM
        self.actor_obs_dim = contract_actor_obs_dim
        self.critic_state_dim = contract_critic_state_dim
        self.action_space = gym.spaces.Dict({
            aid: gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
            for aid in self.red_ids
        })
        self.observation_space = gym.spaces.Dict({
            aid: gym.spaces.Dict({
                "ego": gym.spaces.Box(-1, 1, shape=(11,), dtype=np.float32),
                "allies": gym.spaces.Box(-1, 1, shape=(2, 11), dtype=np.float32),
                "enemies": gym.spaces.Box(-1, 1, shape=(2, 14), dtype=np.float32),
                "incoming_missile": gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
                **({"fire_control": gym.spaces.Box(
                    0, 1, shape=(5,), dtype=np.float32)}
                   if self.formal_contract == "hetero_3v2_pure_happo_v2" else {}),
                "flat": gym.spaces.Box(
                    -1, 1, shape=(self.actor_obs_dim,), dtype=np.float32),
            }) for aid in self.red_ids
        })
        self.aircraft: dict[str, AircraftSimulator] = {}
        self.controllers: dict[str, PIDController] = {}
        self.blue_policy = PaperGreedyOpponent()
        self.missiles: list[FormalMissile] = []
        self.last_launch_time: dict[str, float] = {}
        self.step_count = 0
        self.sim_time_sec = 0.0
        self.event_log: list[dict] = []
        self.death_reasons: dict[str, str] = {}
        self.newly_dead: set[str] = set()
        self.selected_targets: dict[str, str | None] = {}
        self.last_control_targets: dict[str, tuple[float, float, float] | None] = {}
        self.last_critic_state = np.zeros(self.critic_state_dim, np.float32)
        self.previous_missile_risk = {aid: 0.0 for aid in self.red_ids}
        self.previous_missile_speed: dict[str, float] = {}
        self.last_fire_gates: dict[str, dict] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        perturbation = dict((options or {}).get("audit_initial_perturbation", {}))
        if not self.aircraft:
            for aid in (*self.red_ids, *self.blue_ids):
                self.aircraft[aid] = AircraftSimulator(
                    uid=aid, color="Red" if aid.startswith("red") else "Blue",
                    model="f16", init_state=jsbsim_initial_state(aid, perturbation), origin=self.origin,
                    sim_freq=self.sim_freq, num_missiles=MISSILE_COUNTS[aid],
                    suppress_jsbsim_output=bool(self.config.get("suppress_jsbsim_output", True)),
                )
        else:
            for aid, aircraft in self.aircraft.items():
                aircraft.reload(new_state=jsbsim_initial_state(aid, perturbation), new_origin=self.origin)
        for aid, aircraft in self.aircraft.items():
            aircraft.partners = [x for oid, x in self.aircraft.items()
                                 if oid != aid and oid.split("_")[0] == aid.split("_")[0]]
            aircraft.enemies = [x for oid, x in self.aircraft.items()
                                if oid.split("_")[0] != aid.split("_")[0]]
        self.controllers = {aid: PIDController(1.0 / self.sim_freq) for aid in self.aircraft}
        self.missiles = []
        self.last_launch_time = {aid: -self.attack_interval_sec for aid in self.aircraft}
        self.step_count = 0
        self.sim_time_sec = 0.0
        self.event_log = []
        self.death_reasons = {}
        self.newly_dead = set()
        self.selected_targets = {aid: None for aid in self.aircraft}
        self.last_control_targets = {aid: None for aid in self.aircraft}
        self.previous_missile_risk = {aid: 0.0 for aid in self.red_ids}
        self.previous_missile_speed = {}
        self.last_fire_gates = {}
        self.audit_initial_perturbation = perturbation
        obs, self.last_critic_state = self._observation_builder(self)
        return obs, self._info([], {})

    def step(self, actions: dict[str, Any]):
        if set(actions) != set(self.red_ids):
            raise ValueError(f"formal actions require exactly {self.red_ids}")
        red_actions = {aid: require_action(actions[aid], aid) for aid in self.red_ids}
        blue_decisions = self.blue_policy.decisions(self)
        blue_actions = {aid: row["action"] for aid, row in blue_decisions.items()}
        all_actions = {**red_actions, **blue_actions}
        self.newly_dead = set()
        alive_before = {aid: self.aircraft[aid].is_alive for aid in self.aircraft}
        self.selected_targets = {
            **{aid: select_target(self, aid) for aid in self.red_ids},
            **{aid: blue_decisions[aid]["target_id"] for aid in self.blue_ids},
        }
        launch_records = self._automatic_fire()

        numeric_anomaly = False
        missile_events: list[dict] = []
        for _ in range(self.agent_interaction_steps):
            for aid, action in all_actions.items():
                aircraft = self.aircraft[aid]
                if not aircraft.is_alive:
                    self.last_control_targets[aid] = None
                    continue
                target = self._decode_action(action)
                self.last_control_targets[aid] = target
                speed = float(np.linalg.norm(aircraft.get_velocity()))
                vn, ve, vu = aircraft.get_velocity()
                controls = self.controllers[aid].compute_control(
                    aircraft.get_rpy(), speed, *target, ned_velocity=(vn, ve, -vu))
                for prop, value in zip(("fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
                                        "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"), controls):
                    aircraft.set_property_value(prop, float(value))
                aircraft.run()
                if not np.isfinite(np.r_[aircraft.get_position(), aircraft.get_velocity(),
                                         aircraft.get_rpy(), aircraft.get_geodetic()]).all():
                    aircraft.crash()
                    self.death_reasons.setdefault(aid, "numeric_anomaly")
                    numeric_anomaly = True
                elif aircraft.get_position()[2] < 100.0:
                    aircraft.crash()
                    self.death_reasons.setdefault(aid, "crash")
                elif (np.linalg.norm(aircraft.get_position()[:2]) > 50_000.0
                      or aircraft.get_position()[2] > 10_000.0):
                    aircraft.crash()
                    self.death_reasons.setdefault(aid, "out_of_zone")
            for missile in self.missiles:
                event = missile.step(1.0 / self.sim_freq, self.aircraft[missile.target_id])
                if event:
                    missile_events.append(event)
                    self.event_log.append({**event, "step": self.step_count})
                    if event["event"] == "hit":
                        self.death_reasons.setdefault(event["target_id"], "missile_hit")
            self.sim_time_sec += 1.0 / self.sim_freq

        for aid, was_alive in alive_before.items():
            if was_alive and not self.aircraft[aid].is_alive:
                self.newly_dead.add(aid)
        self.step_count += 1
        rewards, reward_components = self._reward_builder(
            self, self.selected_targets, missile_events)
        for aid in self.red_ids:
            self.previous_missile_risk[aid] = float(
                reward_components["per_agent"][aid].get("missile_risk", 0.0))
        red_alive = sum(self.aircraft[aid].is_alive for aid in self.red_ids)
        blue_alive = sum(self.aircraft[aid].is_alive for aid in self.blue_ids)
        terminated = bool(numeric_anomaly or red_alive == 0 or blue_alive == 0)
        truncated = bool(not terminated and self.step_count >= self.max_steps)
        if numeric_anomaly:
            outcome, reason = "invalid", "numeric_anomaly"
        elif red_alive == 0 and blue_alive == 0:
            outcome, reason = "mutual_elimination", "mutual_elimination"
        elif red_alive == 0:
            outcome, reason = "blue_win", "red_eliminated"
        elif blue_alive == 0:
            outcome, reason = "red_win", "blue_eliminated"
        elif truncated:
            outcome, reason = "draw", "timeout"
        else:
            outcome, reason = "ongoing", ""
        observations, self.last_critic_state = self._observation_builder(self)
        terminations = {aid: terminated for aid in self.red_ids}
        truncations = {aid: truncated for aid in self.red_ids}
        info = self._info(launch_records + missile_events, reward_components,
                          alive_before=alive_before)
        info.update({"outcome": outcome, "end_reason": reason, "team_done": terminated or truncated})
        return observations, rewards, terminations, truncations, info

    def _automatic_fire(self) -> list[dict]:
        records = []
        self.last_fire_gates = {}
        for aid in (*self.red_ids, *self.blue_ids):
            target_id = self.selected_targets[aid]
            allowed, gate = fire_gate(self, aid, target_id)
            self.last_fire_gates[aid] = {"target_id": target_id, **gate}
            if not allowed:
                continue
            shooter = self.aircraft[aid]
            target = self.aircraft[target_id]
            velocity = shooter.get_velocity().copy()
            direction = velocity / max(np.linalg.norm(velocity), 1e-6)
            missile_id = f"{aid}_m{shooter.num_missiles - shooter.num_left_missiles}"
            missile = FormalMissile(
                missile_id, aid, target_id, shooter.get_position().copy(),
                direction * float(self.missile_config["speed_mps"]),
                navigation_gain=float(self.missile_config["navigation_gain"]),
                max_overload_g=float(self.missile_config["max_overload_g"]),
                speed_mps=float(self.missile_config["speed_mps"]),
                hit_radius_m=float(self.missile_config["hit_radius_m"]),
                max_flight_time_sec=float(self.missile_config["max_flight_time_sec"]),
                arming_time_sec=float(self.missile_config["arming_time_sec"]),
            )
            self.missiles.append(missile)
            shooter.num_left_missiles -= 1
            self.last_launch_time[aid] = self.sim_time_sec
            record = {"event": "launch", "step": self.step_count, "missile_id": missile_id,
                      "shooter_id": aid, "target_id": target_id, **gate}
            records.append(record)
            self.event_log.append(record)
        return records

    @staticmethod
    def _decode_action(action: np.ndarray) -> tuple[float, float, float]:
        return (float(action[0]) * np.pi / 2, float(action[1]) * np.pi,
                102.0 + (float(action[2]) + 1.0) * 153.0)

    def _info(self, step_events: list[dict], reward_components: dict,
              alive_before: dict | None = None) -> dict:
        return {
            "formal_contract": self.formal_contract,
            "observation_contract": self.observation_contract,
            "reward_contract": self.reward_contract,
            "critic_state": self.last_critic_state.copy(),
            "active_mask": np.asarray([self.aircraft[aid].is_alive for aid in self.red_ids], np.float32),
            "alive_before_mask": np.asarray([
                (alive_before or {}).get(aid, self.aircraft[aid].is_alive) for aid in self.red_ids
            ], np.float32),
            "selected_targets": dict(self.selected_targets),
            "fire_gates": {aid: dict(gate) for aid, gate in self.last_fire_gates.items()},
            "control_targets": dict(self.last_control_targets),
            "step_events": list(step_events),
            "reward_components": reward_components,
            "red_alive": sum(self.aircraft[aid].is_alive for aid in self.red_ids),
            "blue_alive": sum(self.aircraft[aid].is_alive for aid in self.blue_ids),
            "mav_alive": bool(self.aircraft.get("red_0") and self.aircraft["red_0"].is_alive),
            "death_reasons": dict(self.death_reasons),
            "audit_initial_perturbation": getattr(self, "audit_initial_perturbation", {}),
        }

    def close(self):
        for aircraft in getattr(self, "aircraft", {}).values():
            aircraft.close()


def make_formal_env(**config) -> Hetero3v2PureHAPPOEnv:
    return Hetero3v2PureHAPPOEnv(**config)
