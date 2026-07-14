"""Composition root for the isolated tam_paper_env_v1 semantics."""

from __future__ import annotations

import numpy as np

from ..core.scenario import ScenarioBuilder
from .observation import PaperObservation
from .opponent import GreedyPaperOpponent
from .reward import PaperReward
from .situation import assess_pair, select_best_target
from .weapon import PaperWeaponManager


class TAMPaperTask:
    def __init__(self, config: dict):
        self.config = config
        self.published = dict(config["published_parameters"])
        self.inferred = dict(config["inferred_parameters"])
        self.simulation_frequency = int(self.published["simulation_frequency_hz"])
        self.physics_frames = int(self.published["physics_frames_per_action"])
        self.physics_dt = 1.0 / self.simulation_frequency
        self.decision_dt = self.physics_frames * self.physics_dt
        self.episode_limit = int(self.published["episode_limit_steps"])
        self.controlled_side = "red"
        self.scenario = ScenarioBuilder(config)
        self.observation = PaperObservation(
            self.published, self.inferred,
            int(config["max_red_agents"]), int(config["max_blue_agents"]))
        self.weapon = PaperWeaponManager(self.published, self.inferred)
        self.reward = PaperReward(self.published, self.inferred)
        self.opponent = GreedyPaperOpponent()
        self.agents = []
        self.current_targets: dict[str, str | None] = {}
        self.target_scores: dict[str, dict] = {}
        self.step_count = 0
        self.episode_return = {}
        self.crashes = self.out_of_zone = 0
        self.last_info = {}

    def reset(self, rng: np.random.Generator):
        self.agents = self.scenario.build(rng)
        self._link_agents()
        self.weapon.reset()
        self.reward.reset()
        self.current_targets = {a.agent_id: None for a in self.agents}
        self.target_scores = {}
        self.step_count = self.crashes = self.out_of_zone = 0
        self.episode_return = {a.agent_id: 0.0 for a in self.agents}
        obs = self.observation.build(self.agents, self.weapon.missiles)
        self._update_targets(obs)
        info = self._build_info([], {}, {}, None, None)
        self.last_info = info
        return obs, info

    def step(self, actions):
        pre_obs = self.observation.build(self.agents, self.weapon.missiles)
        self._update_targets(pre_obs)
        action_map = self._normalize_actions(actions)
        opponent_diag = {}
        for agent in self.agents:
            if agent.side == "blue" and agent.alive:
                action_map[agent.agent_id], opponent_diag[agent.agent_id] = self.opponent.act(
                    agent.agent_id, pre_obs[agent.agent_id])
        for agent in self.agents:
            indices = action_map.get(agent.agent_id, np.full(4, 20, dtype=np.int64))
            if not agent.alive:
                indices = np.full(4, 20, dtype=np.int64)
            agent.step_direct_fcs(self._apply_static_trim(agent, self.map_action(indices)), self.physics_frames,
                                  self.simulation_frequency,
                                  float(self.published["maximum_speed_mps"]))
        out_step, crash_step, overload_step = self._apply_constraints()
        post_obs = self.observation.build(self.agents, self.weapon.missiles)
        self._update_targets(post_obs)
        by_id = {a.agent_id: a for a in self.agents}
        events = []
        now = self.step_count * self.decision_dt
        for shooter in self.agents:
            target = by_id.get(self.current_targets.get(shooter.agent_id))
            visible = self._target_visible(shooter, target, post_obs)
            launch = self.weapon.try_launch(shooter, target, visible, now)
            if launch:
                events.append(launch)
        events.extend(self.weapon.step(by_id, self.physics_dt, self.physics_frames))
        self.step_count += 1
        terminated, truncated, winner, reason = self._termination()
        rewards, components = self.reward.compute(
            self.agents, self.current_targets, self.target_scores,
            self.weapon.missiles, events, out_step)
        for aid, value in rewards.items():
            self.episode_return[aid] += float(value)
        obs = self.observation.build(self.agents, self.weapon.missiles)
        info = self._build_info(events, components, opponent_diag, winner, reason)
        info["overload_violations"] = sorted(overload_step)
        info["low_altitude_violations"] = sorted(
            a.agent_id for a in self.agents if a.alive and
            a.position[2] < self.published["minimum_safe_altitude_m"])
        info["crash_step"] = sorted(crash_step)
        info["out_of_zone_step"] = sorted(out_step)
        self.last_info = info
        term = {a.agent_id: bool(terminated or not a.alive) for a in self.agents}
        trunc = {a.agent_id: bool(truncated) for a in self.agents}
        return obs, rewards, term, trunc, info

    def map_action(self, indices) -> np.ndarray:
        values = np.clip(np.asarray(indices, dtype=np.int64).reshape(4), 0, 39)
        return np.array([0.4 + values[0] / 39.0 * 0.5,
                         -1.0 + values[1] / 39.0 * 2.0,
                         -1.0 + values[2] / 39.0 * 2.0,
                         -1.0 + values[3] / 39.0 * 2.0], dtype=np.float64)

    def _apply_static_trim(self, agent, command: np.ndarray) -> np.ndarray:
        trim = self.inferred.get("direct_fcs_static_trim_by_model", {}).get(
            agent.aircraft_type.aircraft_model, {})
        result = np.asarray(command, dtype=np.float64).copy()
        for idx, key in enumerate(("throttle", "aileron", "elevator", "rudder")):
            result[idx] += float(trim.get(key, 0.0))
        result[0] = np.clip(result[0], 0.4, 0.9)
        result[1:] = np.clip(result[1:], -1.0, 1.0)
        return result

    def _normalize_actions(self, actions):
        if isinstance(actions, dict):
            return {str(k): np.asarray(v, dtype=np.int64) for k, v in actions.items()}
        controlled = self.controlled_agents()
        arr = np.asarray(actions, dtype=np.int64)
        return {agent.agent_id: arr[idx] for idx, agent in enumerate(controlled)}

    def controlled_agents(self):
        return [a for a in self.agents if a.side == "red"]

    def controlled_agent_ids_from_config(self):
        return [str(entry.get("id", f"red_{idx}"))
                for idx, entry in enumerate(self.config["red_agents"])]

    def _update_targets(self, obs):
        sides = {side: sorted([a for a in self.agents if a.side == side], key=lambda a: a.agent_id)
                 for side in ("red", "blue")}
        self.target_scores = {}
        for ego in self.agents:
            enemies = sides["blue" if ego.side == "red" else "red"]
            visible_mask = obs[ego.agent_id]["enemy_mask"]
            scores = {}
            for idx, target in enumerate(enemies):
                if idx < len(visible_mask) and visible_mask[idx] > 0.5 and target.alive:
                    scores[target.agent_id] = assess_pair(
                        ego.position, ego.velocity, target.position, target.velocity,
                        self.published["maximum_attack_range_m"],
                        self.inferred["situation_height_norm_m"],
                        self.published["maximum_speed_mps"])
            self.target_scores[ego.agent_id] = scores
            current = self.current_targets.get(ego.agent_id)
            if current not in scores:
                current = select_best_target(scores)
            self.current_targets[ego.agent_id] = current

    def _target_visible(self, shooter, target, obs):
        if target is None:
            return False
        enemies = sorted([a for a in self.agents if a.side != shooter.side], key=lambda a: a.agent_id)
        idx = next((i for i, a in enumerate(enemies) if a.agent_id == target.agent_id), None)
        return idx is not None and obs[shooter.agent_id]["enemy_mask"][idx] > 0.5

    def _apply_constraints(self):
        out_step, crash_step, overload_step = set(), set(), set()
        radius = float(self.inferred["combat_zone_radius_m"])
        max_speed = float(self.published["maximum_speed_mps"])
        for agent in self.agents:
            if not agent.alive:
                continue
            finite = np.isfinite(np.concatenate([agent.position, agent.velocity])).all()
            if not finite or agent.position[2] <= 0.0:
                agent.kill("crash")
                crash_step.add(agent.agent_id)
                self.crashes += 1
            elif np.linalg.norm(agent.position[:2]) > radius:
                agent.kill("boundary")
                out_step.add(agent.agent_id)
                self.out_of_zone += 1
            if agent.speed > max_speed:
                agent.velocity *= max_speed / max(agent.speed, 1e-8)
            jsb = getattr(agent, "jsbsim_exec", None)
            if jsb is not None:
                load = abs(float(jsb.get_property_value("accelerations/Nz")))
                if load > float(self.published["maximum_aircraft_overload_g"]):
                    overload_step.add(agent.agent_id)
        return out_step, crash_step, overload_step

    def _termination(self):
        red_alive = [a for a in self.agents if a.side == "red" and a.alive]
        blue_alive = [a for a in self.agents if a.side == "blue" and a.alive]
        if not blue_alive:
            return True, False, "red", "blue_all_inactive"
        if not red_alive:
            return True, False, "blue", "red_all_inactive"
        if self.step_count >= self.episode_limit:
            if len(red_alive) > len(blue_alive):
                winner = "red"
            elif len(blue_alive) > len(red_alive):
                winner = "blue"
            else:
                winner = "draw"
            return False, True, winner, "episode_limit"
        return False, False, None, None

    def _build_info(self, events, components, opponent_diag, winner, reason):
        red = [a for a in self.agents if a.side == "red"]
        blue = [a for a in self.agents if a.side == "blue"]
        info = {
            "paper_environment_mode": "tam_paper_env_v1",
            "winner": winner, "termination_reason": reason,
            "red_alive": sum(a.alive for a in red), "blue_alive": sum(a.alive for a in blue),
            "mav_alive": any(a.alive and a.aircraft_type.role == "mav" for a in red),
            "kills": {"red": sum(not a.alive and not a.crashed and not a.out_of_boundary for a in blue),
                      "blue": sum(not a.alive and not a.crashed and not a.out_of_boundary for a in red)},
            "crashes": self.crashes, "out_of_zone": self.out_of_zone,
            "missiles_fired": self.weapon.total_fired, "missile_hits": self.weapon.total_hits,
            "missiles_left": {a.agent_id: a.missile_left for a in self.agents},
            "current_targets": dict(self.current_targets), "missile_events": list(events),
            "reward_components": components, "opponent_diagnostics": opponent_diag,
            "episode_step": self.step_count, "episode_return": dict(self.episode_return),
        }
        info["missile_termination_reasons"] = {
            key: sum(m.termination_reason == key for m in self.weapon.missiles)
            for key in ("hit", "timeout", "target_dead", "nonfinite")
        }
        for agent in self.agents:
            info[agent.agent_id] = {
                "alive": agent.alive,
                "role": agent.aircraft_type.role,
                "current_target_id": self.current_targets.get(agent.agent_id),
                "missiles_left": agent.missile_left,
                "reward_components": dict(components.get(agent.agent_id, {})),
            }
        return info

    def _link_agents(self):
        for agent in self.agents:
            agent.partners = [a for a in self.agents if a.side == agent.side and a is not agent]
            agent.enemies = [a for a in self.agents if a.side != agent.side]
