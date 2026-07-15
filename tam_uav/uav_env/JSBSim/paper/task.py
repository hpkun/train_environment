"""Unified 60 Hz execution loop for the isolated paper environment."""

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
        decision_frequency = float(self.published["decision_frequency_hz"])
        if not np.isclose(decision_frequency,
                          self.simulation_frequency / self.physics_frames):
            raise ValueError("decision_frequency_hz must equal simulation_frequency_hz / physics_frames_per_action")
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
        self.opponent = GreedyPaperOpponent(self.published, self.inferred)
        self.agents = []
        self.current_targets: dict[str, str | None] = {}
        self.target_scores: dict[str, dict] = {}
        self.target_diagnostics: dict[str, dict] = {}
        self.step_count = 0
        self.simulation_time_s = 0.0
        self.episode_return = {}
        self.crashes = self.out_of_zone = self.structural_failures = 0
        self.last_info = {}

    def reset(self, rng: np.random.Generator):
        self.agents = self.scenario.build(rng)
        self._link_agents()
        self.weapon.reset()
        self.reward.reset()
        self.current_targets = {a.agent_id: None for a in self.agents}
        self.target_scores = {}
        self.target_diagnostics = {}
        self.step_count = 0
        self.simulation_time_s = 0.0
        self.crashes = self.out_of_zone = self.structural_failures = 0
        self.episode_return = {a.agent_id: 0.0 for a in self.agents}
        obs = self.observation.build(self.agents, self.weapon.missiles)
        self._update_targets(obs)
        info = self._build_info([], {}, {}, None, None, {}, {}, {})
        self.last_info = info
        return obs, info

    def step(self, actions):
        pre_obs = self.observation.build(self.agents, self.weapon.missiles)
        self._update_targets(pre_obs)
        alive_at_start = {a.agent_id: a.alive for a in self.agents}
        self.weapon.begin_decision_step()
        by_id = {a.agent_id: a for a in self.agents}

        action_map = self._normalize_actions(actions)
        opponent_diag = {}
        for agent in self.agents:
            if agent.side == "blue" and agent.alive:
                target = by_id.get(self.current_targets.get(agent.agent_id))
                incoming = [m for m in self.weapon.missiles
                            if m.alive and m.target_id == agent.agent_id]
                action_map[agent.agent_id], opponent_diag[agent.agent_id] = (
                    self.opponent.act(agent, target, incoming))

        events = []
        for shooter in self.agents:
            target = by_id.get(self.current_targets.get(shooter.agent_id))
            launch = self.weapon.try_launch(
                shooter, target, self._target_visible(shooter, target, pre_obs),
                self.simulation_time_s)
            if launch:
                events.append(launch)

        commands = {}
        action_indices = {}
        for agent in self.agents:
            indices = np.asarray(action_map.get(agent.agent_id, np.full(4, 20)), dtype=np.int64)
            if not agent.alive:
                indices = np.full(4, 20, dtype=np.int64)
            action_indices[agent.agent_id] = indices.copy()
            commands[agent.agent_id] = self.map_action(indices)

        death_events = {}
        out_step, crash_step, structural_step = set(), set(), set()
        for _ in range(self.physics_frames):
            for agent in self.agents:
                if agent.alive:
                    agent.apply_direct_fcs_command(commands[agent.agent_id])
            for agent in self.agents:
                if agent.alive:
                    agent.step_physics_once(self.physics_dt)
            frame_out, frame_crash, frame_structural = self._apply_constraints_once()
            out_step |= frame_out
            crash_step |= frame_crash
            structural_step |= frame_structural
            for agent in self.agents:
                if alive_at_start[agent.agent_id] and not agent.alive:
                    death_events.setdefault(agent.agent_id, agent.death_reason)
            events.extend(self.weapon.step_physics_once(by_id, self.physics_dt))
            for agent in self.agents:
                if alive_at_start[agent.agent_id] and not agent.alive:
                    death_events.setdefault(agent.agent_id, agent.death_reason)
            self.simulation_time_s += self.physics_dt

        self.step_count += 1
        reward_scores = self._selected_target_scores_at_end()
        terminated, truncated, winner, reason = self._termination()
        rewards, components = self.reward.compute(
            self.agents, self.current_targets, reward_scores,
            self.weapon.missiles, events, out_step, alive_at_start)
        for aid, value in rewards.items():
            self.episode_return[aid] += float(value)
        obs = self.observation.build(self.agents, self.weapon.missiles)
        alive_at_end = {a.agent_id: a.alive for a in self.agents}
        just_died = {aid: alive_at_start[aid] and not alive_at_end[aid]
                     for aid in alive_at_start}
        info = self._build_info(events, components, opponent_diag, winner, reason,
                                alive_at_start, alive_at_end, just_died)
        info.update({
            "death_reason": {a.agent_id: a.death_reason for a in self.agents},
            "crash_step": sorted(crash_step), "out_of_zone_step": sorted(out_step),
            "structural_failure_step": sorted(structural_step),
            "low_altitude_violations": sorted(
                a.agent_id for a in self.agents if a.alive and
                a.position[2] < self.published["minimum_safe_altitude_m"]),
            "action_indices": {aid: value.tolist() for aid, value in action_indices.items()},
            "direct_fcs_commands": {aid: value.tolist() for aid, value in commands.items()},
        })
        self.last_info = info
        term = {a.agent_id: bool(terminated or not a.alive) for a in self.agents}
        trunc = {a.agent_id: bool(truncated) for a in self.agents}
        return obs, rewards, term, trunc, info

    @staticmethod
    def map_action(indices) -> np.ndarray:
        values = np.clip(np.asarray(indices, dtype=np.int64).reshape(4), 0, 39)
        return np.array([0.4 + values[0] / 39.0 * 0.5,
                         -1.0 + values[1] / 39.0 * 2.0,
                         -1.0 + values[2] / 39.0 * 2.0,
                         -1.0 + values[3] / 39.0 * 2.0], dtype=np.float64)

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
        diagnostics, all_scores = {}, {}
        for ego in self.agents:
            previous = self.current_targets.get(ego.agent_id)
            enemies = sides["blue" if ego.side == "red" else "red"]
            visible_mask = obs[ego.agent_id]["enemy_mask"]
            scores = {}
            if ego.alive:
                for idx, target in enumerate(enemies):
                    if idx < len(visible_mask) and visible_mask[idx] > 0.5 and target.alive:
                        scores[target.agent_id] = assess_pair(
                            ego.position, ego.velocity, target.position, target.velocity,
                            self.published["maximum_attack_range_m"],
                            self.inferred["situation_height_norm_m"],
                            self.published["maximum_speed_mps"])
            current = select_best_target(scores)
            self.current_targets[ego.agent_id] = current
            all_scores[ego.agent_id] = scores
            diagnostics[ego.agent_id] = {
                "previous_target": previous, "current_target": current,
                "target_changed": previous != current,
                "target_scores": {tid: score.score for tid, score in scores.items()},
            }
        self.target_scores = all_scores
        self.target_diagnostics = diagnostics

    def _selected_target_scores_at_end(self):
        by_id = {a.agent_id: a for a in self.agents}
        result = {}
        for ego in self.agents:
            target = by_id.get(self.current_targets.get(ego.agent_id))
            result[ego.agent_id] = {}
            if target is not None:
                result[ego.agent_id][target.agent_id] = assess_pair(
                    ego.position, ego.velocity, target.position, target.velocity,
                    self.published["maximum_attack_range_m"],
                    self.inferred["situation_height_norm_m"],
                    self.published["maximum_speed_mps"])
        return result

    def _target_visible(self, shooter, target, obs):
        if target is None:
            return False
        enemies = sorted([a for a in self.agents if a.side != shooter.side], key=lambda a: a.agent_id)
        idx = next((i for i, a in enumerate(enemies) if a.agent_id == target.agent_id), None)
        return idx is not None and obs[shooter.agent_id]["enemy_mask"][idx] > 0.5

    def _apply_constraints_once(self):
        out_step, crash_step, structural_step = set(), set(), set()
        radius = float(self.inferred["combat_zone_radius_m"])
        grace = float(self.inferred["structural_limit_grace_s"])
        for agent in self.agents:
            if not agent.alive:
                continue
            finite = np.isfinite(np.concatenate([agent.position, agent.velocity])).all()
            if not finite or agent.position[2] <= 0.0:
                agent.kill("nonfinite" if not finite else "crash")
                crash_step.add(agent.agent_id)
                self.crashes += 1
                continue
            if np.linalg.norm(agent.position[:2]) > radius:
                agent.kill("boundary")
                out_step.add(agent.agent_id)
                self.out_of_zone += 1
                continue
            if agent.speed > float(self.published["maximum_speed_mps"]):
                agent.speed_violation_time_s += self.physics_dt
                agent.speed_violation_count += 1
            else:
                agent.speed_violation_time_s = 0.0
            if abs(agent.load_factor_g) > float(self.published["maximum_aircraft_overload_g"]):
                agent.overload_violation_time_s += self.physics_dt
                agent.overload_violation_count += 1
            else:
                agent.overload_violation_time_s = 0.0
            reason = None
            if agent.speed_violation_time_s + 1e-12 >= grace:
                reason = "structural_speed"
            elif agent.overload_violation_time_s + 1e-12 >= grace:
                reason = "structural_overload"
            if reason:
                agent.kill(reason)
                structural_step.add(agent.agent_id)
                self.structural_failures += 1
        return out_step, crash_step, structural_step

    def _combat_units(self, side, alive_only=True):
        return [a for a in self.agents if a.side == side
                and a.aircraft_type.role != "mav" and (a.alive or not alive_only)]

    def _termination(self):
        red_alive = self._combat_units("red")
        blue_alive = self._combat_units("blue")
        if not blue_alive:
            return True, False, "red", "blue_combat_units_eliminated"
        if not red_alive:
            return True, False, "blue", "red_combat_units_eliminated"
        if self.step_count >= self.episode_limit:
            if len(red_alive) != len(blue_alive):
                winner = "red" if len(red_alive) > len(blue_alive) else "blue"
            else:
                red_kills = sum(not a.alive and a.death_reason == "shotdown"
                                for a in self._combat_units("blue", False))
                blue_kills = sum(not a.alive and a.death_reason == "shotdown"
                                 for a in self._combat_units("red", False))
                winner = "red" if red_kills > blue_kills else "blue" if blue_kills > red_kills else "draw"
            return False, True, winner, "episode_limit"
        return False, False, None, None

    def _build_info(self, events, components, opponent_diag, winner, reason,
                    alive_start, alive_end, just_died):
        red = [a for a in self.agents if a.side == "red"]
        blue = [a for a in self.agents if a.side == "blue"]
        info = {
            "paper_environment_mode": "tam_paper_env_v1",
            "backend": self.scenario.dynamics_backend,
            "winner": winner, "termination_reason": reason,
            "red_alive": sum(a.alive for a in red), "blue_alive": sum(a.alive for a in blue),
            "red_combat_alive": len(self._combat_units("red")),
            "blue_combat_alive": len(self._combat_units("blue")),
            "mav_alive": any(a.alive and a.aircraft_type.role == "mav" for a in red),
            "kills": {"red": sum(a.death_reason == "shotdown" for a in blue),
                      "blue": sum(a.death_reason == "shotdown" for a in red)},
            "shotdown": sum(a.death_reason == "shotdown" for a in self.agents),
            "crashes": self.crashes, "out_of_zone": self.out_of_zone,
            "structural_failures": self.structural_failures,
            "missiles_fired": self.weapon.total_fired, "missile_hits": self.weapon.total_hits,
            "missiles_left": {a.agent_id: a.missile_left for a in self.agents},
            "current_targets": dict(self.current_targets),
            "target_selection": self.target_diagnostics,
            "missile_events": list(events),
            "missile_telemetry": [m.telemetry() for m in self.weapon.missiles],
            "reward_components": components, "opponent_diagnostics": opponent_diag,
            "episode_step": self.step_count, "simulation_time_s": self.simulation_time_s,
            "episode_return": dict(self.episode_return),
            "alive_at_step_start": alive_start, "alive_at_step_end": alive_end,
            "just_died_this_step": just_died,
            "aircraft_metrics": {a.agent_id: {
                "max_speed_mps": a.max_speed_observed_mps,
                "max_abs_load_factor_g": a.max_abs_load_factor_g,
                "speed_violation_count": a.speed_violation_count,
                "overload_violation_count": a.overload_violation_count,
                "death_reason": a.death_reason,
            } for a in self.agents},
        }
        info["missile_termination_reasons"] = {
            key: sum(m.termination_reason == key for m in self.weapon.missiles)
            for key in ("hit", "timeout", "target_dead", "nonfinite")
        }
        for agent in self.agents:
            info[agent.agent_id] = {
                "alive": agent.alive, "role": agent.aircraft_type.role,
                "current_target_id": self.current_targets.get(agent.agent_id),
                "missiles_left": agent.missile_left,
                "reward_components": dict(components.get(agent.agent_id, {})),
            }
        return info

    def _link_agents(self):
        for agent in self.agents:
            agent.partners = [a for a in self.agents if a.side == agent.side and a is not agent]
            agent.enemies = [a for a in self.agents if a.side != agent.side]
