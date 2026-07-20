"""Unified 60 Hz execution loop for the isolated paper environment."""

from __future__ import annotations

import numpy as np

from ..core.scenario import ScenarioBuilder
from .observation import PaperObservation
from .opponent import GreedyPaperOpponent
from .reward import PaperReward
from .situation import assess_pair, select_best_target
from .weapon import PaperWeaponManager
from .action_semantics import INACTIVE_ACTION_PLACEHOLDER, map_action_indices
from .protocol import (
    BLUE_POLICY_FIDELITY,
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION,
    PAPER_5V4_GENERALIZATION_PROTOCOL, PAPER_NOMINAL_PROTOCOL,
    PAPER_SILENT_ASSUMPTIONS_PRESENT, REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED,
    BASIC_MANOEUVRE_ACTION_MAPPING_UNPUBLISHED,
    TERMINATION_RESOLUTION,
    environment_values, validate_parameter_provenance)


class TAMPaperTask:
    def __init__(self, config: dict):
        self.config = config
        validate_parameter_provenance(config)
        self.published = dict(config["published_parameters"])
        self.unpublished = dict(config["unpublished_parameters"])
        self.derived = environment_values(self.unpublished)
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
        self.initial_conditions = config["scenario_initial_conditions"]
        self.scenario_name = str(config.get(
            "scenario", f"{len(self.initial_conditions['red_agents'])}v"
                        f"{len(self.initial_conditions['blue_agents'])}"))
        self.initial_perturbation = str(
            config.get("initial_perturbation", NOMINAL_PERTURBATION))
        self.experiment_protocol = str(config.get(
            "experiment_protocol",
            PAPER_NOMINAL_PROTOCOL if self.initial_perturbation == NOMINAL_PERTURBATION
            else "pre_fidelity_diagnostic"))
        self.scenario = ScenarioBuilder(config)
        self.observation = PaperObservation(
            self.published, self.unpublished)
        self.weapon = PaperWeaponManager(self.published, self.unpublished)
        self.reward = PaperReward(self.published, self.unpublished)
        self.opponent = GreedyPaperOpponent(self.published, self.unpublished)
        self.agents = []
        self.current_targets: dict[str, str | None] = {}
        self.target_scores: dict[str, dict] = {}
        self.target_diagnostics: dict[str, dict] = {}
        self.step_count = 0
        self.simulation_time_s = 0.0
        self.episode_return = {}
        self.crashes = self.out_of_zone = self.structural_failures = 0
        self.decision_context_counter = 0
        self._decision_context = None
        self.event_sequence_counter = 0
        self.target_consistency_violation_count = 0
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
        self.decision_context_counter = 0
        self._decision_context = None
        self.event_sequence_counter = 0
        self.target_consistency_violation_count = 0
        self.episode_return = {a.agent_id: 0.0 for a in self.agents}
        self._update_targets()
        obs = self.observation.build(
            self.agents, self.weapon.missiles, self.current_targets)
        info = self._build_info([], {}, {}, None, None, {}, {}, {})
        self.last_info = info
        return obs, info

    def prepare_decision_context(self):
        """Prepare and freeze the target snapshot for exactly one decision step."""
        if self._decision_context is not None:
            return self._decision_context
        self._update_targets()
        pre_obs = self.observation.build(
            self.agents, self.weapon.missiles, self.current_targets)
        self.decision_context_counter += 1
        by_id = {agent.agent_id: agent for agent in self.agents}
        self._decision_context = {
            "decision_context_id": self.decision_context_counter,
            "pre_observation": pre_obs,
            "targets": dict(self.current_targets),
            "rule_inputs": {
                agent.agent_id: {
                    "agent": agent,
                    "target": by_id.get(self.current_targets.get(agent.agent_id)),
                    "incoming_missiles": [
                        missile for missile in self.weapon.missiles
                        if missile.alive and missile.target_id == agent.agent_id
                    ],
                }
                for agent in self.agents if agent.alive
            },
            "target_used_by_rule_action": {},
        }
        return self._decision_context

    def build_rule_actions(self, agent_ids=None):
        context = self.prepare_decision_context()
        selected = set(agent_ids) if agent_ids is not None else {
            agent.agent_id for agent in self.controlled_agents()
        }
        actions = {}
        for aid, inputs in context["rule_inputs"].items():
            if aid not in selected:
                continue
            action, _ = self.opponent.act(
                inputs["agent"], inputs["target"], inputs["incoming_missiles"])
            actions[aid] = action
            context["target_used_by_rule_action"][aid] = (
                inputs["target"].agent_id if inputs["target"] is not None else None)
        return actions

    def step(self, actions):
        context = self.prepare_decision_context()
        self.current_targets = dict(context["targets"])
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
                context["target_used_by_rule_action"][agent.agent_id] = (
                    target.agent_id if target is not None else None)

        events = []
        target_used_by_weapon = {}
        for shooter in self.agents:
            target = by_id.get(self.current_targets.get(shooter.agent_id))
            target_used_by_weapon[shooter.agent_id] = (
                target.agent_id if target is not None else None)
            launch = self.weapon.try_launch(
                shooter, target, self.simulation_time_s)
            if launch:
                events.append(self._stamp_event(
                    launch, physics_frame_index=-1,
                    simulation_time_s=self.simulation_time_s))

        commands = {}
        action_indices = {}
        for agent in self.agents:
            indices = np.asarray(action_map.get(
                agent.agent_id, INACTIVE_ACTION_PLACEHOLDER), dtype=np.int64)
            if not agent.alive:
                # Inactive placeholder only; never applied to a dead aircraft.
                indices = np.asarray(INACTIVE_ACTION_PLACEHOLDER, dtype=np.int64)
            action_indices[agent.agent_id] = indices.copy()
            commands[agent.agent_id] = self.map_action(indices)

        recorded_deaths = {aid for aid, alive in alive_at_start.items() if not alive}
        out_step, crash_step = set(), set()
        for physics_frame_index in range(self.physics_frames):
            for agent in self.agents:
                if agent.alive:
                    agent.apply_direct_fcs_command(commands[agent.agent_id])
            for agent in self.agents:
                if agent.alive:
                    agent.step_physics_once(self.physics_dt)
            frame_out, frame_crash = self._apply_constraints_once()
            out_step |= frame_out
            crash_step |= frame_crash
            frame_time = self.simulation_time_s + self.physics_dt
            self._record_new_deaths(
                events, recorded_deaths, physics_frame_index, frame_time)
            missile_events = self.weapon.step_physics_once(by_id, self.physics_dt)
            for missile_event in missile_events:
                events.append(self._stamp_event(
                    missile_event, physics_frame_index, frame_time))
                if missile_event.get("event_type") == "aircraft_death":
                    recorded_deaths.add(missile_event["agent_id"])
            self._record_new_deaths(
                events, recorded_deaths, physics_frame_index, frame_time)
            self.simulation_time_s += self.physics_dt

        self.step_count += 1
        reward_scores = self._selected_target_scores_at_end()
        target_used_by_reward = dict(self.current_targets)
        consistency_violations = self._target_consistency_violations(
            context, target_used_by_weapon, target_used_by_reward)
        self.target_consistency_violation_count += len(consistency_violations)
        if consistency_violations:
            raise RuntimeError(
                f"decision target snapshot mismatch: {consistency_violations}")
        terminated, truncated, winner, reason = self._termination()
        rewards, components = self.reward.compute(
            self.agents, self.current_targets, reward_scores,
            self.weapon.missiles, events, out_step, alive_at_start)
        for aid, value in rewards.items():
            self.episode_return[aid] += float(value)
        obs = self.observation.build(
            self.agents, self.weapon.missiles, self.current_targets)
        alive_at_end = {a.agent_id: a.alive for a in self.agents}
        just_died = {aid: alive_at_start[aid] and not alive_at_end[aid]
                     for aid in alive_at_start}
        info = self._build_info(events, components, opponent_diag, winner, reason,
                                alive_at_start, alive_at_end, just_died)
        info.update({
            "decision_context_id": context["decision_context_id"],
            "target_used_by_rule_action": dict(context["target_used_by_rule_action"]),
            "target_used_by_weapon": target_used_by_weapon,
            "target_used_by_reward": target_used_by_reward,
            "target_consistency_violation": consistency_violations,
            "target_consistency_violation_count": self.target_consistency_violation_count,
            "event_ordering_consistent": self._event_ordering_consistent(events),
            "death_reason": {a.agent_id: a.death_reason for a in self.agents},
            "crash_step": sorted(crash_step), "out_of_zone_step": sorted(out_step),
            # Deprecated compatibility field; structural exceedances are diagnostic only.
            "structural_failure_step": [],
            "low_altitude_violations": sorted(
                a.agent_id for a in self.agents if a.alive and
                a.position[2] < self.published["minimum_safe_altitude_m"]),
            "action_indices": {aid: value.tolist() for aid, value in action_indices.items()},
            "direct_fcs_commands": {aid: value.tolist() for aid, value in commands.items()},
        })
        self.last_info = info
        self._decision_context = None
        term = {a.agent_id: bool(terminated or not a.alive) for a in self.agents}
        trunc = {a.agent_id: bool(truncated) for a in self.agents}
        return obs, rewards, term, trunc, info

    def _stamp_event(self, event, physics_frame_index, simulation_time_s):
        self.event_sequence_counter += 1
        stamped = dict(event)
        stamped.update({
            "physics_frame_index": int(physics_frame_index),
            "simulation_time_s": float(simulation_time_s),
            "event_sequence_id": self.event_sequence_counter,
        })
        return stamped

    def _record_new_deaths(self, events, recorded_deaths, frame_index, event_time):
        for agent in self.agents:
            if agent.alive or agent.agent_id in recorded_deaths:
                continue
            recorded_deaths.add(agent.agent_id)
            events.append(self._stamp_event({
                "event_type": "aircraft_death",
                "agent_id": agent.agent_id,
                "side": agent.side,
                "reason": agent.death_reason,
            }, frame_index, event_time))

    @staticmethod
    def _event_ordering_consistent(events):
        sequence = [event["event_sequence_id"] for event in events]
        times = [event["simulation_time_s"] for event in events]
        return (sequence == sorted(sequence) and len(sequence) == len(set(sequence))
                and times == sorted(times))

    @staticmethod
    def _target_consistency_violations(context, weapon_targets, reward_targets):
        expected = context["targets"]
        violations = []
        for consumer, used in (
                ("rule_action", context["target_used_by_rule_action"]),
                ("weapon", weapon_targets), ("reward", reward_targets)):
            for aid, target_id in used.items():
                if target_id != expected.get(aid):
                    violations.append({
                        "agent_id": aid, "consumer": consumer,
                        "expected_target": expected.get(aid), "used_target": target_id,
                    })
        return violations

    @staticmethod
    def map_action(indices) -> np.ndarray:
        return map_action_indices(indices)

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
                for idx, entry in enumerate(self.initial_conditions["red_agents"])]

    def _update_targets(self):
        sides = {side: sorted([a for a in self.agents if a.side == side], key=lambda a: a.agent_id)
                 for side in ("red", "blue")}
        diagnostics, all_scores = {}, {}
        for ego in self.agents:
            previous = self.current_targets.get(ego.agent_id)
            enemies = sides["blue" if ego.side == "red" else "red"]
            scores = {}
            if ego.alive:
                for target in enemies:
                    if target.alive:
                        scores[target.agent_id] = assess_pair(
                            ego.position, ego.velocity, target.position, target.velocity,
                            self.published["maximum_attack_range_m"],
                            self.derived["situation_height_norm_m"],
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
                    self.derived["situation_height_norm_m"],
                    self.published["maximum_speed_mps"])
        return result

    def _apply_constraints_once(self):
        out_step, crash_step = set(), set()
        radius = float(self.derived["combat_zone_radius_m"])
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
                agent.speed_limit_exceedance_count += 1
            if abs(agent.load_factor_g) > float(self.published["maximum_aircraft_overload_g"]):
                agent.overload_limit_exceedance_count += 1
        return out_step, crash_step

    def _combat_units(self, side, alive_only=True):
        return [a for a in self.agents if a.side == side
                and a.aircraft_type.role != "mav" and (a.alive or not alive_only)]

    def _termination(self):
        red_alive = self._combat_units("red")
        blue_alive = self._combat_units("blue")
        if not red_alive and not blue_alive:
            return True, False, "draw", "mutual_combat_units_eliminated"
        if not blue_alive:
            return True, False, "red", "blue_combat_units_eliminated"
        if not red_alive:
            return True, False, "blue", "red_combat_units_eliminated"
        if self.step_count >= self.episode_limit:
            return False, True, "draw", "episode_limit"
        return False, False, None, None

    def _build_info(self, events, components, opponent_diag, winner, reason,
                    alive_start, alive_end, just_died):
        red = [a for a in self.agents if a.side == "red"]
        blue = [a for a in self.agents if a.side == "blue"]
        info = {
            "paper_environment_mode": "tam_paper_env_v1",
            "backend": self.scenario.dynamics_backend,
            "dynamics_backend": self.scenario.dynamics_backend,
            "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
            "experiment_protocol": self.experiment_protocol,
            "scenario": self.scenario_name,
            "initial_perturbation": self.initial_perturbation,
            "paper_nominal_experiment": self.experiment_protocol == PAPER_NOMINAL_PROTOCOL,
            "paper_generalization_experiment": (
                self.experiment_protocol == PAPER_5V4_GENERALIZATION_PROTOCOL),
            "paper_silent_assumptions_present": PAPER_SILENT_ASSUMPTIONS_PRESENT,
            "termination_resolution": TERMINATION_RESOLUTION,
            "neutral_action_semantics": "nearest_positive_center_not_exact_zero",
            "blue_policy_fidelity": BLUE_POLICY_FIDELITY,
            "reference_8_exact_blue_fsm_reproduced": (
                REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED),
            "basic_manoeuvre_action_mapping_unpublished": (
                BASIC_MANOEUVRE_ACTION_MAPPING_UNPUBLISHED),
            "height_reward_exact_formula_available": False,
            "height_reward_implementation": "unpublished_height_reward_approximation",
            "observation_visibility": "all_alive_fixed_slots_no_range_gate",
            "target_selection_tie_break": self.unpublished["target_selection_tie_break"],
            "execution_aircraft_model_provenance": "unpublished",
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
                "speed_limit_exceedance_count": a.speed_limit_exceedance_count,
                "overload_limit_exceedance_count": a.overload_limit_exceedance_count,
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
