from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from uav_env.JSBSim.paper.action_semantics import (
    ACTION_LEVELS, INACTIVE_ACTION_PLACEHOLDER, LOWER_CENTER_INDEX,
    LOWER_CENTER_VALUE, NEUTRAL_ACTION_SEMANTICS, UPPER_CENTER_INDEX,
    UPPER_CENTER_VALUE, map_action_indices)
from uav_env.JSBSim.paper.missile import PaperMissile, TIMEOUT_DERIVATION
from uav_env.JSBSim.paper.opponent import MANOEUVRES
from uav_env.JSBSim.paper.protocol import (
    MAX_INCOMING_MISSILES, derived_environment_values)
from uav_env.JSBSim.paper.reward import (
    paper_height_reward, scenario_derived_reward_cap)
from uav_env.JSBSim.paper.weapon import PaperWeaponManager
from uav_env.make_env import make_env


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "uav_env" / "JSBSim" / "configs" / "tam_paper_env_v1_3v2.yaml"


def env_for(name="tam_paper_env_v1_3v2.yaml"):
    return make_env(str(CONFIG.parent / name), dynamics_backend="simple")


def test_detection_ranges_and_perfect_mav_sharing():
    env = env_for()
    env.reset(seed=1)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    red_uav, mav, blue = by_id["red_1"], by_id["red_0"], by_id["blue_0"]
    blue.position = np.zeros(3)
    red_uav.position = np.array([14000.0, 0.0, 0.0])
    assert env.task.observation._visible(red_uav, blue, [])
    red_uav.position = np.array([14000.01, 0.0, 0.0])
    assert not env.task.observation._visible(red_uav, blue, [])
    mav.position = np.array([28000.0, 0.0, 0.0])
    assert env.task.observation._visible(mav, blue, [mav])
    mav.position = np.array([28000.01, 0.0, 0.0])
    assert not env.task.observation._visible(mav, blue, [mav])
    red_uav.position = np.array([20000.0, 0.0, 0.0])
    mav.position = np.array([10000.0, 0.0, 0.0])
    assert env.task.observation._visible(red_uav, blue, [mav])
    mav.kill("shotdown")
    assert not env.task.observation._visible(red_uav, blue, [])
    env.close()


def test_scenario_derived_slots_zone_and_normalization():
    values = derived_environment_values(14000.0)
    assert MAX_INCOMING_MISSILES == 4 * 2 == 8
    assert values["combat_zone_radius_m"] == 28000.0
    assert values["position_norm_m"] == 28000.0
    assert values["altitude_norm_m"] == 6000.0
    assert values["situation_height_norm_m"] == 6000.0
    env = env_for("tam_paper_env_v1_5v4.yaml")
    assert env.task.observation.max_incoming == 8
    assert env.task.observation.position_norm_m == 28000.0
    assert env.task.observation.altitude_norm_m == 6000.0
    env.close()


def test_height_simplification_is_continuous_finite_and_monotone():
    assert paper_height_reward(0.0, 750.0) == -1.0
    assert paper_height_reward(750.0, 750.0) == 0.0
    assert paper_height_reward(6000.0, 750.0) == 1.0
    assert paper_height_reward(9000.0, 750.0) == 1.0
    values = np.array([paper_height_reward(height, 750.0)
                       for height in np.linspace(0.0, 9000.0, 100)])
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) >= -1e-12)


def _hit(sequence, target_id="blue_0", shooter_id="red_1"):
    event = {"event_type": "missile_termination", "reason": "hit",
             "shooter_id": shooter_id, "target_id": target_id}
    if sequence is not None:
        event["event_sequence_id"] = sequence
    return event


def _reward_components(env, events, alive_start=None):
    alive = alive_start or {agent.agent_id: True for agent in env.task.agents}
    return env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], events, set(), alive)[1]


def test_mav_thresholds_and_scenario_derived_caps():
    assert scenario_derived_reward_cap(2) == 400.0  # 2v2 and 3v2 enemy set
    assert scenario_derived_reward_cap(4) == 800.0  # 5v4 enemy set
    env = env_for()
    env.reset(seed=2)
    assert env.task.reward.derived | {} == env.task.reward.derived
    assert env.task.reward.derived["mav_d_danger_m"] == 7000.0
    assert env.task.reward.derived["mav_d_safe_m"] == 14000.0
    assert env.task.reward.derived["mav_d_opt_m"] == 14000.0
    assert env.task.reward.derived["mav_d_max_m"] == 28000.0
    mav = next(agent for agent in env.task.agents if agent.aircraft_type.role == "mav")
    components = _reward_components(env, [_hit(1), _hit(2, "blue_1", "red_2")])
    comp = components[mav.agent_id]
    assert comp["r_team_kill_bonus_increment"] == 400.0
    assert comp["r_team_kill_bonus_cumulative"] == 400.0
    assert comp["r_team_kill_bonus_cap"] == 400.0
    assert comp["r_death"] == 0.0 and comp["r_event"] == 400.0
    env.close()


def test_mav_unique_target_credit_across_steps_reset_and_remaining_cap():
    env = env_for(); env.reset(seed=21)
    mav = next(agent for agent in env.task.agents if agent.aircraft_type.role == "mav")
    first = _reward_components(env, [_hit(1), _hit(2)])[mav.agent_id]
    repeat = _reward_components(env, [_hit(3)])[mav.agent_id]
    assert first["r_team_kill_bonus_increment"] == 200.0
    assert repeat["r_team_kill_bonus_increment"] == 0.0
    assert repeat["r_team_kill_bonus_cumulative"] == 200.0
    env.task.reward.cumulative_team_kill_bonus[mav.agent_id] = 300.0
    limited = _reward_components(env, [_hit(4, "blue_1", "red_2")])[mav.agent_id]
    assert limited["r_team_kill_bonus_increment"] == 100.0
    assert limited["r_team_kill_bonus_cumulative"] == 400.0
    env.task.reward.reset()
    reset = _reward_components(env, [_hit(5)])[mav.agent_id]
    assert reset["r_team_kill_bonus_increment"] == 200.0
    env.close()


def test_mav_hit_validation_death_order_and_uav_reward_unchanged():
    env = env_for(); env.reset(seed=22)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    mav = by_id["red_0"]
    malformed = [_hit(None), _hit(1, target_id=None),
                 _hit(2, target_id="red_2"),
                 _hit(3, shooter_id=None)]
    comp = _reward_components(env, malformed)[mav.agent_id]
    assert comp["r_team_kill_bonus_increment"] == 0.0

    env.task.reward.reset(); mav.kill("shotdown")
    death = {"event_type": "aircraft_death", "agent_id": mav.agent_id,
             "event_sequence_id": 2}
    before = _reward_components(env, [_hit(1), death])[mav.agent_id]
    assert before["r_team_kill_bonus_increment"] == 200.0
    assert before["r_death"] == -200.0 and before["r_event"] == 0.0

    env.task.reward.reset()
    after = _reward_components(env, [death, _hit(3)])[mav.agent_id]
    assert after["r_team_kill_bonus_increment"] == 0.0
    assert after["r_death"] == -200.0 and after["r_event"] == -200.0

    env.task.reward.reset(); mav.alive = True; mav.death_reason = None
    uav_components = _reward_components(env, [_hit(1)])["red_1"]
    assert uav_components["r_event"] == 200.0
    env.close()


def test_exactly_seven_atomic_manoeuvres_all_scored_and_tie_breaks_first(monkeypatch):
    assert list(MANOEUVRES) == [
        "level", "accelerate", "decelerate", "left_turn", "right_turn", "climb", "dive"]
    assert all(np.asarray(action).shape == (4,) for action in MANOEUVRES.values())
    env = env_for()
    _, info = env.reset(seed=3)
    assert info["blue_policy_fidelity"] == (
        "minimal_greedy_basic_manoeuvre_reconstruction")
    assert info["reference_8_exact_blue_fsm_reproduced"] is False
    assert info["neutral_action_semantics"] == NEUTRAL_ACTION_SEMANTICS
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    agent, target = by_id["blue_0"], by_id["red_1"]
    fixed = (agent.position.copy(), agent.velocity.copy(), agent.pitch,
             agent.heading, agent.speed)
    monkeypatch.setattr(env.task.opponent, "_predict", lambda _agent, _indices: fixed)
    action, diagnostics = env.task.opponent.act(agent, target, [])
    assert set(diagnostics["candidates"]) == set(MANOEUVRES)
    assert diagnostics["manoeuvre"] == "level"
    assert action.tolist() == list(MANOEUVRES["level"])
    env.close()


def test_weapon_signature_and_missile_timeout_are_simplified():
    assert list(inspect.signature(PaperWeaponManager.try_launch).parameters) == [
        "self", "shooter", "target", "simulation_time_s"]
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inferred = config["inferred_parameters"]
    assert "maximum_missile_speed_mps" not in inferred
    assert "missile_max_flight_time_s" not in inferred
    missile = PaperMissile(
        "m", "red_1", "blue_0", np.zeros(3), np.array([1.0, 0.0, 0.0]),
        config["published_parameters"] | inferred)
    assert missile.timeout_s == pytest.approx(56.0)
    assert missile.timeout_derivation == TIMEOUT_DERIVATION
    for _ in range(600):
        missile.step(np.array([1e6, 1e6, 6000.0]), np.zeros(3), 1.0 / 60.0)
        assert np.isfinite(np.concatenate(
            [missile.position, missile.velocity, [missile.speed_mps]])).all()


def test_complete_60hz_missile_timeout_lifecycle_and_single_report():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manager = PaperWeaponManager(
        config["published_parameters"], config["inferred_parameters"])
    missile = PaperMissile(
        "lifecycle", "red_1", "blue_0", np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        config["published_parameters"] | config["inferred_parameters"])
    manager.missiles.append(missile)
    target = SimpleNamespace(
        agent_id="blue_0", side="blue", alive=True,
        position=np.array([1.0e8, 1.0e8, 6000.0]), velocity=np.zeros(3))
    termination_events = []
    dt = 1.0 / 60.0
    while missile.alive:
        assert missile.termination_reason != "timeout"
        termination_events.extend(manager.step_physics_once({"blue_0": target}, dt))
        state = np.concatenate([
            missile.position, missile.velocity,
            [missile.speed_mps, missile.yaw_rad, missile.pitch_rad,
             missile.longitudinal_overload_g, missile.lateral_overload_y_g,
             missile.lateral_overload_z_g, missile.commanded_overload_g]])
        assert np.isfinite(state).all()
    assert missile.termination_reason == "timeout"
    assert missile.flight_time_s == pytest.approx(56.0, abs=dt)
    telemetry = missile.telemetry()
    assert telemetry["derived_timeout_s"] == pytest.approx(56.0)
    assert telemetry["timeout_derivation"] == TIMEOUT_DERIVATION
    assert [event["reason"] for event in termination_events] == ["timeout"]
    frozen = (missile.position.copy(), missile.velocity.copy(), missile.flight_time_s)
    assert manager.step_physics_once({"blue_0": target}, dt) == []
    assert missile.step(target.position, target.velocity, dt) == "timeout"
    np.testing.assert_array_equal(missile.position, frozen[0])
    np.testing.assert_array_equal(missile.velocity, frozen[1])
    assert missile.flight_time_s == frozen[2]


def test_40_level_action_grid_has_no_exact_zero_and_placeholder_is_inactive():
    assert ACTION_LEVELS == 40
    assert map_action_indices([0, 0, 0, 0]).tolist() == pytest.approx(
        [0.4, -1.0, -1.0, -1.0])
    assert map_action_indices([39, 39, 39, 39]).tolist() == pytest.approx(
        [0.9, 1.0, 1.0, 1.0])
    assert map_action_indices([0, LOWER_CENTER_INDEX] * 2)[1] == pytest.approx(
        LOWER_CENTER_VALUE)
    assert map_action_indices([0, UPPER_CENTER_INDEX] * 2)[1] == pytest.approx(
        UPPER_CENTER_VALUE)
    surface_levels = [map_action_indices([0, index, index, index])[1]
                      for index in range(ACTION_LEVELS)]
    assert not any(value == 0.0 for value in surface_levels)
    assert NEUTRAL_ACTION_SEMANTICS == "nearest_positive_center_not_exact_zero"
    assert np.asarray(MANOEUVRES["level"]).shape == (4,)

    env = env_for(); env.reset(seed=23)
    dead = next(agent for agent in env.task.agents if agent.agent_id == "red_1")
    dead.kill("shotdown")
    dead.apply_direct_fcs_command = lambda _command: pytest.fail(
        "inactive placeholder reached dead-aircraft dynamics")
    _, _, _, _, info = env.step({
        aid: np.asarray(INACTIVE_ACTION_PLACEHOLDER) for aid in env.agent_ids})
    assert info["action_indices"][dead.agent_id] == list(INACTIVE_ACTION_PLACEHOLDER)
    assert env.get_avail_actions()[dead.agent_id][:, UPPER_CENTER_INDEX].sum() == 4
    env.close()


def test_removed_free_parameters_are_absent_from_formal_configs():
    removed = {
        "combat_zone_radius_m", "uav_direct_detection_range_m", "mav_detection_range_m",
        "position_norm_m", "altitude_norm_m", "situation_height_norm_m",
        "max_incoming_missiles", "optimal_altitude_m", "maximum_altitude_m",
        "mav_d_danger_m", "mav_d_safe_m", "mav_d_opt_m", "mav_d_max_m",
        "mav_death_penalty", "mav_team_kill_bonus", "mav_team_kill_bonus_cap",
        "maximum_missile_speed_mps", "missile_max_flight_time_s",
    }
    for path in CONFIG.parent.glob("tam_paper_env_v1_*.yaml"):
        inferred = yaml.safe_load(path.read_text(encoding="utf-8"))["inferred_parameters"]
        assert removed.isdisjoint(inferred)


def test_fidelity_document_records_timeout_and_reference_8_evidence_boundary():
    text = (ROOT / "docs" / "tam_paper_environment_fidelity.md").read_text(
        encoding="utf-8")
    assert "PAPER_SILENT_DERIVED_SIMPLIFICATION" in text
    assert "2_times_attack_range_over_inferred_initial_speed" in text
    assert "10.1016/j.dt.2022.08.010" in text
    assert "REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED = false" in text
    assert "PAPER_ENVIRONMENT_EXACTLY_REPRODUCED = false" in text
