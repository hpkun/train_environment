from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.JSBSim.paper.missile import PaperMissile
from uav_env.JSBSim.paper.opponent import MANOEUVRES
from uav_env.JSBSim.paper.protocol import (
    MAX_INCOMING_MISSILES, derived_environment_values)
from uav_env.JSBSim.paper.reward import paper_height_reward
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


def test_mav_thresholds_and_uncapped_multi_kill_event_scale():
    env = env_for()
    env.reset(seed=2)
    assert env.task.reward.derived | {} == env.task.reward.derived
    assert env.task.reward.derived["mav_d_danger_m"] == 7000.0
    assert env.task.reward.derived["mav_d_safe_m"] == 14000.0
    assert env.task.reward.derived["mav_d_opt_m"] == 14000.0
    assert env.task.reward.derived["mav_d_max_m"] == 28000.0
    assert not hasattr(env.task.reward, "_team_kill_bonus_paid")
    mav = next(agent for agent in env.task.agents if agent.aircraft_type.role == "mav")
    events = [
        {"reason": "hit", "shooter_id": "red_1", "target_id": "blue_0",
         "event_sequence_id": 1},
        {"reason": "hit", "shooter_id": "red_2", "target_id": "blue_1",
         "event_sequence_id": 2},
    ]
    _, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], events, set(), {agent.agent_id: True for agent in env.task.agents})
    assert components[mav.agent_id]["r_event"] == 400.0
    mav.kill("shotdown")
    events.append({"event_type": "aircraft_death", "agent_id": mav.agent_id,
                   "event_sequence_id": 3})
    _, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], events, set(), {agent.agent_id: True for agent in env.task.agents})
    assert components[mav.agent_id]["r_event"] == 200.0
    env.close()


def test_exactly_seven_atomic_manoeuvres_all_scored_and_tie_breaks_first(monkeypatch):
    assert list(MANOEUVRES) == [
        "level", "accelerate", "decelerate", "left_turn", "right_turn", "climb", "dive"]
    assert all(np.asarray(action).shape == (4,) for action in MANOEUVRES.values())
    env = env_for()
    env.reset(seed=3)
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
    for _ in range(600):
        missile.step(np.array([1e6, 1e6, 6000.0]), np.zeros(3), 1.0 / 60.0)
        assert np.isfinite(np.concatenate(
            [missile.position, missile.velocity, [missile.speed_mps]])).all()


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
