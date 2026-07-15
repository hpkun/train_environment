from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.make_env import make_env


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"


def env_for(backend="simple"):
    return make_env(str(CONFIG_DIR / "tam_paper_env_v1_3v2.yaml"),
                    dynamics_backend=backend)


def test_prepared_context_switches_and_all_consumers_share_snapshot(monkeypatch):
    env = env_for()
    env.reset(seed=100)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    ego, target_a, target_b = by_id["red_1"], by_id["blue_0"], by_id["blue_1"]
    env.task.current_targets[ego.agent_id] = target_a.agent_id
    target_a.position = ego.position + np.array([-15000.0, 0.0, 0.0])
    target_b.position = ego.position + np.array([5000.0, 0.0, 1000.0])

    context = env.prepare_decision_context()
    assert context["targets"][ego.agent_id] == target_b.agent_id
    assert env.prepare_decision_context() is context
    actions = env.build_rule_actions(env.agent_ids)
    context_id = context["decision_context_id"]

    _, _, _, _, info = env.step(actions)
    assert info["decision_context_id"] == context_id
    assert info["target_used_by_rule_action"][ego.agent_id] == target_b.agent_id
    assert info["target_used_by_weapon"][ego.agent_id] == target_b.agent_id
    assert info["target_used_by_reward"][ego.agent_id] == target_b.agent_id
    for blue_id, diagnostic in info["opponent_diagnostics"].items():
        assert diagnostic["current_target"] == context["targets"][blue_id]
    assert info["target_consistency_violation"] == []
    env.close()


def test_opponent_scores_target_and_threat_at_same_prediction_time():
    env = env_for()
    env.reset(seed=101)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    agent, target = by_id["blue_0"], by_id["red_1"]
    target.position = agent.position + np.array([6000.0, 2000.0, 500.0])
    target.velocity = np.array([0.0, 350.0, 0.0])
    _, moving = env.task.opponent.act(agent, target, [])
    target.velocity = np.zeros(3)
    _, stationary = env.task.opponent.act(agent, target, [])

    moving_totals = [row["total_dense_reward"]
                     for row in moving["candidates"].values()]
    stationary_totals = [row["total_dense_reward"]
                         for row in stationary["candidates"].values()]
    assert not np.allclose(moving_totals, stationary_totals)
    for row in moving["candidates"].values():
        assert row["prediction_time_s"] == pytest.approx(0.2)
        assert row["threat_prediction"] == "constant_velocity"
        assert row["predicted_target_position_m"] == pytest.approx(
            (target.position + np.array([0.0, 350.0, 0.0]) * 0.2).tolist())
    assert moving["candidates"][moving["manoeuvre"]]["total_dense_reward"] == max(
        moving_totals)
    env.close()


@pytest.mark.parametrize(("hit_sequence", "death_sequence", "expected_event"), [
    (1, 2, 0.0),
    (2, 1, -200.0),
])
def test_mav_team_credit_respects_opposite_event_orders(
        hit_sequence, death_sequence, expected_event):
    env = env_for()
    env.reset(seed=102)
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    mav = by_id["red_0"]
    mav.kill("shotdown")
    events = [
        {"event_type": "missile_termination", "reason": "hit",
         "shooter_id": "red_1", "target_id": "blue_0",
         "event_sequence_id": hit_sequence, "physics_frame_index": 3,
         "simulation_time_s": 0.05},
        {"event_type": "aircraft_death", "reason": "shotdown",
         "agent_id": mav.agent_id, "side": "red",
         "event_sequence_id": death_sequence, "physics_frame_index": 3,
         "simulation_time_s": 0.05},
    ]
    events.sort(key=lambda event: event["event_sequence_id"])
    _, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], events, set(), {agent.agent_id: True for agent in env.task.agents})
    assert components[mav.agent_id]["r_event"] == pytest.approx(expected_event)
    env.close()


def test_aircraft_death_event_has_physics_order_metadata():
    env = env_for()
    env.reset(seed=103)
    agent = next(item for item in env.task.agents if item.agent_id == "red_1")
    agent.position[0] = env.task.inferred["combat_zone_radius_m"] + 1000.0
    _, _, _, _, info = env.step(
        {aid: np.array([20, 20, 20, 20]) for aid in env.agent_ids})
    event = next(item for item in info["missile_events"]
                 if item.get("event_type") == "aircraft_death"
                 and item["agent_id"] == agent.agent_id)
    assert event["physics_frame_index"] == 0
    assert event["simulation_time_s"] == pytest.approx(1 / 60)
    assert event["event_sequence_id"] > 0
    assert info["event_ordering_consistent"] is True
    env.close()


def test_published_missile_metadata_and_explicit_drag_units():
    for path in CONFIG_DIR.glob("tam_paper_env_v1_*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        published = config["published_parameters"]
        inferred = config["inferred_parameters"]
        assert published["missile_mass_kg"] == 84.0
        assert published["missile_length_m"] == 2.87
        assert published["missile_diameter_m"] == 0.127
        assert "drag_coefficient" not in inferred
        assert inferred["effective_quadratic_drag_per_m"] == 0.00012


def test_real_jsbsim_initial_controls_and_command_only_throttle_path(monkeypatch):
    env = env_for("jsbsim")
    env.reset(seed=104)
    for aircraft in env.task.agents[:2]:
        status = aircraft.control_initialization_status()
        assert status["gear_command_norm"] == pytest.approx(0.0)
        assert status["gear_position_norm"] == pytest.approx(0.0)
        assert status["flap_command_norm"] == pytest.approx(0.0)
        assert status["flap_position_norm"] == pytest.approx(0.0)
        assert all(status["engine_running"])
        assert status["throttle_adapter"] == "command_only_no_adapter"
        written = []
        original = aircraft._set_property
        monkeypatch.setattr(aircraft, "_set_property",
                            lambda name, value, fn=original:
                            (written.append(name), fn(name, value))[1])
        aircraft.apply_direct_fcs_command(np.array([0.65, 0.0, 0.0, 0.0]))
        assert "fcs/throttle-cmd-norm" in written
        assert "fcs/throttle-pos-norm" not in written
    env.close()


@pytest.mark.parametrize("agent_id", ["red_0", "red_1"])
def test_real_f22_f16_command_only_throttle_has_monotonic_response(agent_id):
    results = []
    for command in (0.4, 0.65, 0.9):
        env = env_for("jsbsim")
        env.reset(seed=105)
        aircraft = next(item for item in env.task.agents if item.agent_id == agent_id)
        for _ in range(300):
            aircraft.apply_direct_fcs_command(
                np.array([command, 0.0, 0.0, 0.0]))
            assert aircraft.step_physics_once(1 / 60)
        results.append((
            aircraft._get_property("fcs/throttle-pos-norm"),
            aircraft._get_property("propulsion/engine[0]/thrust-lbs"),
            aircraft.speed,
        ))
        env.close()
    for column in range(3):
        values = [result[column] for result in results]
        assert values[0] < values[1] < values[2]
