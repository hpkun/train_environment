from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.JSBSim.paper.missile import G, PaperMissile, analytic_los_rates
from uav_env.JSBSim.paper.opponent import GreedyPaperOpponent
from uav_env.JSBSim.paper.situation import assess_pair
from uav_env.make_env import make_env


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"


def env_for(name="tam_paper_env_v1_3v2.yaml", backend="simple"):
    return make_env(str(CONFIG_DIR / name), dynamics_backend=backend)


def missile_config():
    env = env_for()
    config = {**env.task.published, **env.task.inferred}
    env.close()
    return config


def test_formal_config_has_single_frequency_source_and_no_trim_or_warmup():
    cfg = yaml.safe_load((CONFIG_DIR / "tam_paper_env_v1_3v2.yaml").read_text())
    assert "simulation_frequency" not in cfg
    assert "direct_fcs_static_trim_by_model" not in cfg["inferred_parameters"]
    assert "reset_warmup_seconds" not in cfg["inferred_parameters"]
    assert "reset_warmup_throttle" not in cfg["inferred_parameters"]
    assert all("model_path" not in value and "radar_range" not in value
               for value in cfg["aircraft_type_params"].values())
    assert (cfg["published_parameters"]["decision_frequency_hz"]
            == cfg["published_parameters"]["simulation_frequency_hz"]
            / cfg["published_parameters"]["physics_frames_per_action"])


def test_analytic_los_rates_match_hand_geometry():
    r = np.array([1000.0, 200.0, 300.0])
    rd = np.array([-100.0, 40.0, -20.0])
    _, _, beta_dot, epsilon_dot = analytic_los_rates(r, rd)
    rho = np.hypot(r[0], r[1])
    expected_beta = (r[0] * rd[1] - r[1] * rd[0]) / rho**2
    rho_dot = (r[0] * rd[0] + r[1] * rd[1]) / rho
    expected_epsilon = (rho * rd[2] - r[2] * rho_dot) / np.dot(r, r)
    assert beta_dot == pytest.approx(expected_beta)
    assert epsilon_dot == pytest.approx(expected_epsilon)


def test_pn_uses_target_velocity_and_paper_cos_terms():
    cfg = missile_config()
    pitch = np.deg2rad(70.0)
    direction = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
    stationary = PaperMissile("a", "s", "t", np.zeros(3), direction, cfg)
    moving = PaperMissile("b", "s", "t", np.zeros(3), direction, cfg)
    target = np.array([8000.0, 1000.0, 2000.0])
    stationary.step(target, np.zeros(3), 1 / 60)
    moving.step(target, np.array([0.0, 200.0, -50.0]), 1 / 60)
    assert moving.los_yaw_rate_rps != pytest.approx(stationary.los_yaw_rate_rps)
    expected_ny = (3.0 * cfg["missile_initial_speed_mps"] / G
                   * np.cos(pitch) * moving.los_yaw_rate_rps)
    assert moving.lateral_overload_y_g == pytest.approx(expected_ny)
    assert moving.commanded_overload_g <= 30.0 + 1e-9


def test_unified_loop_interleaves_all_aircraft_and_missile_frames(monkeypatch):
    env = env_for("tam_paper_env_v1_2v2.yaml")
    env.reset(seed=1)
    order = []
    for aircraft in env.task.agents:
        original = aircraft.step_physics_once
        monkeypatch.setattr(aircraft, "step_physics_once",
                            lambda dt, aid=aircraft.agent_id, fn=original:
                            (order.append(("aircraft", aid)), fn(dt))[1])
    original_weapon = env.task.weapon.step_physics_once
    monkeypatch.setattr(env.task.weapon, "step_physics_once",
                        lambda by_id, dt: (order.append(("missile", None)),
                                           original_weapon(by_id, dt))[1])
    env.step({aid: np.array([20, 20, 20, 20]) for aid in env.agent_ids})
    assert sum(kind == "missile" for kind, _ in order) == 12
    frame_width = len(env.task.agents) + 1
    for frame in range(12):
        chunk = order[frame * frame_width:(frame + 1) * frame_width]
        assert [kind for kind, _ in chunk] == ["aircraft"] * len(env.task.agents) + ["missile"]
    assert env.task.simulation_time_s == pytest.approx(0.2)
    env.close()


def test_every_decision_reselects_best_target_and_records_change():
    env = env_for()
    env.reset(seed=2)
    by_id = {a.agent_id: a for a in env.task.agents}
    ego, a, b = by_id["red_1"], by_id["blue_0"], by_id["blue_1"]
    a.position = ego.position + np.array([5000.0, 0.0, 0.0])
    b.position = ego.position + np.array([-15000.0, 0.0, 0.0])
    obs = env.task.observation.build(env.task.agents, [])
    env.task._update_targets(obs)
    assert env.task.current_targets[ego.agent_id] == a.agent_id
    a.position = ego.position + np.array([-15000.0, 0.0, 0.0])
    b.position = ego.position + np.array([5000.0, 0.0, 1000.0])
    obs = env.task.observation.build(env.task.agents, [])
    env.task._update_targets(obs)
    assert env.task.current_targets[ego.agent_id] == b.agent_id
    assert env.task.target_diagnostics[ego.agent_id]["target_changed"] is True
    env.close()


def test_dead_agents_have_zero_reward_components_for_ten_steps():
    env = env_for()
    obs, _ = env.reset(seed=3)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["red_1"].kill("shotdown")
    for _ in range(10):
        obs, rewards, _, _, info = env.step(
            {aid: np.array([20, 20, 20, 20]) for aid in env.agent_ids})
        assert rewards["red_1"] == 0.0
        assert all(value == 0.0 for value in info["red_1"]["reward_components"].values())
        assert np.count_nonzero(env.flatten_observation(obs["red_1"])) == 0
        assert env.get_avail_actions()["red_1"][:, 20].sum() == 4
    by_id["red_0"].kill("shotdown")
    for _ in range(10):
        _, rewards, _, _, info = env.step(
            {aid: np.array([20, 20, 20, 20]) for aid in env.agent_ids})
        assert rewards["red_0"] == 0.0
        assert all(value == 0.0 for value in info["red_0"]["reward_components"].values())
    env.close()


def test_mav_dead_before_team_kill_gets_no_bonus():
    env = env_for()
    env.reset(seed=4)
    by_id = {a.agent_id: a for a in env.task.agents}
    mav = by_id["red_0"]
    mav.kill("shotdown")
    event = {"reason": "hit", "shooter_id": "red_1", "target_id": "blue_0"}
    rewards, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], [event], set(), {a.agent_id: a.agent_id != mav.agent_id for a in env.task.agents})
    assert rewards[mav.agent_id] == 0.0
    assert components[mav.agent_id]["r_event"] == 0.0
    env.close()


@pytest.mark.parametrize(("reason", "expected"), [
    ("boundary", -100.0), ("crash", -200.0), ("shotdown", -200.0),
])
def test_uav_death_event_is_paid_once(reason, expected):
    env = env_for()
    env.reset(seed=42)
    agent = next(a for a in env.task.agents if a.agent_id == "red_1")
    alive = {a.agent_id: True for a in env.task.agents}
    agent.kill(reason)
    out = {agent.agent_id} if reason == "boundary" else set()
    _, first = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], [], out, alive)
    dead_at_start = dict(alive)
    dead_at_start[agent.agent_id] = False
    reward, second = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], [], set(), dead_at_start)
    assert first[agent.agent_id]["r_event"] == expected
    assert reward[agent.agent_id] == 0.0
    assert all(value == 0.0 for value in second[agent.agent_id].values())
    env.close()


def test_dodge_speed_reward_uses_whole_decision_span():
    env = env_for()
    env.reset(seed=41)
    by_id = {a.agent_id: a for a in env.task.agents}
    agent = by_id["red_1"]
    missile = PaperMissile("m", "blue_0", agent.agent_id,
                           agent.position - np.array([1000.0, 0.0, 0.0]),
                           np.array([1.0, 0.0, 0.0]), missile_config())
    missile.decision_start_speed_mps = 500.0
    missile.previous_speed_mps = 451.0
    missile.speed_mps = 450.0
    rewards, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [missile], [], set(), {a.agent_id: True for a in env.task.agents})
    assert components[agent.agent_id]["r_dodge_speed"] == pytest.approx(0.05)
    assert np.isfinite(rewards[agent.agent_id])
    env.close()


def test_structural_limits_do_not_rewrite_velocity_and_fail_after_grace():
    env = env_for()
    env.reset(seed=5)
    aircraft = env.task.agents[1]
    aircraft.velocity = np.array([450.0, 0.0, 0.0])
    original = aircraft.velocity.copy()
    for _ in range(12):
        env.task._apply_constraints_once()
    assert np.array_equal(aircraft.velocity, original)
    assert not aircraft.alive
    assert aircraft.death_reason == "structural_speed"
    env.close()


def test_overload_limit_uses_grace_and_records_structural_reason():
    env = env_for()
    env.reset(seed=51)
    aircraft = env.task.agents[1]
    env.task.published["maximum_aircraft_overload_g"] = 0.5
    for _ in range(12):
        env.task._apply_constraints_once()
    assert not aircraft.alive
    assert aircraft.death_reason == "structural_overload"
    assert aircraft.overload_violation_count == 12
    env.close()


def test_overload_limit_uses_absolute_nz(monkeypatch):
    env = env_for()
    env.reset(seed=511)
    aircraft = env.task.agents[1]
    monkeypatch.setattr(type(aircraft), "load_factor_g", property(lambda self: -10.0))
    for _ in range(12):
        env.task._apply_constraints_once()
    assert not aircraft.alive
    assert aircraft.death_reason == "structural_overload"
    env.close()


def test_reselected_target_is_shared_by_launch_and_reward(monkeypatch):
    env = env_for()
    env.reset(seed=512)
    by_id = {a.agent_id: a for a in env.task.agents}
    ego, old, new = by_id["red_1"], by_id["blue_0"], by_id["blue_1"]
    old.position = ego.position + np.array([-15000.0, 0.0, 0.0])
    new.position = ego.position + np.array([5000.0, 0.0, 1000.0])
    launched = {}
    original_launch = env.task.weapon.try_launch

    def capture_launch(shooter, target, visible, simulation_time_s):
        if shooter.agent_id == ego.agent_id:
            launched[shooter.agent_id] = target.agent_id
        return original_launch(shooter, target, visible, simulation_time_s)

    monkeypatch.setattr(env.task.weapon, "try_launch", capture_launch)
    _, _, _, _, info = env.step(
        {aid: np.array([20, 20, 20, 20]) for aid in env.agent_ids})
    assert info["current_targets"][ego.agent_id] == new.agent_id
    assert launched[ego.agent_id] == new.agent_id
    assert set(env.task._selected_target_scores_at_end()[ego.agent_id]) == {new.agent_id}
    env.close()


def test_opponent_argmax_uses_formal_reward_and_distinguishes_left_right():
    env = env_for()
    env.reset(seed=52)
    by_id = {a.agent_id: a for a in env.task.agents}
    agent, target = by_id["blue_0"], by_id["red_1"]
    target.position = agent.position + np.array([5000.0, 5000.0, 0.0])
    action_left, diag_left = env.task.opponent.act(agent, target, [])
    target.position = agent.position + np.array([5000.0, -5000.0, 0.0])
    action_right, diag_right = env.task.opponent.act(agent, target, [])
    assert "left" in diag_left["manoeuvre"]
    assert "right" in diag_right["manoeuvre"]
    assert not np.array_equal(action_left, action_right)
    for diag in (diag_left, diag_right):
        totals = {name: row["total_dense_reward"] for name, row in diag["candidates"].items()}
        assert totals[diag["manoeuvre"]] == max(totals.values())
        assert set(next(iter(diag["candidates"].values()))["reward_components"]) == {
            "r_height", "r_speed", "r_angle", "r_distance", "r_dodge"
        }
    env.close()


def test_combat_unit_termination_ignores_mav_count():
    env = env_for()
    env.reset(seed=6)
    env.task.step_count = 1000
    assert env.task._termination() == (False, True, "draw", "episode_limit")
    red_uavs = [a for a in env.task.agents if a.side == "red" and a.aircraft_type.role != "mav"]
    for aircraft in red_uavs:
        aircraft.kill("shotdown")
    assert env.task._termination() == (True, False, "blue", "red_combat_units_eliminated")
    env.close()


def test_mav_loss_does_not_terminate_but_blue_elimination_does():
    env = env_for()
    env.reset(seed=7)
    mav = next(a for a in env.task.agents if a.aircraft_type.role == "mav")
    mav.kill("shotdown")
    assert env.task._termination() == (False, False, None, None)
    for aircraft in [a for a in env.task.agents if a.side == "blue"]:
        aircraft.kill("shotdown")
    assert env.task._termination() == (True, False, "red", "blue_combat_units_eliminated")
    env.close()


@pytest.mark.parametrize("model_agent", ["red_0", "red_1"])
def test_real_jsbsim_reset_and_50_near_neutral_frames(model_agent):
    env = env_for(backend="jsbsim")
    env.reset(seed=8)
    aircraft = next(a for a in env.task.agents if a.agent_id == model_agent)
    assert np.linalg.norm(aircraft.position - aircraft._initial_position) < 1e-5
    assert aircraft.speed == pytest.approx(np.linalg.norm(aircraft._initial_velocity), abs=1e-6)
    heading_error = (aircraft.heading - aircraft._initial_heading + np.pi) % (2 * np.pi) - np.pi
    assert heading_error == pytest.approx(0.0, abs=1e-8)
    start = aircraft.position.copy()
    command = env.map_action([24, 20, 20, 20])
    for _ in range(50):
        aircraft.apply_direct_fcs_command(command)
        assert aircraft.step_physics_once(1 / 60)
    assert aircraft.alive
    assert np.isfinite(np.concatenate([aircraft.position, aircraft.velocity])).all()
    assert np.linalg.norm(aircraft.position - start) < 10000.0
    env.close()


def test_real_jsbsim_fcs_receives_untrimmed_mapped_command():
    env = env_for("tam_paper_env_v1_2v2.yaml", backend="jsbsim")
    env.reset(seed=9)
    aircraft = env.task.agents[0]
    command = env.map_action([39, 39, 39, 39])
    aircraft.apply_direct_fcs_command(command)
    assert aircraft._get_property("fcs/throttle-cmd-norm") == pytest.approx(0.9)
    assert aircraft._get_property("fcs/aileron-cmd-norm") == pytest.approx(1.0)
    assert aircraft._get_property("fcs/elevator-cmd-norm") == pytest.approx(1.0)
    assert aircraft._get_property("fcs/rudder-cmd-norm") == pytest.approx(1.0)
    env.close()
