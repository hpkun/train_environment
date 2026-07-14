from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.JSBSim.paper.missile import PaperMissile
from uav_env.JSBSim.paper.reward import (
    paper_height_reward, uav_angle_reward, uav_distance_reward, uav_speed_reward,
)
from uav_env.JSBSim.paper.situation import assess_pair, select_best_target
from uav_env.make_env import make_env


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"
PAPER_CONFIGS = {
    "tam_paper_env_v1_2v2.yaml": (2, 2, 0),
    "tam_paper_env_v1_3v2.yaml": (3, 2, 1),
    "tam_paper_env_v1_5v4.yaml": (5, 4, 1),
}


@pytest.mark.parametrize(("name", "red", "blue", "mavs"), [
    (name, *counts) for name, counts in PAPER_CONFIGS.items()
])
def test_paper_config_and_environment_contract(name, red, blue, mavs):
    path = CONFIG_DIR / name
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["env_type"] == "jsbsim_tam_paper"
    assert cfg["paper_environment_mode"] == "tam_paper_env_v1"
    assert cfg["published_parameters"]["simulation_frequency_hz"] == 60
    assert cfg["published_parameters"]["physics_frames_per_action"] == 12
    assert cfg["published_parameters"]["episode_limit_steps"] == 1000
    assert cfg["published_parameters"]["maximum_attack_range_m"] == 14000
    assert cfg["published_parameters"]["launch_interval_s"] == 25
    assert len(cfg["red_agents"]) == red
    assert len(cfg["blue_agents"]) == blue
    assert sum(a["role"] == "mav" for a in cfg["red_agents"]) == mavs

    env = make_env(str(path), dynamics_backend="simple")
    obs, info = env.reset(seed=7)
    assert env.action_space[env.agent_ids[0]].nvec.tolist() == [40, 40, 40, 40]
    assert info["paper_environment_mode"] == "tam_paper_env_v1"
    for aid in env.agent_ids:
        item = obs[aid]
        assert item["ego_state"].shape == (7,)
        assert item["ally_states"].shape[-1] == 5
        assert item["enemy_states"].shape[-1] == 5
        assert item["incoming_missile_states"].shape[-1] == 5
        assert np.isfinite(env.flatten_observation(item)).all()
    assert np.isfinite(env.get_state()).all()
    env.close()


def test_action_mapping_is_direct_and_exact():
    env = make_env(str(CONFIG_DIR / "tam_paper_env_v1_3v2.yaml"), dynamics_backend="simple")
    assert env.map_action([0, 0, 0, 0]) == pytest.approx([0.4, -1.0, -1.0, -1.0])
    assert env.map_action([39, 39, 39, 39]) == pytest.approx([0.9, 1.0, 1.0, 1.0])
    mid = env.map_action([20, 20, 20, 20])
    assert mid[0] == pytest.approx(0.4 + 20.0 / 39.0 * 0.5)
    assert mid[1:] == pytest.approx([-1.0 + 40.0 / 39.0] * 3)
    env.close()


def test_situation_weights_and_target_selection():
    ego_pos = np.array([0.0, 0.0, 6000.0])
    ego_vel = np.array([250.0, 0.0, 0.0])
    ahead = assess_pair(ego_pos, ego_vel, np.array([10000.0, 0.0, 6000.0]),
                        np.array([250.0, 0.0, 0.0]), 14000.0, 12000.0, 400.0)
    behind = assess_pair(ego_pos, ego_vel, np.array([-10000.0, 0.0, 6000.0]),
                         np.array([250.0, 0.0, 0.0]), 14000.0, 12000.0, 400.0)
    assert ahead.score == pytest.approx(
        0.35 * ahead.e_angle + 0.25 * ahead.e_distance
        + 0.20 * ahead.e_height + 0.20 * ahead.e_speed)
    assert select_best_target({"ahead": ahead, "behind": behind}) == "ahead"


def test_paper_distance_reward_boundaries():
    assert uav_distance_reward(5000.0) == pytest.approx(1.0)
    assert uav_distance_reward(5000.1) < 1.0
    assert uav_distance_reward(9999.9) > 0.0
    assert uav_distance_reward(10000.0) == pytest.approx(-1.0)


def test_uav_speed_angle_and_height_reward_boundaries():
    assert uav_speed_reward(200.0, 99.0) == pytest.approx(1.0)
    assert uav_speed_reward(200.0, 200.0) == pytest.approx(0.0)
    assert uav_speed_reward(200.0, 301.0) == pytest.approx(-1.0)
    assert uav_angle_reward(0.0, 0.0) == pytest.approx(1.0)
    assert uav_angle_reward(np.pi, 0.0) == pytest.approx(0.0)
    assert paper_height_reward(750.0, 750.0, 6000.0, 12000.0) == pytest.approx(-1.0)
    assert paper_height_reward(6000.0, 750.0, 6000.0, 12000.0) == pytest.approx(1.0)


def test_point_mass_missile_has_pn_limits_and_variable_speed():
    cfg = {
        "simulation_frequency_hz": 60,
        "navigation_gain_y": 3.0,
        "navigation_gain_z": 3.0,
        "maximum_overload_g": 30.0,
        "missile_initial_speed_mps": 500.0,
        "maximum_missile_speed_mps": 900.0,
        "powered_duration_s": 2.0,
        "powered_acceleration_mps2": 100.0,
        "drag_coefficient": 1.0e-5,
        "missile_max_flight_time_s": 60.0,
        "hit_radius_m": 30.0,
    }
    missile = PaperMissile("m1", "red_0", "blue_0", np.zeros(3),
                           np.array([500.0, 0.0, 0.0]), cfg)
    speeds = []
    for _ in range(30):
        missile.step(np.array([10000.0, 1000.0, 200.0]),
                     np.array([200.0, 0.0, 0.0]), 1.0 / 60.0)
        speeds.append(missile.speed_mps)
        assert missile.commanded_overload_g <= 30.0 + 1e-6
    assert missile.navigation_gain_y == 3.0
    assert missile.navigation_gain_z == 3.0
    assert max(speeds) - min(speeds) > 1e-3


def test_seeded_initial_perturbation_is_reproducible():
    path = str(CONFIG_DIR / "tam_paper_env_v1_3v2.yaml")
    env = make_env(path, dynamics_backend="simple", initial_perturbation="low")
    env.reset(seed=123)
    first = {a.agent_id: a.position.copy() for a in env.task.agents}
    env.reset(seed=123)
    second = {a.agent_id: a.position.copy() for a in env.task.agents}
    assert all(np.array_equal(first[k], second[k]) for k in first)
    env.close()
