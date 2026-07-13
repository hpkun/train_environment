import copy
import math

import numpy as np
import pytest

import rule_based_agent as rules
from my_uav_env.alignment.state_extractor import _rotation_inertial_to_body
from rule_based_agent import (
    _blue_simple_pursuit_action_impl,
    _paper_pursuit_geometry,
    blue_coordinated_actions,
)
from train_vanilla_mappo import Config, _compute_global_state_dim, _compute_obs_dim


def ego(roll=0.0, pitch=0.0, heading=0.0):
    return np.array([0, 0, 6096, 250, roll, pitch, heading, 0, 0, 0], np.float32)


def target_from_neu(relative_neu, roll=0.0, pitch=0.0, heading=0.0):
    body = _rotation_inertial_to_body(roll, pitch, heading) @ np.asarray(relative_neu)
    row = np.zeros(10, np.float32)
    row[:3] = body
    row[9] = np.linalg.norm(relative_neu)
    return row


@pytest.mark.parametrize("roll,pitch", [
    (0.0, 0.0), (math.pi / 4, 0.0), (-math.pi / 4, 0.0),
    (0.0, math.radians(20)), (0.0, math.radians(-20)),
])
def test_same_altitude_target_has_zero_pitch_for_attitude(roll, pitch):
    row = target_from_neu([5000, 1000, 0], roll, pitch, 0.3)
    geometry = _paper_pursuit_geometry(ego(roll, pitch, 0.3), row)
    assert geometry is not None
    assert abs(geometry[1]) < 1e-6


def test_high_low_and_pitch_clipping():
    high = _paper_pursuit_geometry(ego(), target_from_neu([100, 0, 10000]))
    low = _paper_pursuit_geometry(ego(), target_from_neu([100, 0, -10000]))
    assert high[1] == pytest.approx(math.radians(15))
    assert low[1] == pytest.approx(math.radians(-15))


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, math.pi - 1e-6, -math.pi + 1e-6])
def test_absolute_heading_uses_inertial_neu_and_wraps(heading):
    relative_neu = np.array([-1000.0, -1.0, 0.0])
    row = target_from_neu(relative_neu, 0.0, 0.0, heading)
    desired, _pitch, _range = _paper_pursuit_geometry(ego(0, 0, heading), row)
    expected = (math.atan2(relative_neu[1], relative_neu[0]) + math.pi) % (
        2 * math.pi) - math.pi
    assert desired == pytest.approx(expected)


@pytest.mark.parametrize("bad", [np.zeros(9), np.full(10, np.nan), np.full(10, np.inf)])
def test_invalid_target_geometry_returns_none(bad):
    assert _paper_pursuit_geometry(ego(), bad) is None


def strict_obs(enemy_rows, alive=None):
    if alive is None:
        alive = [1] * len(enemy_rows)
    return {
        "ego_state": ego(),
        "ally_states": np.zeros((2, 10), np.float32),
        "enemy_states": np.asarray(enemy_rows, np.float32),
        "alive_mask": np.array([1, 1, 1, *alive], np.float32),
        "altitude": np.array([6096], np.float32),
        "velocity": np.array([250, 0, 0], np.float32),
    }


def test_invalid_paper_target_falls_back_without_touching_memories():
    obs = strict_obs([np.full(10, np.nan), np.zeros(10), np.zeros(10)])
    before = {
        name: copy.deepcopy(getattr(rules, name)) for name in (
            "_prev_heading_cmd", "_prev_lead_bearing", "_last_target_bearing",
            "_lost_target_steps", "_simple_last_seen_bearing",
            "_simple_lost_steps", "_simple_debug_state")
    }
    action = _blue_simple_pursuit_action_impl(
        obs, 3, 3, 0, 0, own_heading=0.4, paper_profile=True)
    assert np.isfinite(action).all()
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.4 / math.pi)
    for name, value in before.items():
        assert getattr(rules, name) == value


def test_three_by_three_assignment_is_distinct_and_engaged_does_not_stop_pursuit(monkeypatch):
    rows = [target_from_neu([2000 + 1000 * i, 100 * i, 0]) for i in range(3)]
    observations = {f"blue_{i}": strict_obs(rows) for i in range(3)}
    assigned = []
    original = rules._blue_simple_pursuit_action_impl

    def capture(*args, **kwargs):
        assigned.append(kwargs["forced_target_idx"])
        return original(*args, **kwargs)

    monkeypatch.setattr(rules, "_blue_simple_pursuit_action_impl", capture)
    actions = blue_coordinated_actions(
        observations, 3, 3,
        engaged_targets={"red_0", "red_1", "red_2"},
        pursuit_mode="paper_pursuit")
    assert set(actions) == {"blue_0", "blue_1", "blue_2"}
    assert sorted(assigned) == [0, 1, 2]


def test_main_defaults_are_strict_3v3_dimensions():
    cfg = Config()
    assert (cfg.num_red, cfg.num_blue, cfg.max_episode_length) == (3, 3, 1400)
    assert cfg.obs_mode == "paper_strict"
    assert cfg.reward_mode == "paper_joint"
    assert cfg.pid_profile == "paper"
    assert _compute_obs_dim(3, 3, True, "paper_strict") == 66
    assert _compute_global_state_dim(3, "paper_strict") == 30
