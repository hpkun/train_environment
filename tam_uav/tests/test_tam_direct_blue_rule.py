from __future__ import annotations

import numpy as np

from algorithms.mappo.opponent_policy import OpponentPolicy


def test_tam_direct_fsm_level_cruises_without_target():
    policy = OpponentPolicy("tam_direct_fsm")

    actions = policy.act({"blue_0": {}}, ["blue_0"])

    np.testing.assert_allclose(actions["blue_0"], [0.6, 0.0, 0.0, 0.0])


def test_tam_direct_fsm_outputs_four_finite_clipped_controls():
    policy = OpponentPolicy("tam_direct_fsm")
    obs = {"blue_0": {"enemy_states": np.array([[0.2, 3.0, -4.0]])}}

    action = policy.act(obs, ["blue_0"])["blue_0"]

    assert action.shape == (4,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert action[1] > 0.0
    assert action[2] < 0.0
    assert action[3] == 0.0


def test_tam_direct_fsm_distinguishes_left_right_and_high_low_targets():
    policy = OpponentPolicy("tam_direct_fsm")

    left = policy.act(
        {"blue_0": {"enemy_states": np.array([[0.2, -0.4, 0.0]])}}, ["blue_0"]
    )["blue_0"]
    right = policy.act(
        {"blue_0": {"enemy_states": np.array([[0.2, 0.4, 0.0]])}}, ["blue_0"]
    )["blue_0"]
    low = policy.act(
        {"blue_0": {"enemy_states": np.array([[0.2, 0.0, -0.4]])}}, ["blue_0"]
    )["blue_0"]
    high = policy.act(
        {"blue_0": {"enemy_states": np.array([[0.2, 0.0, 0.4]])}}, ["blue_0"]
    )["blue_0"]

    assert left[1] < 0.0 < right[1]
    assert low[2] < 0.0 < high[2]
