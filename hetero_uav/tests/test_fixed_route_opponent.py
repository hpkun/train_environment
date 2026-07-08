from __future__ import annotations

import numpy as np
import pytest

from algorithms.mappo.opponent_policy import OpponentPolicy


class _DummyEnv:
    VELOCITY_MIN = 102.0
    VELOCITY_MAX = 408.0

    def __init__(self):
        self.refresh_calls = 0

    def refresh_engaged_targets(self):
        self.refresh_calls += 1
        return set()


def test_fixed_route_outputs_constant_level_flight_actions():
    policy = OpponentPolicy(mode="fixed_route", seed=0)
    env = _DummyEnv()

    actions = policy.act({}, ["blue_0", "blue_1"], env=env)

    expected_speed = 2.0 * (230.0 - 102.0) / (408.0 - 102.0) - 1.0
    assert set(actions) == {"blue_0", "blue_1"}
    for action in actions.values():
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert action[0] == pytest.approx(0.0)
        assert action[1] == pytest.approx(0.0)
        assert action[2] == pytest.approx(expected_speed)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    assert env.refresh_calls == 1


def test_fixed_route_does_not_depend_on_enemy_observation_or_tracking_state():
    policy = OpponentPolicy(mode="fixed_route", seed=0)
    policy.last_targets[0] = 2
    policy.last_assigned_targets["blue_0"] = 1
    policy.lost_target_steps[0] = 5
    policy.last_target_distances[0] = 123.0

    actions = policy.act(
        {"blue_0": {"enemy_states": np.zeros((0, 3), dtype=np.float32)}},
        ["blue_0"],
        env=None,
    )

    expected_speed = 2.0 * (230.0 - 102.0) / (408.0 - 102.0) - 1.0
    np.testing.assert_allclose(actions["blue_0"], np.array([0.0, 0.0, expected_speed], dtype=np.float32))
    assert policy.last_targets == {0: 2}
    assert policy.last_assigned_targets == {"blue_0": 1}
    assert policy.lost_target_steps == {0: 5}
    assert policy.last_target_distances == {0: 123.0}
