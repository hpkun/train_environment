from __future__ import annotations

import numpy as np

from algorithms.mappo import opponent_policy
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def test_command_to_indices_maps_endpoints_and_returns_int64():
    indices = opponent_policy.tam_direct_command_to_indices([0.4, -1.0, 0.0, 1.0], 40, 0.4, 0.9)
    assert indices.dtype == np.int64
    assert indices.shape == (4,)
    np.testing.assert_array_equal(indices, [0, 0, 20, 39])


def test_tam_direct_fsm_outputs_valid_indices_for_formal_env():
    env = make_env(CONFIG)
    obs, _info = env.reset(seed=0)
    policy = OpponentPolicy("tam_direct_fsm")
    actions = policy.act(obs, env.blue_ids, env=env)
    for action in actions.values():
        assert action.dtype == np.int64
        assert action.shape == (4,)
        assert np.all((0 <= action) & (action < 40))
    env.close()


def test_tam_direct_fsm_legacy_env_keeps_continuous_command():
    class LegacyEnv:
        tam_action_distribution = "continuous_quantized"

    policy = OpponentPolicy("tam_direct_fsm")
    action = policy.act({"blue_0": {}}, ["blue_0"], env=LegacyEnv())["blue_0"]
    assert action.dtype == np.float32
    assert action.shape == (4,)


def test_tam_direct_fsm_blue_survives_300_step_short_rollout():
    env = make_env(CONFIG)
    obs, _info = env.reset(seed=3)
    policy = OpponentPolicy("tam_direct_fsm")
    neutral_red = np.array([39, 20, 18, 20], dtype=np.int64)
    for _ in range(300):
        actions = {rid: neutral_red.copy() for rid in env.red_ids}
        actions.update(policy.act(obs, env.blue_ids, env=env))
        obs, _rewards, terminated, truncated, _info = env.step(actions)
        if all(terminated.values()) or all(truncated.values()):
            break
    assert all(env.blue_planes[bid].is_alive for bid in env.blue_ids)
    env.close()
