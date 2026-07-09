import math

import numpy as np

from my_uav_env import UavCombatEnv
from my_uav_env.alignment.los_geometry import compute_body_x_q_los
from my_uav_env.alignment.reward_utils import (
    REWARD_VERSION,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)
from my_uav_env.alignment.state_extractor import extract_relative_state
from train_vanilla_mappo import Config


def test_ta_uses_paper_eq20_scale():
    samples = {
        0.0: 10.0,
        4.0: 10.0,
        10.0: 1.0 + 2.0 * (15.0 - 10.0) / 15.0,
        15.0: 1.0,
        35.0: 0.0,
        40.0: 0.0,
    }

    for q_deg, expected in samples.items():
        assert math.isclose(
            ta_angle_advantage_fixed(q_deg),
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


def test_td_uses_15km_with_meter_inputs():
    assert td_distance_advantage(0.0) == 1.0
    assert td_distance_advantage(15000.0) == 1.0
    assert math.isclose(
        td_distance_advantage(30000.0),
        math.exp(1.0 - 30.0 / 15.0),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_reward_version_names_paper_eq20_default():
    assert "paper_eq20" in REWARD_VERSION
    assert "fixed_ta" not in REWARD_VERSION


def test_main_training_defaults_match_paper_scale():
    cfg = Config()
    assert cfg.num_red == 6
    assert cfg.num_blue == 6
    assert cfg.max_episode_length == 1400


def test_blue_gcas_default_is_disabled():
    env = UavCombatEnv()
    try:
        assert env.enable_gcas_for_blue is False
    finally:
        env.close()


def test_paper_strict_observation_entities_are_10_dim():
    env = UavCombatEnv(
        max_num_red=1,
        max_num_blue=1,
        obs_mode="paper_strict",
        suppress_jsbsim_output=True,
    )
    try:
        obs, _ = env.reset()
        red_obs = obs["red_0"]
        assert red_obs["ego_state"].shape == (10,)
        assert red_obs["enemy_states"].shape == (1, 10)
        assert red_obs["ally_states"].shape == (0, 10)
        assert np.isfinite(red_obs["ego_state"]).all()
    finally:
        env.close()


def test_missile_timing_scales_with_sim_freq():
    env = UavCombatEnv(sim_freq=40)
    try:
        assert env.missile_lock_delay_frames == 10
        assert env.missile_cooldown_frames == 20
    finally:
        env.close()


class _FakeSim:
    def __init__(self, pos, vel, rpy=(0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return self._rpy


def test_radar_blind_relative_state_masks_target_velocity_fields():
    observer = _FakeSim([0.0, 0.0, 0.0], [200.0, 0.0, 0.0])
    target = _FakeSim([1000.0, 100.0, 50.0], [123.0, 45.0, -6.0])

    row = extract_relative_state(observer, target, radar_detected=False)

    assert row[3] == 0.0
    assert row[4] == 0.0
    assert row[5] == 0.0


def test_body_x_q_los_geometry_cardinal_cases():
    pos = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])

    assert math.isclose(compute_body_x_q_los(pos, rpy, [1000.0, 0.0, 0.0]), 0.0)
    assert math.isclose(compute_body_x_q_los(pos, rpy, [-1000.0, 0.0, 0.0]), math.pi)
    assert math.isclose(
        compute_body_x_q_los(pos, rpy, [0.0, 1000.0, 0.0]),
        math.pi / 2.0,
    )
