from __future__ import annotations

import numpy as np
import pytest

from configs.paper_3v3_spec import paper_environment_snapshot
from configs.paper_6v6_spec import (
    MISSILE_LAUNCH_SPEED_MPS_6V6,
    PAPER_BLUE_POLICY_PROFILE_6V6,
    PAPER_ENVIRONMENT_CONFIG_6V6,
    PAPER_ENVIRONMENT_PROFILE_6V6,
    PAPER_MISSILE_GUIDANCE_MODE_6V6,
    PAPER_PID_PROFILE_6V6,
    PAPER_REWARD_MODE_6V6,
    PAPER_REWARD_VERSION_6V6,
    PID_THROTTLE_BASE_6V6,
    paper_6v6_environment_snapshot,
)
from my_uav_env.blue_policy_profiles import BluePolicyController
from my_uav_env.env import UavCombatEnv
from my_uav_env.pid_controller import PIDController
from my_uav_env.simulator import MissileSimulator
from train_vanilla_mappo import (
    Config,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _flatten_obs,
    _global_state_from_local_obs_flats,
    _paper_6v6_altitude_reward_config,
    _training_core_config,
    _unpack_actor_checkpoint_for_evaluation,
    VanillaActor,
)


def make_env() -> UavCombatEnv:
    return UavCombatEnv(
        max_num_red=6, max_num_blue=6, max_steps=1400,
        environment_profile=PAPER_ENVIRONMENT_PROFILE_6V6,
        blue_policy_profile=PAPER_BLUE_POLICY_PROFILE_6V6,
        pid_profile=PAPER_PID_PROFILE_6V6,
        pid_throttle_base=PID_THROTTLE_BASE_6V6,
        reward_mode=PAPER_REWARD_MODE_6V6,
        missile_guidance_mode=PAPER_MISSILE_GUIDANCE_MODE_6V6,
        altitude_reward_config=_paper_6v6_altitude_reward_config(),
        suppress_jsbsim_output=True)


def test_6v6_contract_is_stable_and_isolated_from_3v3():
    first = paper_6v6_environment_snapshot(seed=3)
    second = paper_6v6_environment_snapshot(seed=7)
    assert first["environment_config_fingerprint"] == second[
        "environment_config_fingerprint"]
    assert first["environment_config_fingerprint"] != paper_environment_snapshot(
        seed=3)["environment_config_fingerprint"]
    assert first["actor_input_dim"]["value"] == 120
    assert first["critic_input_dim"]["value"] == 60


def test_6v6_vanilla_dimensions_and_flatten_exclude_masks():
    assert _compute_obs_dim(6, 6, True) == 120
    assert _compute_global_state_dim(6) == 60
    obs = {
        "ego_state": np.ones(10, dtype=np.float32),
        "ally_states": np.ones((5, 10), dtype=np.float32),
        "enemy_states": np.ones((6, 10), dtype=np.float32),
        "entity_mask": np.zeros(12, dtype=np.int64),
        "observable_mask": np.ones(12, dtype=np.int64),
        "enemy_detected_mask": np.ones(12, dtype=np.int64),
        "radar_mask": np.ones(12, dtype=np.int64),
        "approximate_position_mask": np.zeros(12, dtype=np.int64),
    }
    assert _flatten_obs(obs).shape == (120,)


def test_6v6_checkpoint_metadata_uses_independent_contract():
    config = Config()
    config.num_red = config.num_blue = 6
    config.environment_profile = PAPER_ENVIRONMENT_PROFILE_6V6
    config.environment_version = PAPER_ENVIRONMENT_PROFILE_6V6
    config.blue_policy_profile = PAPER_BLUE_POLICY_PROFILE_6V6
    config.pid_profile = PAPER_PID_PROFILE_6V6
    config.pid_throttle_base = PID_THROTTLE_BASE_6V6
    config.reward_mode = PAPER_REWARD_MODE_6V6
    config.missile_guidance_mode = PAPER_MISSILE_GUIDANCE_MODE_6V6
    config.altitude_reward_config = _paper_6v6_altitude_reward_config()
    metadata = _checkpoint_metadata(config, 120, 60)
    assert metadata["reward_version"] == PAPER_REWARD_VERSION_6V6
    assert metadata["actor_obs_dim"] == 120
    assert metadata["global_state_dim"] == 60
    assert metadata["environment_profile"] == PAPER_ENVIRONMENT_PROFILE_6V6
    core = _training_core_config(config, metadata)
    assert core["actor_input_dim"] == 120
    assert core["critic_input_dim"] == 60
    assert core["entity_count"] == 12
    assert core["observation_profile"] == "paper_table1_table2_6v6_v1"
    assert core["global_state_profile"] == "paper_red_self_state_6x10_v1"


def test_6v6_pid_uses_requested_eq13_atan2_ratios():
    errors = PIDController.paper_6v6_direction_errors(
        (0.0, 0.0, 0.0), 0.2, 0.3)
    d_body = errors[4]
    assert errors[0] == pytest.approx(np.arctan2(d_body[1], d_body[0]))
    assert errors[1] == pytest.approx(np.arctan2(d_body[2], d_body[0]))
    vertical = PIDController.paper_6v6_direction_errors(
        (0.0, 0.0, 0.0), np.pi / 2.0, 0.0)
    assert np.all(np.isfinite(vertical[0:2]))
    assert vertical[1] == pytest.approx(-np.pi / 2.0)


def test_6v6_reset_observation_and_short_step_are_finite():
    env = make_env()
    try:
        obs, info = env.reset(seed=3)
        assert len(obs) == 12
        red = obs["red_0"]
        assert red["ego_state"].shape == (10,)
        assert red["ally_states"].shape == (5, 10)
        assert red["enemy_states"].shape == (6, 10)
        for name in ("entity_mask", "observable_mask", "enemy_detected_mask",
                     "radar_mask",
                     "approximate_position_mask"):
            assert red[name].shape == (12,)
        assert _flatten_obs(red).shape == (120,)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids}
        actions.update(env.blue_policy_actions(
            {aid: obs[aid] for aid in env.blue_ids}))
        next_obs, rewards, _, _, step_info = env.step(actions)
        assert all(np.isfinite(value) for value in rewards.values())
        assert all(np.all(np.isfinite(_flatten_obs(next_obs[aid])))
                   for aid in env.agent_ids)
        assert step_info["__environment_config__"][
            "environment_config_fingerprint"] == info[
                "__environment_config__"]["environment_config_fingerprint"]
    finally:
        env.close()


def test_6v6_unobserved_enemy_has_no_velocity_leak_and_dead_slot_is_fixed():
    env = make_env()
    try:
        env.reset(seed=3)
        env._is_detected_by_radar = lambda *_: False
        env._is_detected_by_electro_optical = lambda *_: False
        obs = env._get_obs()
        red = obs["red_0"]
        np.testing.assert_array_equal(red["enemy_states"][:, 3:6], 0.0)
        assert np.all(red["enemy_detected_mask"][6:] == 0)
        assert np.all(red["approximate_position_mask"][6:] == 1)

        env.red_planes["red_2"].shotdown()
        obs = env._get_obs()
        assert np.all(obs["red_2"]["ego_state"] == 0.0)
        flats = [_flatten_obs(obs[aid]) for aid in env.red_ids]
        global_state = _global_state_from_local_obs_flats(flats)
        assert global_state.shape == (60,)
        assert np.all(global_state[20:30] == 0.0)
    finally:
        env.close()


def test_6v6_joint_reward_identity_and_terminal_coefficient():
    env = make_env()
    try:
        env.reset(seed=3)
        for sim in env.blue_planes.values():
            sim.shotdown()
        rewards, components = env._compute_rewards()
        local_sum = sum(components[aid]["local_reward"] for aid in env.red_ids)
        expected = local_sum + 30.0 * 6.0
        assert rewards["red_0"] == pytest.approx(expected)
        assert all(rewards[aid] == pytest.approx(expected) for aid in env.red_ids)
        assert components["red_0"]["red_joint_reward"] == pytest.approx(expected)
    finally:
        env.close()


def test_3v3_actor_checkpoint_is_rejected_by_6v6_evaluation_contract():
    config = Config()
    metadata = _checkpoint_metadata(config, 60, 30)
    actor = VanillaActor(obs_dim=60, action_dim=3)
    payload = {"model_kind": "actor", "state_dict": actor.state_dict(),
               "metadata": metadata}
    with pytest.raises(ValueError, match="contract mismatch"):
        _unpack_actor_checkpoint_for_evaluation(
            payload, num_red=6, num_blue=6, obs_dim=120, action_dim=3,
            obs_mode="paper_strict", obs_normalization="paper_fixed_v1",
            pid_profile=PAPER_PID_PROFILE_6V6,
            pid_throttle_base=PID_THROTTLE_BASE_6V6,
            reward_mode=PAPER_REWARD_MODE_6V6,
            missile_guidance_mode=PAPER_MISSILE_GUIDANCE_MODE_6V6,
            blue_policy_profile=PAPER_BLUE_POLICY_PROFILE_6V6,
            environment_profile=PAPER_ENVIRONMENT_PROFILE_6V6,
            max_episode_length=1400,
            initial_condition_randomization_mode="deterministic_v1")


def test_6v6_blue_profile_accepts_six_aircraft():
    controller = BluePolicyController(PAPER_BLUE_POLICY_PROFILE_6V6)
    controller.reset(
        [f"blue_{i}" for i in range(6)], [f"red_{i}" for i in range(6)],
        {}, {})
    assert len(controller.current_targets) == 6


class FakeAircraft:
    def __init__(self, uid, color, position, velocity, rpy=(0.0, 0.0, 0.0)):
        self.uid = uid
        self.color = color
        self.dt = 1.0 / 60.0
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.is_alive = True
        self.launch_missiles = []
        self.under_missiles = []
        self.lon0, self.lat0, self.alt0 = 120.0, 60.0, 6000.0

    def get_position(self):
        return self._position.copy()

    def get_velocity(self):
        return self._velocity.copy()

    def get_rpy(self):
        return self._rpy.copy()

    def get_geodetic(self):
        return np.array([self.lon0, self.lat0, self.alt0 + self._position[2]])

    def shotdown(self):
        self.is_alive = False


def test_6v6_eq7_eq11_missile_state_remains_finite_and_bounded():
    parent = FakeAircraft("red_0", "Red", (0.0, 0.0, 0.0), (300.0, 0.0, 0.0))
    target = FakeAircraft("blue_0", "Blue", (5000.0, 500.0, 0.0),
                          (-300.0, 0.0, 0.0))
    missile = MissileSimulator.create(
        parent, target, "m0", guidance_mode=PAPER_MISSILE_GUIDANCE_MODE_6V6,
        config=PAPER_ENVIRONMENT_CONFIG_6V6.missile,
        launch_speed_mps=MISSILE_LAUNCH_SPEED_MPS_6V6,
        rng=np.random.default_rng(3))
    initial_speed = np.linalg.norm(missile.get_velocity())
    for _ in range(10):
        missile.run()
        assert np.all(np.isfinite(missile.get_position()))
        assert np.all(np.isfinite(missile.get_velocity()))
        assert missile._maximum_command_g <= 30.0 + 1e-12
        if missile.is_done:
            break
    assert missile.target_distance < np.linalg.norm(
        target.get_position() - parent.get_position())
    assert np.linalg.norm(missile.get_velocity()) != pytest.approx(initial_speed)
