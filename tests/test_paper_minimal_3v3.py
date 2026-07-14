from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    environment_config_snapshot,
)
from configs.experiment_presets import get_preset
from configs.paper_minimal_3v3_spec import (
    MINIMAL_PAPER_ENVIRONMENT_CONFIG,
    PAPER_MINIMAL_ENVIRONMENT_PROFILE,
    minimal_environment_snapshot,
)
from my_uav_env.blue_policy_profiles import BluePolicyController
from my_uav_env.env import UavCombatEnv
from my_uav_env.pid_controller import PIDController
from my_uav_env.sensors import radar_diagnostic
from my_uav_env.simulator import MissileSimulator
from train_vanilla_mappo import (
    Config,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _episode_is_invalid,
    _minimal_altitude_reward_config,
    _unpack_and_validate_checkpoint,
)


def _strict_obs(num_own=3, num_enemy=3, heading=0.0):
    ego = np.zeros(10, dtype=np.float32)
    ego[2] = 6000.0
    ego[3] = 300.0
    ego[6] = heading
    enemies = np.zeros((num_enemy, 10), dtype=np.float32)
    for index in range(num_enemy):
        enemies[index, 0] = 10_000.0
        enemies[index, 1] = 1000.0 * (index - 1)
        enemies[index, 7] = np.arctan2(enemies[index, 1], enemies[index, 0])
        enemies[index, 9] = np.hypot(enemies[index, 0], enemies[index, 1])
    return {
        "ego_state": ego,
        "ally_states": np.zeros((num_own - 1, 10), dtype=np.float32),
        "enemy_states": enemies,
        "alive_mask": np.ones(num_own + num_enemy, dtype=np.float32),
        "missile_warning": np.zeros(1, dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
    }


def test_minimal_profile_snapshot_is_isolated_and_sourced():
    minimal = minimal_environment_snapshot(
        num_red=3, num_blue=3, sim_freq=60, agent_interaction_steps=12,
        max_episode_length=1400, blue_policy_profile="paper_minimal_fixed_pair_v1",
        seed=7)
    reference = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, num_red=3, num_blue=3,
        sim_freq=60, agent_interaction_steps=12, seed=7,
        blue_policy_profile="paper_pursuit")
    assert minimal["environment_profile"]["value"] == PAPER_MINIMAL_ENVIRONMENT_PROFILE
    assert minimal["initial_altitude_m"] == {
        "value": 6000.0, "source": "paper_unspecified_minimal", "note": ""}
    assert minimal["environment_config_fingerprint"] != reference[
        "environment_config_fingerprint"]


def test_minimal_presets_and_dimensions_are_fixed():
    p = get_preset("vanilla_3v3_paper_minimal_smoke")
    assert (p["num_red"], p["num_blue"], p["num_envs"]) == (3, 3, 2)
    assert p["total_env_steps"] == 2000
    assert p["environment_profile"] == PAPER_MINIMAL_ENVIRONMENT_PROFILE
    assert _compute_obs_dim(3, 3, True, "paper_strict") == 60
    assert _compute_global_state_dim(3, "paper_strict") == 30


def test_minimal_fixed_pair_never_switches_and_commands_300_mps():
    obs = {f"blue_{i}": _strict_obs() for i in range(3)}
    obs["blue_0"]["enemy_states"][0, 2] = -10_000.0
    controller = BluePolicyController("paper_minimal_fixed_pair_v1")
    controller.reset(
        [f"blue_{i}" for i in range(3)], [f"red_{i}" for i in range(3)],
        {f"blue_{i}": 0.0 for i in range(3)},
        {f"blue_{i}": 6000.0 for i in range(3)})
    actions = controller.act(
        obs, 3, 3, engaged_targets={"red_0", "red_1", "red_2"},
        own_positions={}, own_headings={f"blue_{i}": 0.0 for i in range(3)},
        current_step=1)
    expected_speed = 2.0 * (300.0 - 102.0) / 306.0 - 1.0
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1", "blue_2": "red_2"}
    assert controller.snapshot_episode_diagnostics()["blue_target_switches_total"] == 0
    assert all(action[2] == pytest.approx(expected_speed) for action in actions.values())
    assert abs(actions["blue_0"][0]) <= 10.0 / 90.0 + 1e-7

    obs["blue_0"]["enemy_states"][0] = 0.0
    held = controller.act(
        obs, 3, 3, engaged_targets=set(), own_positions={},
        own_headings={f"blue_{i}": 0.25 for i in range(3)}, current_step=2)
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert held["blue_0"][1] == pytest.approx(0.25 / np.pi)
    assert row["assigned_target_id"] == "red_0"
    assert row["assignment_reason"] == "track_unavailable_hold"

    obs["blue_0"]["alive_mask"][3] = 0.0
    controller.act(
        obs, 3, 3, engaged_targets=set(), own_positions={},
        own_headings={f"blue_{i}": 0.5 for i in range(3)}, current_step=3)
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["assigned_target_id"] == "red_0"
    assert row["assignment_reason"] == "target_dead_hold"
    assert controller.snapshot_episode_diagnostics()["blue_target_switches_total"] == 0


def test_minimal_straight_patrol_does_not_read_enemy_states():
    class Forbidden(dict):
        def get(self, key, default=None):
            if key == "enemy_states":
                raise AssertionError("straight patrol read enemy_states")
            return super().get(key, default)

    obs = Forbidden(_strict_obs())
    controller = BluePolicyController("paper_minimal_straight_patrol_v1")
    controller.reset(["blue_0"], ["red_0"], {"blue_0": 1.0}, {"blue_0": 6000.0})
    action = controller.act(
        {"blue_0": obs}, 1, 1, set(), {}, {"blue_0": 0.2}, 5)["blue_0"]
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(1.0 / np.pi)


class _RadarSim:
    def __init__(self, position, rpy):
        self._position = np.asarray(position, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)

    def get_position(self):
        return self._position

    def get_rpy(self):
        return self._rpy


def test_minimal_constant_rcs_and_radar_are_color_symmetric():
    a = _RadarSim([0, 0, 0], [0, 0, 0])
    front = _RadarSim([1000, 0, 0], [0, 0, 0])
    turned = _RadarSim([1000, 0, 0], [0.7, -0.4, 2.0])
    d1 = radar_diagnostic(
        a, front, MINIMAL_PAPER_ENVIRONMENT_CONFIG.radar,
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.rcs)
    d2 = radar_diagnostic(
        a, turned, MINIMAL_PAPER_ENVIRONMENT_CONFIG.radar,
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.rcs)
    reverse = radar_diagnostic(
        front, a, MINIMAL_PAPER_ENVIRONMENT_CONFIG.radar,
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.rcs)
    assert d1["interpolated_rcs_m2"] == d2["interpolated_rcs_m2"] == 1.0
    assert d1["radar_max_range_m"] == d2["radar_max_range_m"]
    assert isinstance(reverse["radar_detected"], bool)


def test_minimal_eo_visibility_is_distance_only():
    env = object.__new__(UavCombatEnv)
    env.is_paper_minimal = True
    env.environment_config = MINIMAL_PAPER_ENVIRONMENT_CONFIG
    observer = _RadarSim([0, 0, 0], [0, 0, 0])
    behind = _RadarSim([-9999, 0, 0], [0, 0, 0])
    outside = _RadarSim([10001, 0, 0], [0, 0, 0])
    assert env._is_detected_by_electro_optical(observer, behind)
    assert not env._is_detected_by_electro_optical(observer, outside)


def test_minimal_shared_pid_has_identical_mirrored_outputs():
    left = PIDController(1 / 60, profile="paper_minimal_shared_v1",
                         config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid)
    right = PIDController(1 / 60, profile="paper_minimal_shared_v1",
                          config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid)
    out_left = left.compute_control(
        [0.0, 0.0, 0.0], 300.0, 0.1, 0.2, 320.0)
    out_right = right.compute_control(
        [0.0, 0.0, np.pi], 300.0, 0.1, -np.pi + 0.2, 320.0)
    assert np.all(np.isfinite(out_left))
    assert np.allclose(out_left, out_right, atol=1e-8)
    pitch = PIDController(1 / 60, profile="paper_minimal_shared_v1",
                          config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid)
    heading = PIDController(1 / 60, profile="paper_minimal_shared_v1",
                            config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid)
    speed = PIDController(1 / 60, profile="paper_minimal_shared_v1",
                          config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid)
    # F-16 actuator convention: negative elevator command pitches up.
    assert pitch.compute_control([0, 0, 0], 300, 0.1, 0, 300)[1] < 0
    assert heading.compute_control([0, 0, 0], 300, 0, 0.1, 300)[0] > 0
    assert speed.compute_control([0, 0, 0], 250, 0, 0, 300)[3] > 0


def test_minimal_point_mass_keeps_speed_and_has_no_propulsion_update():
    missile = MissileSimulator(
        guidance_mode="paper_minimal_point_mass_v1",
        config=MINIMAL_PAPER_ENVIRONMENT_CONFIG.missile)
    missile._position[:] = [0.0, 0.0, 6000.0]
    missile._geodetic[:] = [120.0, 60.0, 6000.0]
    missile._velocity[:] = [300.0, 0.0, 0.0]
    missile._posture[:] = [0.0, 0.0, 0.0]
    missile.lon0, missile.lat0, missile.alt0 = 120.0, 60.0, 6000.0
    missile._m = missile._m0
    missile._t = 0.0
    mass = missile._m
    missile._state_trans([0.0, 1.0])
    assert np.linalg.norm(missile.get_velocity()) == pytest.approx(300.0)
    assert missile._m == mass
    assert missile.Isp == 0.0


def test_minimal_altitude_reward_has_zero_high_tail_and_mean_aggregation():
    cfg = _minimal_altitude_reward_config()
    assert cfg.high_altitude_tail == 0.0
    env = object.__new__(UavCombatEnv)
    env.is_paper_minimal = True
    env.altitude_reward_config = cfg

    class Sim:
        color = "Red"
        is_alive = True
        def __init__(self, altitude): self.altitude = altitude
        def get_geodetic(self): return (0.0, 0.0, self.altitude)

    ego = Sim(6000.0)
    env.blue_planes = {"blue_0": Sim(3000.0), "blue_1": Sim(3000.0)}
    assert env._altitude_reward(ego) == pytest.approx(1.0)
    ego.altitude = 14001.0
    env.blue_planes = {"blue_0": Sim(3000.0)}
    assert env._altitude_reward(ego) == 0.0


def test_minimal_checkpoint_metadata_cannot_mix_with_reference():
    cfg = Config()
    cfg.num_red = cfg.num_blue = 3
    cfg.num_envs = 2
    cfg.replay_buffer_size = 100
    cfg.environment_profile = PAPER_MINIMAL_ENVIRONMENT_PROFILE
    cfg.environment_version = PAPER_MINIMAL_ENVIRONMENT_PROFILE
    cfg.blue_policy_profile = "paper_minimal_fixed_pair_v1"
    cfg.reward_mode = "paper_minimal_joint_v1"
    cfg.pid_profile = "paper_minimal_shared_v1"
    cfg.missile_guidance_mode = "paper_minimal_point_mass_v1"
    cfg.altitude_reward_config = _minimal_altitude_reward_config()
    minimal = _checkpoint_metadata(cfg, 60, 30)
    payload = {"state_dict": {}, "metadata": minimal, "model_kind": "actor"}
    reference = dict(minimal)
    reference["environment_profile"] = "brma_paper_profile_v1"
    with pytest.raises(ValueError, match="environment_profile"):
        _unpack_and_validate_checkpoint(payload, reference, "actor")
    reference_payload = {
        "state_dict": {}, "metadata": reference, "model_kind": "actor"}
    with pytest.raises(ValueError, match="environment_profile"):
        _unpack_and_validate_checkpoint(reference_payload, minimal, "actor")


def test_minimal_env_reset_is_mirrored_and_10_dimensional():
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3,
        environment_profile=PAPER_MINIMAL_ENVIRONMENT_PROFILE,
        suppress_jsbsim_output=True)
    try:
        obs, info = env.reset(seed=3)
        assert obs["red_0"]["ego_state"].shape == (10,)
        assert _compute_obs_dim(3, 3, True, "paper_strict") == 60
        assert env.sim_freq == 60
        assert env.agent_interaction_steps == 12
        assert env.env_dt == pytest.approx(0.2)
        assert env.max_steps == 1400
        for index in range(3):
            red = env.red_planes[f"red_{index}"]
            blue = env.blue_planes[f"blue_{index}"]
            assert np.linalg.norm(red.get_position() - blue.get_position()) == pytest.approx(
                10_000.0, rel=2e-3)
            assert red.get_geodetic()[2] == pytest.approx(6000.0, rel=2e-3)
            assert blue.get_geodetic()[2] == pytest.approx(6000.0, rel=2e-3)
            assert np.linalg.norm(red.get_velocity()) == pytest.approx(300.0, rel=2e-2)
            assert np.linalg.norm(blue.get_velocity()) == pytest.approx(300.0, rel=2e-2)
        snapshot = info["__environment_config__"]
        assert snapshot["environment_profile"]["value"] == PAPER_MINIMAL_ENVIRONMENT_PROFILE
    finally:
        env.close()


def test_minimal_invalid_episode_is_truncated_without_winner():
    env = object.__new__(UavCombatEnv)
    env.is_paper_minimal = True
    env._invalid_numerical_episode = True
    env._invalid_numerical_reasons = ["red_0:NonFiniteState"]
    env.current_step = 1
    env.max_steps = 1400
    env.agent_ids = ["red_0", "blue_0"]
    assert all(env._get_truncated().values())
    assert _episode_is_invalid({
        "__episode__": {"invalid_numerical_episode": True}})
    assert not _episode_is_invalid({
        "__episode__": {"invalid_numerical_episode": False}})
