from __future__ import annotations

import copy
from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.experiment_presets import EXPERIMENT_PRESETS
from configs.paper_3v3_spec import (
    PAPER_BLUE_POLICY_PROFILE,
    PAPER_ENVIRONMENT_CONFIG,
    PAPER_ENVIRONMENT_PROFILE,
    PAPER_MWS_PROFILE,
    PAPER_MISSILE_GUIDANCE_MODE,
    PAPER_PID_PROFILE,
    PAPER_SENSOR_SUPPORT_PROFILE,
    PAPER_UNSPECIFIED_ENGINEERING,
    paper_environment_snapshot,
)
from my_uav_env.alignment.reward_utils import (
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
    AltitudeRewardConfig,
    altitude_reward_paper_eq17,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)
from my_uav_env.alignment.state_extractor import (
    extract_relative_state,
    extract_self_state,
    extract_self_state_with_meta,
)
from my_uav_env.blue_policy_profiles import BluePolicyController
from my_uav_env.env import UavCombatEnv
from my_uav_env.pid_controller import PIDController, PIDLoop
from my_uav_env.sensors import (
    bilinear_rcs_m2,
    radar_diagnostic,
    select_most_dangerous_missile,
)
from my_uav_env.simulator import MissileSimulator
from train_vanilla_mappo import (
    CHECKPOINT_SCHEMA_VERSION,
    TRAINING_STATE_SCHEMA_VERSION,
    Config,
    VanillaActor,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _flatten_obs,
    _minimal_altitude_reward_config,
    _training_core_config,
    _unpack_and_validate_checkpoint,
    _unpack_actor_checkpoint_for_evaluation,
    _validate_training_state,
)


class FakeAircraft:
    ALIVE = 0

    def __init__(self, uid, position, velocity=(300.0, 0.0, 0.0),
                 rpy=(0.0, 0.0, 0.0), color="Red"):
        self.uid = uid
        self.color = color
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self._status = self.ALIVE
        self.is_alive = True
        self.launch_missiles = []
        self.under_missiles = []
        self.dt = 1.0 / 60.0
        self.lon0 = 120.0
        self.lat0 = 60.0
        self.alt0 = 6000.0

    def get_position(self):
        return self._position.copy()

    def get_velocity(self):
        return self._velocity.copy()

    def get_rpy(self):
        return self._rpy.copy()

    def get_geodetic(self):
        return np.array([self.lon0, self.lat0, self._position[2]])

    def shotdown(self):
        self.is_alive = False

    def get_property_value(self, name):
        values = {"aero/alpha-rad": 0.12, "aero/beta-rad": -0.03}
        if name not in values:
            raise KeyError(name)
        return values[name]


def _evaluation_actor_payload() -> tuple[dict, Config]:
    config = Config()
    config.seed = 3
    config.total_env_steps = 100_000
    config.num_envs = 32
    config.replay_buffer_size = 2000
    config.altitude_reward_config = _minimal_altitude_reward_config()
    metadata = _checkpoint_metadata(config, 60, 30)
    actor = VanillaActor(
        obs_dim=60, action_dim=3,
        hidden=config.mlp_hidden, rnn_hidden=config.rnn_hidden_size)
    return {
        "state_dict": actor.state_dict(),
        "metadata": metadata,
        "model_kind": "actor",
    }, config


def _unpack_evaluation_actor(payload: dict, **overrides):
    arguments = {
        "num_red": 3,
        "num_blue": 3,
        "obs_dim": 60,
        "action_dim": 3,
        "obs_mode": "paper_strict",
        "obs_normalization": "paper_fixed_v1",
        "pid_profile": PAPER_PID_PROFILE,
        "pid_throttle_base": 0.8,
        "reward_mode": "paper_joint",
        "missile_guidance_mode": PAPER_MISSILE_GUIDANCE_MODE,
        "blue_policy_profile": PAPER_BLUE_POLICY_PROFILE,
        "environment_profile": PAPER_ENVIRONMENT_PROFILE,
        "max_episode_length": 1400,
        "initial_condition_randomization_mode": "deterministic_v1",
    }
    arguments.update(overrides)
    return _unpack_actor_checkpoint_for_evaluation(payload, **arguments)


def test_formal_profile_contract_and_dimensions():
    snapshot = paper_environment_snapshot(seed=3)
    assert snapshot["environment_profile"]["value"] == PAPER_ENVIRONMENT_PROFILE
    assert snapshot["current_scale"]["source"] == "intentional_3v3_deviation"
    assert _compute_obs_dim(3, 3, True) == 60
    assert _compute_global_state_dim(3) == 30
    assert CHECKPOINT_SCHEMA_VERSION == "vanilla_mappo_paper_3v3_v2"
    assert snapshot["entity_feature_dimension"]["source"] == "paper_explicit"
    assert snapshot["actor_input_dim"]["source"] == (
        "intentional_3v3_deviation_derived")
    assert snapshot["critic_input_dim"]["source"] == (
        "intentional_3v3_deviation_derived")
    assert snapshot["observation_schema"]["source"] == (
        "paper_equation_operational")
    low_altitude = snapshot["environment_config"]["scenario"][
        "arena_altitude_min_m"]
    assert low_altitude["value"] == pytest.approx(100.0)
    assert low_altitude["source"] == PAPER_UNSPECIFIED_ENGINEERING
    assert snapshot["reward_version"]["value"] == REWARD_VERSION
    assert snapshot["altitude_reward_high_altitude_tail"]["value"] == (
        pytest.approx(0.1))
    assert snapshot["altitude_reward_high_altitude_tail"]["source"] == (
        "paper_explicit")
    thresholds = snapshot["altitude_reward_engineering_thresholds_m"]
    assert thresholds["source"] == PAPER_UNSPECIFIED_ENGINEERING
    assert thresholds["value"]["d_att_max_m"] == pytest.approx(10000.000001)
    assert snapshot["situation_reward_q_los_definition"]["value"] == (
        "observer_velocity_to_target_los_3d_v1")
    assert snapshot["situation_reward_q_los_definition"]["source"] == (
        "paper_inferred")
    assert snapshot["environment_config_fingerprint"] != (
        "472a29e9da6bc6e96f7b9dac77889d3f7fcfa055835fca19d8efffe4f0b9d7b0")
    assert snapshot["mws_profile"]["value"] == PAPER_MWS_PROFILE
    assert snapshot["red_mws_mode"]["value"] == PAPER_MWS_PROFILE
    assert snapshot["blue_mws_mode"]["value"] == PAPER_MWS_PROFILE
    assert snapshot["sensor_support_profile"]["value"] == (
        PAPER_SENSOR_SUPPORT_PROFILE)
    assert snapshot["missile_model_scope"]["source"] == (
        "intentional_model_simplification")
    assert snapshot["pid_structure"]["value"] == "roll_pitch_speed_three_loop"
    assert snapshot["pid_structure"]["source"] == "paper_explicit"
    assert snapshot["pid_error_definition"]["source"] == (
        PAPER_UNSPECIFIED_ENGINEERING)
    assert all(not enabled for enabled in
               snapshot["formal_control_extras"]["value"].values())
    assert snapshot["environment_config"]["missile"]["navigation_constant"][
        "source"] == "paper_unspecified_engineering"
    assert snapshot["environment_config"]["missile"]["model"]["source"] == (
        "intentional_model_simplification")
    for preset_name in (
            "vanilla_3v3_paper_smoke", "vanilla_3v3_paper_main",
            "vanilla_3v3_paper_100k_diag"):
        preset = EXPERIMENT_PRESETS[preset_name]
        assert preset["environment_profile"] == PAPER_ENVIRONMENT_PROFILE
        assert preset["blue_policy_profile"] == PAPER_BLUE_POLICY_PROFILE
        assert preset["pid_profile"] == PAPER_PID_PROFILE


def test_environment_rejects_nonformal_contracts():
    with pytest.raises(ValueError):
        UavCombatEnv(max_num_red=2)
    with pytest.raises(ValueError):
        UavCombatEnv(obs_mode="engineering")
    with pytest.raises(ValueError):
        UavCombatEnv(blue_policy_profile="paper_pursuit")


def test_reset_observation_fields_dimensions_and_ammo():
    env = UavCombatEnv()
    try:
        obs, _ = env.reset(seed=3)
        row = obs["red_0"]
        assert set(row) == {
            "ego_state", "ally_states", "enemy_states", "entity_mask"}
        assert row["ego_state"].shape == (10,)
        assert row["ally_states"].shape == (2, 10)
        assert row["enemy_states"].shape == (3, 10)
        assert row["entity_mask"].tolist() == [0, 0, 0, 0, 0, 0]
        assert _flatten_obs(row).shape == (60,)
        assert all(env._get_sim(aid).num_left_missiles == 2
                   for aid in env.agent_ids)
    finally:
        env.close()


def test_real_jsbsim_alpha_beta_sources_survive_reset_and_step():
    env = UavCombatEnv()
    try:
        obs, _ = env.reset(seed=3)
        for _phase in range(2):
            state, meta = extract_self_state_with_meta(
                env.red_planes["red_0"], require_real_alpha_beta=True)
            assert meta["alpha_source"].startswith(("jsbsim:", "getter:"))
            assert meta["beta_source"].startswith(("jsbsim:", "getter:"))
            assert meta["alpha_source"] != "placeholder:0"
            assert meta["beta_source"] != "placeholder:0"
            assert np.all(np.isfinite(state[[7, 8]]))
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids}
            actions.update(env.blue_policy_actions(
                {aid: obs[aid] for aid in env.blue_ids}))
            obs, *_ = env.step(actions)
    finally:
        env.close()


def test_formal_environment_source_contains_no_legacy_branches():
    source = Path("my_uav_env/env.py").read_text(encoding="utf-8")
    for forbidden in (
            "if False", "_make_entity_vec", "_normalize_obs_vec",
            "_build_body_frame_entity", "GCAS", "11-dim"):
        assert forbidden not in source
    tree = ast.parse(source)
    reward = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_rewards")
    assert len(reward.body) == 2
    assert isinstance(reward.body[-1], ast.Return)


def test_table1_field_order_and_jsbsim_alpha_beta():
    sim = FakeAircraft(
        "red_0", (10.0, 20.0, 6000.0), (3.0, 4.0, -2.0),
        (0.1, 0.2, 0.3))
    state = extract_self_state(sim)
    np.testing.assert_allclose(state, [
        10.0, 20.0, 6000.0, np.sqrt(29.0),
        0.1, 0.2, 0.3, 0.12, -0.03, 2.0])


def test_table2_body_axes_and_masked_enemy_velocity():
    observer = FakeAircraft("red_0", (0.0, 0.0, 0.0))
    target = FakeAircraft("blue_0", (100.0, 20.0, -30.0), (1.0, 2.0, 3.0))
    full = extract_relative_state(observer, target, radar_detected=True)
    coarse = extract_relative_state(observer, target, radar_detected=False)
    np.testing.assert_allclose(full[:3], [100.0, 20.0, 30.0], atol=1e-6)
    assert full[5] == pytest.approx(np.sqrt(14.0))
    np.testing.assert_allclose(coarse[3:6], 0.0)
    assert np.all(np.isfinite(full))


def test_dead_entity_mask_and_zero_row():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        env.blue_planes["blue_1"].crash()
        obs = env._get_agent_obs_paper_strict("red_0")
        assert obs["entity_mask"].tolist() == [0, 0, 0, 0, 1, 0]
        np.testing.assert_array_equal(obs["enemy_states"][1], np.zeros(10))
    finally:
        env.close()


def test_rcs_is_nonconstant_and_fourth_root_range():
    cfg = PAPER_ENVIRONMENT_CONFIG.rcs
    front = bilinear_rcs_m2(
        0.0, 0.0, cfg.azimuth_grid_deg.value,
        cfg.elevation_grid_deg.value, cfg.table_m2.value)
    side = bilinear_rcs_m2(
        np.pi / 2, 0.0, cfg.azimuth_grid_deg.value,
        cfg.elevation_grid_deg.value, cfg.table_m2.value)
    assert front == pytest.approx(0.1)
    assert side == pytest.approx(2.0)
    assert side > front
    ratio = ((side / front) ** 0.25)
    assert ratio == pytest.approx(
        (cfg.range_constant.value * side ** 0.25)
        / (cfg.range_constant.value * front ** 0.25))
    left = bilinear_rcs_m2(
        -np.pi / 2, 0.0, cfg.azimuth_grid_deg.value,
        cfg.elevation_grid_deg.value, cfg.table_m2.value)
    above = bilinear_rcs_m2(
        0.0, np.pi / 2, cfg.azimuth_grid_deg.value,
        cfg.elevation_grid_deg.value, cfg.table_m2.value)
    below = bilinear_rcs_m2(
        0.0, -np.pi / 2, cfg.azimuth_grid_deg.value,
        cfg.elevation_grid_deg.value, cfg.table_m2.value)
    assert left == pytest.approx(side)
    assert above == pytest.approx(below)
    assert cfg.table_m2.source == PAPER_UNSPECIFIED_ENGINEERING


def _strict_blue_obs(ranges, bearings=None, elevations=None):
    ranges = list(ranges)
    bearings = [0.0] * len(ranges) if bearings is None else list(bearings)
    elevations = [0.0] * len(ranges) if elevations is None else list(elevations)
    enemies = np.zeros((len(ranges), 10), dtype=np.float32)
    for index, (distance, bearing, elevation) in enumerate(
            zip(ranges, bearings, elevations)):
        enemies[index, 6] = elevation
        enemies[index, 7] = bearing
        enemies[index, 9] = distance
    return {"enemy_states": enemies}


def test_dynamic_nearest_unique_assignment_and_live_switch():
    controller = BluePolicyController(PAPER_BLUE_POLICY_PROFILE)
    controller.reset(["blue_0", "blue_1", "blue_2"],
                     ["red_0", "red_1", "red_2"], {}, {})
    blue_obs = {
        "blue_0": _strict_blue_obs([100.0, 110.0, 120.0]),
        "blue_1": _strict_blue_obs([110.0, 100.0, 120.0]),
        "blue_2": _strict_blue_obs([120.0, 110.0, 100.0]),
    }
    kwargs = dict(
        blue_obs=blue_obs, num_blue=3, num_red=3, engaged_targets=set(),
        own_positions={}, own_headings={key: 0.0 for key in blue_obs},
        current_step=0, own_alive={key: True for key in blue_obs},
        enemy_positions={"red_0": np.full(3, 999999.0)},
        enemy_alive={f"red_{index}": True for index in range(3)})
    actions = controller.act(**kwargs)
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1", "blue_2": "red_2"}
    assert all(action[2] == pytest.approx((300.0 - 102.0) / 306.0 * 2 - 1)
               for action in actions.values())
    blue_obs["blue_0"] = _strict_blue_obs([1000.0, 10.0, 120.0])
    controller.act(**kwargs)
    assert controller.current_targets["blue_0"] == "red_1"
    assert controller.target_switch_counts["blue_0"] == 1


def test_blue_assignment_ties_use_fixed_ids_and_no_random(monkeypatch):
    def random_forbidden(*_args, **_kwargs):
        raise AssertionError("Blue policy must not consume random numbers")

    monkeypatch.setattr(np.random, "random", random_forbidden)
    monkeypatch.setattr(np.random, "choice", random_forbidden)
    controller = BluePolicyController(PAPER_BLUE_POLICY_PROFILE)
    controller.reset(["blue_0", "blue_1"], ["red_0", "red_1"], {}, {})
    blue_obs = {
        "blue_0": _strict_blue_obs([100.0, 100.0]),
        "blue_1": _strict_blue_obs([100.0, 100.0]),
    }
    controller.act(
        blue_obs, 2, 2, set(), {}, {key: 0.0 for key in blue_obs}, 0,
        own_alive={key: True for key in blue_obs},
        enemy_positions={"red_0": np.zeros(3), "red_1": np.zeros(3)},
        enemy_alive={"red_0": True, "red_1": True})
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1"}


def test_blue_pure_pursuit_uses_current_los_and_fixed_speed():
    controller = BluePolicyController(PAPER_BLUE_POLICY_PROFILE)
    controller.reset(["blue_0"], ["red_0"], {}, {})
    elevation = np.arctan2(100.0, np.hypot(100.0, 100.0))
    blue_obs = {"blue_0": _strict_blue_obs(
        [np.sqrt(30000.0)], [np.pi / 4.0], [elevation])}
    actions = controller.act(
        blue_obs, 1, 1, set(), {}, {"blue_0": 0.0}, 0,
        own_alive={"blue_0": True},
        enemy_positions={"red_0": np.array([-999.0, -999.0, -999.0])},
        enemy_alive={"red_0": True})
    action = actions["blue_0"]
    assert action[0] * np.pi / 2.0 == pytest.approx(
        elevation)
    assert action[1] * np.pi == pytest.approx(np.pi / 4.0)
    assert action[2] == pytest.approx((300.0 - 102.0) / 306.0 * 2.0 - 1.0)
    source = Path("my_uav_env/blue_policy_profiles.py").read_text(
        encoding="utf-8").lower()
    assert "lead pursuit" not in source
    assert "prediction" not in source
    assert "np.random" not in source


def test_mws_selects_positive_ttc_and_rejects_receding():
    aircraft = FakeAircraft("red_0", (1000.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    approaching = SimpleNamespace(
        uid="m1", is_alive=True, target_aircraft=aircraft,
        get_position=lambda: np.array([0.0, 0.0, 0.0]),
        get_velocity=lambda: np.array([500.0, 0.0, 0.0]))
    receding = SimpleNamespace(
        uid="m0", is_alive=True, target_aircraft=aircraft,
        get_position=lambda: np.array([0.0, 0.0, 0.0]),
        get_velocity=lambda: np.array([-500.0, 0.0, 0.0]))
    selected, diag = select_most_dangerous_missile(
        aircraft, [receding, approaching])
    assert selected is approaching
    assert diag["closing_speed_mps"] > 0.0
    assert select_most_dangerous_missile(aircraft, [receding]) == (None, None)


def test_mws_selects_minimum_positive_ttc_with_deterministic_tie_break():
    aircraft = FakeAircraft("red_0", (1000.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    slow = SimpleNamespace(
        uid="m1", is_alive=True, target_aircraft=aircraft,
        get_position=lambda: np.array([0.0, 0.0, 0.0]),
        get_velocity=lambda: np.array([500.0, 0.0, 0.0]))
    urgent = SimpleNamespace(
        uid="m0", is_alive=True, target_aircraft=aircraft,
        get_position=lambda: np.array([500.0, 0.0, 0.0]),
        get_velocity=lambda: np.array([600.0, 0.0, 0.0]))
    selected, diag = select_most_dangerous_missile(
        aircraft, [slow, urgent])
    assert selected is urgent
    assert diag["time_to_closest_approach_s"] < 1.0


def test_mws_direction_is_geometric_symmetric_and_deterministic():
    assert UavCombatEnv._mws_turn_direction("red_0", 0.2) == -1.0
    assert UavCombatEnv._mws_turn_direction("blue_0", 0.2) == -1.0
    assert UavCombatEnv._mws_turn_direction("red_0", -0.2) == 1.0
    assert UavCombatEnv._mws_turn_direction("blue_0", -0.2) == 1.0
    assert UavCombatEnv._mws_turn_direction("red_0", 0.0) == 1.0
    assert UavCombatEnv._mws_turn_direction("blue_0", 0.0) == 1.0
    assert UavCombatEnv._mws_turn_direction("red_1", 0.0) == -1.0
    assert UavCombatEnv._mws_turn_direction("blue_1", 0.0) == -1.0


def test_pid_direction_errors_and_first_derivative():
    roll, pitch, *_ = PIDController.paper_direction_errors(
        (0.0, 0.0, 0.0), 0.0, np.pi)
    assert roll == pytest.approx(np.pi)
    assert pitch == pytest.approx(0.0)
    loop = PIDLoop(0.0, 0.0, 1.0, -100.0, 100.0)
    loop.step(10.0, 0.5)
    assert loop._last_diagnostic["d"] == 0.0
    loop.step(11.0, 0.5)
    assert loop._last_diagnostic["d"] == pytest.approx(2.0)


def test_minimal_pid_does_not_add_setpoint_change_derivative_suppression():
    pid = PIDController(
        1 / 60, profile=PAPER_PID_PROFILE,
        throttle_base=0.8, config=PAPER_ENVIRONMENT_CONFIG.pid)
    pid.compute_control((0, 0, 0), 300, 0, 0, 300)
    pid.compute_control((0, 0, 0), 300, 0, 1, 300)
    assert not pid._last_diagnostic["setpoint_changed_this_frame"]
    assert not pid._last_diagnostic[
        "derivative_suppressed_for_setpoint_change"]
    assert pid._last_diagnostic["roll_pid"]["d"] != 0.0


def test_formal_pid_has_only_three_feedback_loops_and_bounded_outputs():
    pid = PIDController(
        1 / 60, profile=PAPER_PID_PROFILE,
        throttle_base=0.8, config=PAPER_ENVIRONMENT_CONFIG.pid)
    loop_names = sorted(
        name for name, value in vars(pid).items()
        if isinstance(value, PIDLoop))
    assert loop_names == ["_pitch_pid", "_roll_pid", "_velocity_pid"]
    controls = pid.compute_control(
        (0.0, 0.0, 0.0), 0.0, np.pi / 2.0, np.pi, 408.0)
    assert all(np.isfinite(controls))
    assert -1.0 <= controls[0] <= 1.0
    assert -1.0 <= controls[1] <= 1.0
    assert controls[2] == 0.0
    assert 0.0 <= controls[3] <= 1.0
    assert all(np.isfinite(loop._integral)
               for loop in (pid._roll_pid, pid._pitch_pid, pid._velocity_pid))
    source = Path("my_uav_env/pid_controller.py").read_text(
        encoding="utf-8").lower()
    for forbidden in (
            "gcas", "rate limiter", "low-pass", "gain scheduling",
            "stall recovery", "overspeed recovery", "automatic pull-up"):
        assert forbidden not in source


def test_missile_creation_frame_guard_then_contact_sample(monkeypatch):
    parent = FakeAircraft("red_0", (0.0, 0.0, 6000.0), color="Red")
    target = FakeAircraft("blue_0", (50.0, 0.0, 6000.0), color="Blue")
    missile = MissileSimulator.create(
        parent, target, "m0", guidance_mode=PAPER_MISSILE_GUIDANCE_MODE,
        config=PAPER_ENVIRONMENT_CONFIG.missile,
        rng=np.random.default_rng(0), launch_speed_mps=800.0,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    monkeypatch.setattr(missile, "_roll_hit_probability", lambda: True)
    missile.run()
    assert missile.is_alive
    assert missile._pn_guidance_frames == 1
    missile.run()
    assert missile.is_success
    assert missile._pn_guidance_frames == 1


@pytest.mark.parametrize("distance", [1000.0, 3000.0, 5000.0, 8000.0, 10000.0])
def test_missile_trajectory_is_finite_constant_speed_and_limited(distance):
    parent = FakeAircraft("red_0", (0.0, 0.0, 6000.0), color="Red")
    target = FakeAircraft(
        "blue_0", (distance, 500.0, 6200.0), (250.0, -20.0, 0.0), color="Blue")
    missile = MissileSimulator.create(
        parent, target, "m0", guidance_mode=PAPER_MISSILE_GUIDANCE_MODE,
        config=PAPER_ENVIRONMENT_CONFIG.missile,
        rng=np.random.default_rng(1), launch_speed_mps=800.0,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    for _ in range(30):
        if missile.is_done:
            break
        target._position += target._velocity * target.dt
        missile.run()
        assert np.all(np.isfinite(missile.get_position()))
        assert np.linalg.norm(missile.get_velocity()) == pytest.approx(800.0)
        assert missile._maximum_command_g <= 30.0 + 1e-9


def test_hit_probability_direction():
    parent = FakeAircraft("red_0", (0.0, 0.0, 6000.0), color="Red")
    target = FakeAircraft("blue_0", (1000.0, 0.0, 6000.0), color="Blue")
    missile = MissileSimulator.create(
        parent, target, "m0", guidance_mode=PAPER_MISSILE_GUIDANCE_MODE,
        config=PAPER_ENVIRONMENT_CONFIG.missile,
        launch_speed_mps=800.0)
    assert missile._hit_probability() == pytest.approx(1.0)
    missile._velocity *= -1.0
    assert missile._hit_probability() == pytest.approx(0.05)


@pytest.mark.parametrize("angle, expected", [
    (0.0, 1.0), (3.999999, 1.0),
    (4.0, 1.0 + 22.0 / 15.0),
    (15.0, 1.0), (15.000001, 0.99999995),
    (35.0, 0.0), (36.0, 0.0)])
def test_ta_reward_boundaries(angle, expected):
    assert ta_angle_advantage_fixed(angle) == pytest.approx(expected)


def test_formal_eq17_tail_and_engineering_thresholds():
    config = _minimal_altitude_reward_config()
    assert config.version == "eq17_minimal_finite_tail01_v2"
    assert config.high_altitude_tail == pytest.approx(0.1)
    assert config.h_min_m == pytest.approx(0.0)
    assert config.h_att_m == pytest.approx(2000.0)
    assert config.h_adv_m == pytest.approx(5000.0)
    assert config.h_max_m == pytest.approx(10000.0)
    assert config.d_att_max_m == pytest.approx(10000.000001)
    assert altitude_reward_paper_eq17(
        10000.0000005, config) == pytest.approx(0.1)


def test_reward_formula_change_preserves_nonreward_runtime(monkeypatch):
    import my_uav_env.env as env_module

    current_ta = env_module.ta_angle_advantage_fixed
    legacy_altitude = AltitudeRewardConfig(
        version="eq17_minimal_finite_tail_v1",
        h_min_m=0.0,
        h_att_m=2000.0,
        h_adv_m=5000.0,
        h_max_m=10000.0,
        d_att_max_m=10000.000001,
        high_altitude_tail=0.0,
    )

    def legacy_ta(q_los_deg):
        if abs(float(q_los_deg)) < 4.0:
            return 10.0
        return current_ta(q_los_deg)

    def run_once(altitude_config, ta_function):
        monkeypatch.setattr(env_module, "ta_angle_advantage_fixed", ta_function)
        env = UavCombatEnv(altitude_reward_config=altitude_config)
        try:
            env.reset(seed=3)
            actions = {
                aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            obs, _rewards, terminated, truncated, _info = env.step(actions)
            return {
                "obs": copy.deepcopy(obs),
                "terminated": dict(terminated),
                "truncated": dict(truncated),
                "aircraft": {
                    aid: {
                        "position": env._get_sim(aid).get_position(),
                        "velocity": env._get_sim(aid).get_velocity(),
                        "rpy": env._get_sim(aid).get_rpy(),
                        "alive": env._get_sim(aid).is_alive,
                        "missiles_left": env._get_sim(aid).num_left_missiles,
                    }
                    for aid in env.agent_ids
                },
                "death_reasons": dict(env._death_reasons),
                "missile_ids": sorted(env._missiles_in_flight),
                "engaged_targets": sorted(env._engaged_targets),
                "fire_control": {
                    aid: copy.deepcopy(vars(state))
                    for aid, state in env._fire_control_states.items()
                },
                "rng_state": copy.deepcopy(env.np_random.bit_generator.state),
            }
        finally:
            env.close()

    legacy = run_once(legacy_altitude, legacy_ta)
    current = run_once(_minimal_altitude_reward_config(), current_ta)

    assert legacy.keys() == current.keys()
    for aid in legacy["obs"]:
        assert legacy["obs"][aid].keys() == current["obs"][aid].keys()
        for field in legacy["obs"][aid]:
            np.testing.assert_array_equal(
                legacy["obs"][aid][field], current["obs"][aid][field])
        for field in ("position", "velocity", "rpy"):
            np.testing.assert_array_equal(
                legacy["aircraft"][aid][field],
                current["aircraft"][aid][field])
        assert legacy["aircraft"][aid]["alive"] == (
            current["aircraft"][aid]["alive"])
        assert legacy["aircraft"][aid]["missiles_left"] == (
            current["aircraft"][aid]["missiles_left"])
    for field in (
            "terminated", "truncated", "death_reasons", "missile_ids",
            "engaged_targets", "fire_control", "rng_state"):
        assert legacy[field] == current[field]


def test_active_alignment_docs_have_no_superseded_ta_or_q_los_claims():
    paths = (
        Path("docs/current_environment_alignment_status.md"),
        Path("docs/experiment_guide.md"),
        Path("docs/paper_alignment_audit_vanilla_baseline.md"),
        Path("docs/paper_env_reward_audit.md"),
        Path("docs/reward_formula_alignment_plan.md"),
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for stale in (
            "Ta=10", "Ta = 10", "10-scale Ta", "3D body-x q_LOS",
            "paper_eq20_ta_alt_eq17_3dlos_v1",
            "fixed_ta_alt_eq17_3dlos_v1"):
        assert stale not in text
    assert "paper_literal_eq15_eq20_ta1_tail01_joint_v4" in text
    assert "UNRESOLVED / PAPER_INFERRED" in text
    assert "velocity-to-LOS" in text


@pytest.mark.parametrize("distance, expected", [
    (0.0, 1.0), (15000.0, 1.0),
    (16000.0, np.exp(1.0 - 16000.0 / 15000.0))])
def test_td_reward_uses_metres(distance, expected):
    assert td_distance_advantage(distance) == pytest.approx(expected)


def test_checkpoint_fingerprint_rejects_old_environment():
    config = Config()
    config.seed = 3
    expected = _checkpoint_metadata(config, 60, 30)
    assert expected["missile_hit_radius_m"] == pytest.approx(100.0)
    payload = {"state_dict": {}, "model_kind": "actor",
               "metadata": dict(expected)}
    payload["metadata"]["environment_config_fingerprint"] = (
        "3686728c40163c88388932de0beb132e15024687879e03f982fd4715aa6d4d9d")
    with pytest.raises(ValueError):
        _unpack_and_validate_checkpoint(payload, expected, "actor")


def test_actor_evaluation_ignores_seed_budget_and_rollout_layout():
    payload, config = _evaluation_actor_payload()
    metadata = payload["metadata"]
    assert config.seed == 3
    assert metadata["total_env_steps"] == 100_000
    assert metadata["rollout_horizon_per_env"] == 62
    assert metadata["transitions_per_update"] == 1984
    assert metadata["unused_replay_slots"] == 16

    state, restored, altitude_config, hidden, rnn_hidden = (
        _unpack_evaluation_actor(payload))

    assert state is payload["state_dict"]
    assert restored["environment_config"]["seed"] == 3
    assert altitude_config == _minimal_altitude_reward_config()
    assert hidden == config.mlp_hidden
    assert rnn_hidden == config.rnn_hidden_size


def test_actor_evaluation_ignores_num_envs_and_buffer_metadata():
    payload, _ = _evaluation_actor_payload()
    payload["metadata"].update({
        "num_envs": 32,
        "replay_buffer_size": 2000,
        "requested_replay_buffer_size": 2000,
    })
    _unpack_evaluation_actor(payload)


def test_actor_evaluation_restores_checkpoint_altitude_reward_config():
    payload, _ = _evaluation_actor_payload()
    _, _, restored, _, _ = _unpack_evaluation_actor(payload)
    assert restored.version == "eq17_minimal_finite_tail01_v2"
    assert restored.d_att_max_m == pytest.approx(10000.000001)
    assert restored.high_altitude_tail == pytest.approx(0.1)
    assert restored != DEFAULT_ALTITUDE_REWARD_CONFIG

    env = UavCombatEnv(altitude_reward_config=restored)
    try:
        assert env.altitude_reward_config is restored
        assert env.altitude_reward_config.version == (
            "eq17_minimal_finite_tail01_v2")
    finally:
        env.close()


@pytest.mark.parametrize("field, incompatible", [
    ("environment_config_fingerprint", "incompatible"),
    ("reward_version", "paper_literal_eq15_eq20_joint_v3"),
    ("action_distribution", "incompatible"),
    ("pid_profile", "incompatible"),
    ("reward_mode", "incompatible"),
    ("missile_guidance_mode", "incompatible"),
    ("blue_policy_profile", "incompatible"),
    ("mws_profile", "incompatible"),
    ("sensor_support_profile", "incompatible"),
    ("missile_model_scope", "incompatible"),
])
def test_actor_evaluation_rejects_runtime_semantic_mismatch(
        field, incompatible):
    payload, _ = _evaluation_actor_payload()
    payload = copy.deepcopy(payload)
    payload["metadata"][field] = incompatible
    with pytest.raises(ValueError, match=field):
        _unpack_evaluation_actor(payload)


def test_actor_evaluation_rejects_obs_and_action_shape_mismatch():
    payload, _ = _evaluation_actor_payload()
    with pytest.raises(ValueError, match="actor_obs_dim|obs_dim"):
        _unpack_evaluation_actor(payload, obs_dim=50)

    action_payload = copy.deepcopy(payload)
    action_payload["state_dict"] = VanillaActor(
        obs_dim=60, action_dim=4, hidden=128, rnn_hidden=128).state_dict()
    with pytest.raises(ValueError, match="action_dim"):
        _unpack_evaluation_actor(action_payload)


@pytest.mark.parametrize("field, incompatible, message", [
    ("actor_hidden_sizes", [64, 64], "MLP hidden"),
    ("actor_rnn_hidden_size", 64, "GRU hidden"),
])
def test_actor_evaluation_rejects_metadata_network_shape_mismatch(
        field, incompatible, message):
    payload, _ = _evaluation_actor_payload()
    payload["metadata"][field] = incompatible
    with pytest.raises(ValueError, match=message):
        _unpack_evaluation_actor(payload)


def test_actor_evaluation_rejects_unsupported_schema_and_missing_state():
    payload, _ = _evaluation_actor_payload()
    incompatible = copy.deepcopy(payload)
    incompatible["metadata"]["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema_version"):
        _unpack_evaluation_actor(incompatible)

    missing_state = copy.deepcopy(payload)
    missing_state.pop("state_dict")
    with pytest.raises(ValueError, match="state_dict"):
        _unpack_evaluation_actor(missing_state)


def test_actor_evaluation_requires_altitude_reward_config():
    payload, _ = _evaluation_actor_payload()
    payload["metadata"].pop("altitude_reward_config")
    with pytest.raises(ValueError, match="altitude_reward_config"):
        _unpack_evaluation_actor(payload)


def test_actor_evaluation_rejects_old_eq17_zero_tail_config():
    payload, _ = _evaluation_actor_payload()
    payload["metadata"]["altitude_reward_config"] = {
        "version": "eq17_minimal_finite_tail_v1",
        "h_min_m": 0.0,
        "h_att_m": 2000.0,
        "h_adv_m": 5000.0,
        "h_max_m": 10000.0,
        "d_att_max_m": 10000.000001,
        "high_altitude_tail": 0.0,
    }
    with pytest.raises(ValueError, match="altitude_reward_config"):
        _unpack_evaluation_actor(payload)


def test_evaluate_and_acmi_import_the_shared_evaluation_contract():
    evaluate_source = Path("evaluate_vanilla_mappo.py").read_text(
        encoding="utf-8")
    acmi_source = Path("eval_acmi.py").read_text(encoding="utf-8")
    helper = "_unpack_actor_checkpoint_for_evaluation"
    assert helper in evaluate_source
    assert helper in acmi_source
    assert "_unpack_and_validate_checkpoint(" not in evaluate_source
    assert "_unpack_and_validate_checkpoint(" not in acmi_source


def test_training_resume_contract_still_rejects_rollout_mismatch():
    _, config = _evaluation_actor_payload()
    metadata = _checkpoint_metadata(config, 60, 30)
    payload = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "checkpoint_metadata": metadata,
        "core_config": _training_core_config(config, metadata),
        "runtime": {"total_steps": 100_000},
    }
    payload["core_config"]["replay_buffer_size"] = 500
    with pytest.raises(ValueError, match="core configuration mismatch"):
        _validate_training_state(payload, config, metadata)


def test_training_resume_accepts_new_reward_and_rejects_old_reward_version():
    _, config = _evaluation_actor_payload()
    metadata = _checkpoint_metadata(config, 60, 30)
    payload = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "checkpoint_metadata": dict(metadata),
        "core_config": _training_core_config(config, metadata),
        "runtime": {"total_steps": 100_000},
    }
    _validate_training_state(payload, config, metadata)

    old = copy.deepcopy(payload)
    old["checkpoint_metadata"]["reward_version"] = (
        "paper_literal_eq15_eq20_joint_v3")
    old["core_config"]["reward_version"] = (
        "paper_literal_eq15_eq20_joint_v3")
    with pytest.raises(ValueError, match="core configuration mismatch"):
        _validate_training_state(old, config, metadata)


def test_minimal_environment_rejects_old_profile_training_and_actor_state():
    payload, config = _evaluation_actor_payload()
    old_actor = copy.deepcopy(payload)
    old_actor["metadata"]["environment_profile"] = "paper_3v3_v1"
    old_actor["metadata"]["environment_version"] = "paper_3v3_v1"
    old_actor["metadata"]["environment_config_fingerprint"] = (
        "472a29e9da6bc6e96f7b9dac77889d3f7fcfa055835fca19d8efffe4f0b9d7b0")
    with pytest.raises(
            ValueError, match="environment_profile|environment_version|fingerprint"):
        _unpack_evaluation_actor(old_actor)

    metadata = _checkpoint_metadata(config, 60, 30)
    training_state = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "checkpoint_metadata": dict(metadata),
        "core_config": _training_core_config(config, metadata),
        "runtime": {"total_steps": 100_000},
    }
    old_training_state = copy.deepcopy(training_state)
    old_training_state["core_config"]["environment_profile"] = "paper_3v3_v1"
    old_training_state["core_config"]["environment_config_fingerprint"] = (
        "472a29e9da6bc6e96f7b9dac77889d3f7fcfa055835fca19d8efffe4f0b9d7b0")
    with pytest.raises(ValueError, match="core configuration mismatch"):
        _validate_training_state(old_training_state, config, metadata)

    _unpack_evaluation_actor(payload)
    _validate_training_state(training_state, config, metadata)


def test_random_evaluation_does_not_resolve_or_validate_checkpoint():
    import torch
    from evaluate_vanilla_mappo import _load_actor

    args = SimpleNamespace(random=True, checkpoint="does-not-exist.pt")
    actor, rnn_hidden, checkpoint, altitude_config = _load_actor(
        args, torch.device("cpu"))
    assert actor is None
    assert rnn_hidden == 128
    assert checkpoint is None
    assert altitude_config == _minimal_altitude_reward_config()


def test_speed_envelope_is_monitor_only(monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        sim = env.red_planes["red_0"]
        monkeypatch.setattr(
            sim, "get_velocity", lambda: np.array([601.0, 0.0, 0.0]))
        env._check_aircraft_post_physics_state("red_0", sim)
        env._check_aircraft_post_physics_state("red_0", sim)
        assert sim.is_alive
        env._check_aircraft_post_physics_state("red_0", sim)
        assert sim.is_alive
        diag = env._aircraft_diagnostics["red_0"]
        assert diag["speed_envelope_violation"]
        assert diag["overspeed_frames"] == 3
        assert diag["maximum_consecutive_above_600mps_frames"] == 3
        assert "red_0" not in env._death_reasons
        assert not env._invalid_numerical_episode
        monkeypatch.setattr(
            sim, "get_velocity", lambda: np.array([300.0, 0.0, 0.0]))
        env._check_aircraft_post_physics_state("red_0", sim)
        assert diag["consecutive_above_600mps_frames"] == 0
    finally:
        env.close()


def test_overload_envelope_is_monitor_only(monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        original = sim.get_property_value
        load = {"value": 10.0}

        def property_value(name):
            if name == "accelerations/n-pilot-x-norm":
                return load["value"]
            if name in ("accelerations/n-pilot-y-norm",
                        "accelerations/n-pilot-z-norm"):
                return 0.0
            return original(name)

        monkeypatch.setattr(sim, "get_property_value", property_value)
        target = (0.0, float(sim.get_rpy()[2]), 300.0)
        env._apply_pid_controls({aid: target})
        load["value"] = 8.0
        env._apply_pid_controls({aid: target})
        assert sim.is_alive
        load["value"] = 10.0
        env._apply_pid_controls({aid: target})
        env._apply_pid_controls({aid: target})
        assert sim.is_alive
        env._apply_pid_controls({aid: target})
        assert sim.is_alive
        diag = env._aircraft_diagnostics[aid]
        assert diag["overload_envelope_violation"]
        assert diag["frames_above_9g"] == 4
        assert diag["maximum_consecutive_above_9g_frames"] == 3
        assert aid not in env._death_reasons
        assert not env._invalid_numerical_episode
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["catastrophic", "persistent", "nonfinite"])
def test_load_numerical_invalid_remains_terminal(monkeypatch, mode):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        original = sim.get_property_value
        value = {"catastrophic": 101.0, "persistent": 31.0,
                 "nonfinite": float("nan")}[mode]

        def property_value(name):
            if name == "accelerations/n-pilot-x-norm":
                return value
            if name in ("accelerations/n-pilot-y-norm",
                        "accelerations/n-pilot-z-norm"):
                return 0.0
            return original(name)

        monkeypatch.setattr(sim, "get_property_value", property_value)
        target = (0.0, float(sim.get_rpy()[2]), 300.0)
        repeats = 3 if mode == "persistent" else 1
        for _ in range(repeats):
            env._apply_pid_controls({aid: target})
        assert not sim.is_alive
        assert env._invalid_numerical_episode
        expected = {
            "catastrophic": "CatastrophicFiniteLoad",
            "persistent": "PersistentExtremeFiniteLoad",
            "nonfinite": "NonFiniteLoad",
        }[mode]
        assert any(expected in reason for reason in env._invalid_numerical_reasons)
    finally:
        env.close()


def test_nonfinite_aircraft_state_remains_numerical_invalid(monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        sim = env.red_planes["red_0"]
        monkeypatch.setattr(
            sim, "get_velocity", lambda: np.array([np.nan, 0.0, 0.0]))
        env._check_aircraft_post_physics_state("red_0", sim)
        assert not sim.is_alive
        assert env._invalid_numerical_episode
        assert any("NonFiniteState" in reason
                   for reason in env._invalid_numerical_reasons)
    finally:
        env.close()


@pytest.mark.parametrize("altitude_m, expected_alive", [
    (100.1, True),
    (100.0, False),
    (99.9, False),
])
def test_low_altitude_hard_boundary(monkeypatch, altitude_m, expected_alive):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        monkeypatch.setattr(
            sim, "get_geodetic",
            lambda: np.array([120.0, 60.0, altitude_m]))

        result = env._check_aircraft_post_physics_state(aid, sim)

        assert result is expected_alive
        assert sim.is_alive is expected_alive
        if expected_alive:
            assert aid not in env._death_reasons
        else:
            assert env._death_reasons[aid] == "Crash_LowAlt"
        assert not env._invalid_numerical_episode
        assert not env._invalid_numerical_reasons
    finally:
        env.close()


@pytest.mark.parametrize("altitude_m, expected_alive", [
    (9999.9, True),
    (10000.0, False),
    (10000.1, False),
])
def test_high_altitude_hard_boundary(monkeypatch, altitude_m, expected_alive):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        monkeypatch.setattr(
            sim, "get_geodetic",
            lambda: np.array([120.0, 60.0, altitude_m]))

        result = env._check_aircraft_post_physics_state(aid, sim)

        assert result is expected_alive
        assert sim.is_alive is expected_alive
        if expected_alive:
            assert aid not in env._death_reasons
        else:
            assert env._death_reasons[aid] == "Crash_HighAlt"
        assert not env._invalid_numerical_episode
    finally:
        env.close()


@pytest.mark.parametrize("north_m, east_m, expected_alive", [
    (49999.9, 0.0, True),
    (50000.0, 0.0, False),
    (-50000.0, 0.0, False),
    (0.0, 50000.0, False),
])
def test_horizontal_hard_boundary(
        monkeypatch, north_m, east_m, expected_alive):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        monkeypatch.setattr(
            sim, "get_position",
            lambda: np.array([north_m, east_m, 6000.0]))

        result = env._check_aircraft_post_physics_state(aid, sim)

        assert result is expected_alive
        assert sim.is_alive is expected_alive
        if expected_alive:
            assert aid not in env._death_reasons
        else:
            assert env._death_reasons[aid] == "Crash_BattleVolume"
        assert not env._invalid_numerical_episode
    finally:
        env.close()


@pytest.mark.parametrize("north_m, expected_penalty", [
    (39999.9, 0.0),
    (40000.0, 0.0),
    (40000.1, -10.0),
    (49999.9, -10.0),
])
def test_eq18_penalty_precedes_hard_boundary(
        monkeypatch, north_m, expected_penalty):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        monkeypatch.setattr(
            sim, "get_position",
            lambda: np.array([north_m, 0.0, 6000.0]))

        assert env._boundary_penalty(sim) == expected_penalty
        assert env._check_aircraft_post_physics_state(aid, sim)
        assert sim.is_alive
    finally:
        env.close()


def test_mid_frame_ground_crossing_is_classified_before_next_load_read(
        monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        state = {"altitude": 100.1, "runs": 0,
                 "load_reads_after_ground": 0}
        original_property = sim.get_property_value

        monkeypatch.setattr(
            sim, "get_position",
            lambda: np.array([0.0, 0.0, state["altitude"]]))
        monkeypatch.setattr(
            sim, "get_geodetic",
            lambda: np.array([120.0, 60.0, state["altitude"]]))

        def run():
            state["runs"] += 1
            state["altitude"] = 99.9
            return True

        def property_value(name):
            if (state["altitude"] <= 100.0
                    and name.startswith("accelerations/n-pilot-")):
                state["load_reads_after_ground"] += 1
                return 500.0
            return original_property(name)

        monkeypatch.setattr(sim, "run", run)
        monkeypatch.setattr(sim, "get_property_value", property_value)
        env._run_one_physics_frame()

        assert state["runs"] == 1
        assert not sim.is_alive
        assert env._death_reasons[aid] == "Crash_LowAlt"
        assert not env._invalid_numerical_episode
        target = (0.0, 0.0, 300.0)
        env._apply_pid_controls({aid: target})
        env._update_overload_timers()
        env._run_one_physics_frame()
        assert state["runs"] == 1
        assert state["load_reads_after_ground"] == 0

        rewards, components = env._compute_rewards()
        assert all(np.isfinite(value) for value in rewards.values())
        assert components[aid]["r_death"] == 0.0
        assert env._get_terminated()[aid]
        assert not any(env._get_truncated().values())
    finally:
        env.close()


def test_dead_aircraft_stops_mid_decision_while_survivor_finishes_frames(
        monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        dead_id = "red_0"
        dead = env.red_planes[dead_id]
        survivor = env.red_planes["red_1"]
        state = {"altitude": 100.1, "dead_runs": 0, "survivor_runs": 0,
                 "load_reads_after_ground": 0}
        original_property = dead.get_property_value

        monkeypatch.setattr(
            dead, "get_position",
            lambda: np.array([0.0, 0.0, state["altitude"]]))
        monkeypatch.setattr(
            dead, "get_geodetic",
            lambda: np.array([120.0, 60.0, state["altitude"]]))

        def dead_run():
            state["dead_runs"] += 1
            if state["dead_runs"] == 3:
                state["altitude"] = 99.9
            return True

        def survivor_run():
            state["survivor_runs"] += 1
            return True

        def property_value(name):
            if (state["altitude"] <= 100.0
                    and name.startswith("accelerations/n-pilot-")):
                state["load_reads_after_ground"] += 1
                return 500.0
            return original_property(name)

        monkeypatch.setattr(dead, "run", dead_run)
        monkeypatch.setattr(dead, "get_property_value", property_value)
        monkeypatch.setattr(survivor, "run", survivor_run)
        actions = {
            aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _, rewards, _, truncated, info = env.step(actions)

        assert state["dead_runs"] == 3
        assert state["survivor_runs"] == env.agent_interaction_steps
        assert state["load_reads_after_ground"] == 0
        assert env._death_reasons[dead_id] == "Crash_LowAlt"
        assert not env._invalid_numerical_episode
        assert all(np.isfinite(value) for value in rewards.values())
        assert not any(truncated.values())
        assert not info["__episode__"]["invalid_numerical_episode"]
    finally:
        env.close()


def test_real_jsbsim_low_dive_crosses_ground_as_normal_crash():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        initial_state = dict(sim.init_state)
        initial_state.update({
            "ic/h-sl-ft": 120.0 / 0.3048,
            "ic/theta-deg": -30.0,
            "ic/u-fps": 300.0 / 0.3048,
        })
        sim.reload(new_state=initial_state)

        previous_altitude = float(sim.get_geodetic()[2])
        for _ in range(30):
            env._run_one_physics_frame()
            if not sim.is_alive:
                break
            previous_altitude = float(sim.get_geodetic()[2])

        assert not sim.is_alive
        crash_altitude = float(sim.get_geodetic()[2])
        assert previous_altitude > 100.0
        assert 0.0 < crash_altitude <= 100.0
        assert env._death_reasons[aid] == "Crash_LowAlt"
        assert not env._invalid_numerical_episode
        assert not env._invalid_numerical_reasons
        for other_id in env.agent_ids:
            if other_id == aid:
                continue
            other = env._get_sim(other_id)
            assert other.is_alive
            assert np.all(np.isfinite(np.concatenate([
                other.get_position(), other.get_velocity(), other.get_rpy()])))
    finally:
        env.close()


def test_fire_control_lock_cooldown_and_same_target_deconfliction(monkeypatch):
    import my_uav_env.env as env_module

    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        for aid in ("red_1", "red_2", "blue_1", "blue_2"):
            env._get_sim(aid).crash()
        monkeypatch.setattr(
            env, "_is_detected_by_electro_optical", lambda *_: True)
        monkeypatch.setattr(env_module, "compute_body_x_q_los", lambda *_: 0.0)
        monkeypatch.setattr(
            env_module, "get2d_heading_AO_TA_R",
            lambda *_: (0.0, np.pi, 1000.0))
        monkeypatch.setattr(env_module, "compute_3d_range", lambda *_: 1000.0)
        launches = []

        def launch(parent, target, quality):
            launches.append((parent.uid, target.uid))
            env._missile_cooldown[parent.uid] = env.missile_cooldown_frames
            parent.num_left_missiles -= 1

        monkeypatch.setattr(env, "_launch_missile", launch)
        for _ in range(env.missile_lock_delay_frames - 1):
            env._check_missile_launch()
        assert launches == []
        env._check_missile_launch()
        assert launches == [("blue_0", "red_0"), ("red_0", "blue_0")]
        assert env._engaged_targets == {"red_0", "blue_0"}
        env._engaged_targets.clear()
        env._check_missile_launch()
        assert len(launches) == 2
        assert env._missile_cooldown["red_0"] == env.missile_cooldown_frames - 1
    finally:
        env.close()


def test_initial_state_is_deterministic_head_on_10km():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        first = {aid: env._get_sim(aid).get_position().copy()
                 for aid in env.agent_ids}
        env.reset(seed=23)
        second = {aid: env._get_sim(aid).get_position().copy()
                  for aid in env.agent_ids}
        for aid in env.agent_ids:
            np.testing.assert_allclose(first[aid], second[aid])
        for index in range(3):
            distance = np.linalg.norm(
                first[f"red_{index}"] - first[f"blue_{index}"])
            assert distance == pytest.approx(10000.0, abs=1e-2)
    finally:
        env.close()


def test_coarse_enemy_track_is_deterministically_quantized(monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        observer = env.red_planes["red_0"]
        target = env.blue_planes["blue_0"]
        monkeypatch.setattr(env, "_is_detected_by_radar", lambda *_: False)
        monkeypatch.setattr(
            target, "get_position", lambda: np.array([1234.0, -876.0, 6123.0]))
        track = env._get_sensor_track(observer, target)
        assert track.source == "quantized_position_only"
        np.testing.assert_allclose(
            track.position_estimate, [1000.0, -1000.0, 6000.0])
        assert not track.velocity_available
        assert not track.relative_velocity_available
    finally:
        env.close()


def test_both_teams_mws_override_with_same_semantics():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        red = env.red_planes["red_0"]
        blue = env.blue_planes["blue_0"]
        env._launch_missile(red, blue)
        env._launch_missile(blue, red)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        targets = env._parse_actions(actions)
        assert env._mws_enabled_for_agent("red_0")
        assert env._mws_enabled_for_agent("blue_0")
        assert env._learnable_command_sources["red_0"] == "mws_override"
        assert env._learnable_command_sources["blue_0"] == "mws_override"
        red_delta = abs((targets["red_0"][1] - red.get_rpy()[2] + np.pi)
                        % (2 * np.pi) - np.pi)
        blue_delta = abs((targets["blue_0"][1] - blue.get_rpy()[2] + np.pi)
                         % (2 * np.pi) - np.pi)
        assert np.rad2deg(red_delta) == pytest.approx(60.0)
        assert np.rad2deg(blue_delta) == pytest.approx(60.0)
    finally:
        env.close()


def test_mws_no_threat_immediately_restores_base_action(monkeypatch):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        aid = "red_0"
        sim = env.red_planes[aid]
        incoming = SimpleNamespace(
            uid="blue_missile_test",
            get_position=lambda: sim.get_position() + np.array([1000.0, 0.0, 0.0]))
        warning = {"value": (incoming, {
            "closing_speed_mps": 500.0,
            "time_to_closest_approach_s": 2.0,
            "candidate_is_approaching": True,
        })}
        monkeypatch.setattr(
            sim, "get_missile_warning_diagnostic", lambda: warning["value"])
        actions = {aid: np.zeros(3, dtype=np.float32)}
        rng_before = copy.deepcopy(env.np_random.bit_generator.state)
        evasion = env._parse_actions(actions)[aid]
        assert env.np_random.bit_generator.state == rng_before
        assert env._learnable_command_sources[aid] == "mws_override"
        assert evasion[2] == pytest.approx(300.0)

        warning["value"] = (None, None)
        base = env._parse_actions(actions)[aid]
        assert env.np_random.bit_generator.state == rng_before
        assert env._learnable_command_sources[aid] == "base_policy"
        assert base == pytest.approx((0.0, 0.0, 255.0))
        assert env._learnable_mws_state[aid]["active_missile_uid"] is None
    finally:
        env.close()


@pytest.mark.parametrize("aid, team, other", [
    ("red_0", "red", "blue"),
    ("blue_0", "blue", "red"),
])
def test_mws_is_not_direction_latched_and_statistics_are_team_scoped(
        aid, team, other):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        incoming = SimpleNamespace(uid=f"{other}_missile_test")
        first = env._mws_evasion_target(aid, incoming, 0.0, 1.0)
        second = env._mws_evasion_target(aid, incoming, 0.2, -1.0)
        diag = env._mws_decision_diagnostics
        assert diag[f"{team}_warning_generations"] == 1
        assert diag[f"{team}_suppressed_direction_flip_attempts"] == 0
        assert diag[f"{team}_direction_changes_within_same_missile"] == 1
        assert diag[f"{team}_maximum_continuous_decisions"] == 2
        assert diag[f"{team}_target_heading_delta_max_deg"] == pytest.approx(60.0)
        assert first[1] == pytest.approx(np.deg2rad(60.0))
        assert second[1] == pytest.approx(0.2 - np.deg2rad(60.0))
        assert diag[f"{other}_warning_generations"] == 0
        assert diag[f"{other}_suppressed_direction_flip_attempts"] == 0
        assert diag[f"{other}_maximum_continuous_decisions"] == 0
    finally:
        env.close()


def test_engaged_target_released_after_last_missile_cleanup():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        red = env.red_planes["red_0"]
        blue = env.blue_planes["blue_0"]
        env._launch_missile(red, blue)
        missile = next(iter(env._missiles_in_flight.values()))
        env.refresh_engaged_targets()
        assert "blue_0" in env._engaged_targets
        missile._status = MissileSimulator.MISS
        missile._termination_reason = "timeout"
        env._cleanup_missiles()
        assert "blue_0" not in env.refresh_engaged_targets()
    finally:
        env.close()


def test_joint_reward_is_identical_within_each_team_and_finite():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        rewards, components = env._compute_rewards()
        assert len({rewards[aid] for aid in env.red_ids}) == 1
        assert len({rewards[aid] for aid in env.blue_ids}) == 1
        assert all(np.isfinite(value) for value in rewards.values())
        assert all("raw_r_adv" in components[aid] for aid in env.agent_ids)
        sim = env.red_planes["red_0"]
        original = sim.get_position
        sim.get_position = lambda: np.array([40001.0, 0.0, 6000.0])
        assert env._boundary_penalty(sim) == -10.0
        sim.get_position = original
    finally:
        env.close()


def test_seed3_reset_reward_components_match_pre_simplification_snapshot():
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        rewards, components = env._compute_rewards()
        assert rewards["red_0"] == pytest.approx(0.4481508138835726)
        assert rewards["blue_0"] == pytest.approx(0.44815081388362965)
        expected_raw_adv = {
            "red_0": 0.9461188332581016,
            "red_1": 1.0954346329103632,
            "red_2": 0.946118626388383,
            "blue_0": 0.9461188332584443,
            "blue_1": 1.0954346329108664,
            "blue_2": 0.9461186263876147,
        }
        expected_raw_alt = {
            "red_0": 1.1356841393232269e-12,
            "red_1": 0.0,
            "red_2": 0.0,
            "blue_0": 1.5140481461154802e-12,
            "blue_1": 0.0,
            "blue_2": 7.567280135845067e-13,
        }
        expected_fields = {
            "raw_r_pitch", "raw_r_roll", "raw_r_alt", "raw_r_bound",
            "raw_r_vel", "raw_r_adv", "r_pitch", "r_roll", "r_alt",
            "r_bound", "r_vel", "r_adv", "r_death", "r_end",
            "local_reward", "joint_reward", "reward_mode", "reward_version",
            "red_local_reward_sum", "blue_local_reward_sum",
            "red_team_terminal_reward", "blue_team_terminal_reward",
            "red_joint_reward", "blue_joint_reward",
        }
        for aid, expected in expected_raw_adv.items():
            assert set(components[aid]) == expected_fields
            assert components[aid]["raw_r_adv"] == pytest.approx(expected)
            assert components[aid]["raw_r_alt"] == pytest.approx(
                expected_raw_alt[aid], abs=1e-15)
            assert components[aid]["r_adv"] == pytest.approx(0.15 * expected)
            assert components[aid]["r_alt"] == pytest.approx(
                0.04 * expected_raw_alt[aid], abs=1e-16)
            assert components[aid]["local_reward"] == pytest.approx(
                components[aid]["r_adv"] + components[aid]["r_alt"])
            assert components[aid]["reward_version"] == REWARD_VERSION
            for field in ("raw_r_pitch", "raw_r_roll", "raw_r_bound",
                          "raw_r_vel", "r_death", "r_end"):
                assert components[aid][field] == pytest.approx(0.0)
    finally:
        env.close()


def test_minimal_profile_repeated_fixed_action_step_is_deterministic():
    def run_once():
        env = UavCombatEnv()
        try:
            obs, _ = env.reset(seed=3)
            blue_actions = env.blue_policy_actions(
                {aid: obs[aid] for aid in env.blue_ids})
            actions = {
                aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids}
            actions.update(blue_actions)
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            return {
                "blue_actions": copy.deepcopy(blue_actions),
                "obs": copy.deepcopy(next_obs),
                "rewards": dict(rewards),
                "terminated": dict(terminated),
                "truncated": dict(truncated),
                "rng": copy.deepcopy(env.np_random.bit_generator.state),
            }
        finally:
            env.close()

    first = run_once()
    second = run_once()
    assert first.keys() == second.keys()
    for aid in first["blue_actions"]:
        np.testing.assert_array_equal(
            first["blue_actions"][aid], second["blue_actions"][aid])
    for aid in first["obs"]:
        for field in first["obs"][aid]:
            np.testing.assert_array_equal(
                first["obs"][aid][field], second["obs"][aid][field])
    for field in ("rewards", "terminated", "truncated", "rng"):
        assert first[field] == second[field]
