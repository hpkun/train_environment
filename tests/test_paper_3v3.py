from __future__ import annotations

from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.paper_3v3_spec import (
    PAPER_BLUE_POLICY_PROFILE,
    PAPER_ENVIRONMENT_CONFIG,
    PAPER_ENVIRONMENT_PROFILE,
    PAPER_MISSILE_GUIDANCE_MODE,
    PAPER_PID_PROFILE,
    PAPER_UNSPECIFIED_ENGINEERING,
    paper_environment_snapshot,
)
from my_uav_env.alignment.reward_utils import (
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
    Config,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _flatten_obs,
    _unpack_and_validate_checkpoint,
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
    assert snapshot["environment_config_fingerprint"] != (
        "3686728c40163c88388932de0beb132e15024687879e03f982fd4715aa6d4d9d")
    assert snapshot["environment_config"]["missile"]["navigation_constant"][
        "source"] == "paper_unspecified_engineering"
    assert snapshot["environment_config"]["missile"]["model"]["source"] == (
        "intentional_model_simplification")


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


def test_dynamic_nearest_unique_assignment_and_live_switch():
    controller = BluePolicyController(PAPER_BLUE_POLICY_PROFILE)
    controller.reset(["blue_0", "blue_1", "blue_2"],
                     ["red_0", "red_1", "red_2"], {}, {})
    own = {"blue_0": np.array([0.0, 0.0, 0.0]),
           "blue_1": np.array([0.0, 10.0, 0.0]),
           "blue_2": np.array([0.0, 20.0, 0.0])}
    enemies = {"red_0": np.array([100.0, 0.0, 0.0]),
               "red_1": np.array([100.0, 10.0, 0.0]),
               "red_2": np.array([100.0, 20.0, 0.0])}
    kwargs = dict(
        blue_obs={}, num_blue=3, num_red=3, engaged_targets=set(),
        own_positions=own, own_headings={key: 0.0 for key in own},
        current_step=0, own_alive={key: True for key in own},
        enemy_positions=enemies,
        enemy_alive={key: True for key in enemies})
    actions = controller.act(**kwargs)
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1", "blue_2": "red_2"}
    assert all(action[2] == pytest.approx((300.0 - 102.0) / 306.0 * 2 - 1)
               for action in actions.values())
    enemies["red_0"] = np.array([1000.0, 0.0, 0.0])
    enemies["red_1"] = np.array([10.0, 0.0, 0.0])
    controller.act(**kwargs)
    assert controller.current_targets["blue_0"] == "red_1"
    assert controller.target_switch_counts["blue_0"] == 1


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


def test_pid_setpoint_change_suppresses_one_derivative_frame():
    pid = PIDController(
        1 / 60, profile=PAPER_PID_PROFILE,
        throttle_base=0.8, config=PAPER_ENVIRONMENT_CONFIG.pid)
    pid.compute_control((0, 0, 0), 300, 0, 0, 300)
    pid.compute_control((0, 0, 0), 300, 0, 1, 300)
    assert pid._last_diagnostic["setpoint_changed_this_frame"]
    assert pid._last_diagnostic["roll_pid"]["d"] == 0.0
    pid.compute_control((0, 0, 0), 300, 0, 1, 300)
    assert not pid._last_diagnostic["setpoint_changed_this_frame"]


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
    (0.0, 10.0), (4.0, 1.0 + 22.0 / 15.0),
    (15.0, 1.0), (35.0, 0.0), (36.0, 0.0)])
def test_ta_reward_boundaries(angle, expected):
    assert ta_angle_advantage_fixed(angle) == pytest.approx(expected)


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


@pytest.mark.parametrize("aid, team, other", [
    ("red_0", "red", "blue"),
    ("blue_0", "blue", "red"),
])
def test_mws_generation_and_flip_statistics_are_team_scoped(aid, team, other):
    env = UavCombatEnv()
    try:
        env.reset(seed=3)
        incoming = SimpleNamespace(uid=f"{other}_missile_test")
        env._mws_evasion_target(aid, incoming, 0.0, 1.0)
        env._mws_evasion_target(aid, incoming, 0.0, -1.0)
        diag = env._mws_decision_diagnostics
        assert diag[f"{team}_warning_generations"] == 1
        assert diag[f"{team}_suppressed_direction_flip_attempts"] == 1
        assert diag[f"{team}_maximum_continuous_decisions"] == 2
        assert diag[f"{team}_target_heading_delta_max_deg"] == pytest.approx(60.0)
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
