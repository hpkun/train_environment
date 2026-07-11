import math
import ast
from pathlib import Path

import numpy as np
import rule_based_agent as rule_agent

from my_uav_env import UavCombatEnv
from my_uav_env.alignment.los_geometry import compute_body_x_q_los
from my_uav_env.alignment.launch_quality import LAUNCH_QUALITY_FIELDS, make_launch_quality_record
from my_uav_env.pid_controller import PIDController
from my_uav_env.simulator import MissileSimulator
from my_uav_env.utils import get2d_heading_AO_TA_R
from my_uav_env.alignment.reward_utils import (
    REWARD_VERSION,
    pitch_penalty_current,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)
from my_uav_env.alignment.state_extractor import extract_relative_state
from rule_based_agent import (
    _blue_simple_pursuit_action_impl,
    _ego_roll_pitch_heading,
    _ego_speed_mps,
    _entity_range_m,
    _relative_bearing_rad,
    _simple_body_vector_to_world_bearing,
    reset_rule_memory,
)
from train_vanilla_mappo import (
    Config,
    CentralizedCritic,
    VanillaActor,
    _classify_death_reason,
    _compute_global_state_dim,
    _compute_obs_dim,
    _episode_outcome,
    _flatten_obs,
    _global_state_from_local_obs_flats,
)
import torch


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


def test_pitch_penalty_eq15_is_continuous_between_45_and_60_degrees():
    assert pitch_penalty_current(math.pi / 4.0) == 0.0
    assert math.isclose(pitch_penalty_current(math.radians(50.0)), -1.0 / 3.0)
    assert math.isclose(
        pitch_penalty_current(math.pi / 3.0 - 1e-9),
        -1.0,
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert pitch_penalty_current(math.pi / 3.0 + 1e-9) == -1.0


def test_reward_version_names_paper_eq20_default():
    assert "eq15" in REWARD_VERSION
    assert "eq20" in REWARD_VERSION
    assert "fixed_ta" not in REWARD_VERSION


def test_main_training_defaults_match_paper_scale():
    cfg = Config()
    assert cfg.num_red == 6
    assert cfg.num_blue == 6
    assert cfg.max_episode_length == 1400
    assert cfg.obs_mode == "paper_strict"


def test_blue_gcas_default_is_disabled():
    env = UavCombatEnv()
    try:
        assert env.enable_gcas_for_blue is False
        assert env.obs_mode == "paper_strict"
    finally:
        env.close()


def test_paper_default_disables_legacy_post_hit_kill_gates():
    env = UavCombatEnv()
    try:
        assert env.enable_kill_cooldown_gate is False
        assert env.enable_single_kill_per_step_gate is False
        assert math.isclose(env.missile_launch_ao_thresh, math.radians(45.0))
        assert env.missile_launch_min_range == 500.0
    finally:
        env.close()


def test_legacy_post_hit_kill_gates_are_explicitly_guarded():
    source = Path("my_uav_env/env.py").read_text(encoding="utf-8")
    assert "on_kill_cooldown = self.enable_kill_cooldown_gate" in source
    assert "if self.enable_kill_cooldown_gate and shooter_id in self._agents_deny_kill" in source
    assert "self.enable_single_kill_per_step_gate" in source


def test_launch_quality_keeps_raw_termination_reason_field():
    record = make_launch_quality_record(
        team="red",
        shooter_id="red_0",
        target_id="blue_0",
        current_step=1,
        physics_frame=2,
        range_m=1000.0,
        AO_rad=0.1,
        TA_rad=2.0,
        shooter_pos=np.array([0.0, 0.0, 6000.0]),
        shooter_vel=np.array([250.0, 0.0, 0.0]),
        target_pos=np.array([1000.0, 0.0, 6000.0]),
        target_vel=np.array([200.0, 0.0, 0.0]),
        target_alive_at_launch=True,
    )

    assert "raw_termination_reason" in LAUNCH_QUALITY_FIELDS
    assert "raw_termination_reason" in record
    assert "raw_termination_reason" in Path("train_vanilla_mappo.py").read_text(encoding="utf-8")


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


def test_rule_helpers_read_paper_strict_ego_and_entity_layout():
    obs = {
        "ego_state": np.array(
            [0.0, 0.0, 6000.0, 250.0, 0.1, 0.2, 1.3, 0.0, 0.0, -5.0],
            dtype=np.float32,
        ),
    }
    entity = np.array(
        [1000.0, 200.0, -50.0, 0.0, 0.0, 220.0, 0.0, 0.75, 0.4, 1234.0],
        dtype=np.float32,
    )

    assert _ego_speed_mps(obs) == 250.0
    assert np.allclose(_ego_roll_pitch_heading(obs), (0.1, 0.2, 1.3))
    assert np.allclose(_ego_roll_pitch_heading(obs, own_heading=2.0), (0.1, 0.2, 2.0))
    assert _entity_range_m(entity) == 1234.0
    assert _relative_bearing_rad(entity) == 0.75


def test_paper_strict_safe_pursuit_does_not_use_heading_as_speed():
    reset_rule_memory()
    obs = {
        "ego_state": np.array(
            [0.0, 0.0, 6000.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "enemy_states": np.array(
            [[2000.0, 0.0, 0.0, 0.0, 0.0, 230.0, 0.0, 0.0, 0.0, 2000.0]],
            dtype=np.float32,
        ),
        "ally_states": np.zeros((0, 10), dtype=np.float32),
        "death_mask": np.array([1, 1], dtype=np.float32),
        "missile_warning": np.array([0.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([250.0, 0.0, 0.0], dtype=np.float32),
    }

    action = _blue_simple_pursuit_action_impl(
        obs, num_blue=1, num_red=1, blue_id=0, forced_target_idx=None)

    low_speed_throttle = (90.0 + 100.0 * 1.0) / 306.0
    pursuit_throttle = (90.0 + 100.0 * 0.8) / 306.0
    assert math.isclose(float(action[2]), pursuit_throttle, rel_tol=1e-6)
    assert not math.isclose(float(action[2]), low_speed_throttle, rel_tol=1e-6)


def test_run_one_episode_does_not_reference_outer_args():
    tree = ast.parse(Path("evaluate_vanilla_mappo.py").read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_one_episode"
    )
    names = {node.id for node in ast.walk(func) if isinstance(node, ast.Name)}
    assert "args" not in names


def test_paper_strict_world_bearing_zero_roll_pitch_matches_yaw_plane():
    state = np.array(
        [1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(2) * 1000.0],
        dtype=np.float32,
    )
    ego = np.array([0.0, 0.0, 6000.0, 250.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                   dtype=np.float32)

    bearing = _simple_body_vector_to_world_bearing(state, ego, own_heading=float(ego[6]))

    assert math.isclose(bearing, 0.5 + math.pi / 4.0, rel_tol=1e-6, abs_tol=1e-6)


def test_paper_strict_world_bearing_front_target_cardinal_headings():
    state = np.array(
        [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0],
        dtype=np.float32,
    )
    ego_north = np.array([0.0, 0.0, 6000.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         dtype=np.float32)
    ego_east = ego_north.copy()
    ego_east[6] = math.pi / 2.0

    assert math.isclose(
        _simple_body_vector_to_world_bearing(state, ego_north, own_heading=float(ego_north[6])),
        0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        _simple_body_vector_to_world_bearing(state, ego_east, own_heading=float(ego_east[6])),
        math.pi / 2.0,
        abs_tol=1e-6,
    )


def test_paper_strict_world_bearing_uses_roll_pitch_for_3d_body_vector():
    state = np.array(
        [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0],
        dtype=np.float32,
    )
    ego = np.array(
        [0.0, 0.0, 6000.0, 250.0, math.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    bearing = _simple_body_vector_to_world_bearing(state, ego, own_heading=float(ego[6]))

    assert np.isfinite(bearing)
    assert math.isclose(bearing, -math.pi / 2.0, abs_tol=1e-6)


def test_flatten_obs_dim_matches_compute_obs_dim_for_paper_strict():
    obs = {
        "ego_state": np.zeros(10, dtype=np.float32),
        "ally_states": np.zeros((0, 10), dtype=np.float32),
        "enemy_states": np.zeros((1, 10), dtype=np.float32),
        "death_mask": np.ones(2, dtype=np.float32),
        "missile_warning": np.array([1.0], dtype=np.float32),
        "altitude": np.array([6100.0], dtype=np.float32),
        "velocity": np.array([200.0, 0.0, 0.0], dtype=np.float32),
    }

    flat = _flatten_obs(obs, obs_mode="paper_strict")
    assert flat.shape == (_compute_obs_dim(1, 1, is_red=True, obs_mode="paper_strict"),)


def test_flatten_obs_dim_matches_compute_obs_dim_for_engineering():
    obs = {
        "ego_state": np.zeros(11, dtype=np.float32),
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "enemy_states": np.zeros((1, 11), dtype=np.float32),
        "death_mask": np.ones(2, dtype=np.float32),
        "missile_warning": np.array([1.0], dtype=np.float32),
        "altitude": np.array([6100.0], dtype=np.float32),
        "velocity": np.array([200.0, 0.0, 0.0], dtype=np.float32),
    }

    flat = _flatten_obs(obs, obs_mode="engineering")
    assert flat.shape == (_compute_obs_dim(1, 1, is_red=True, obs_mode="engineering"),)


def test_missile_timing_scales_with_sim_freq():
    env = UavCombatEnv(sim_freq=40)
    try:
        assert env.missile_lock_delay_frames == 10
        assert env.missile_cooldown_frames == 20
    finally:
        env.close()


class _FakeSim:
    def __init__(self, pos, vel, rpy=(0.0, 0.0, 0.0), uid="fake", color="Red"):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.uid = uid
        self.color = color
        self.is_alive = True

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


def test_paper_relative_body_axes_use_z_down_and_positive_pitch_nose_up():
    level = _FakeSim([0.0, 0.0, 0.0], [200.0, 0.0, 0.0])
    above = _FakeSim([1000.0, 0.0, 100.0], [200.0, 0.0, 0.0])
    row_above = extract_relative_state(level, above)
    assert row_above[0] > 0.0
    assert row_above[2] < 0.0

    pitch = math.radians(30.0)
    pitched = _FakeSim([0.0, 0.0, 0.0], [200.0, 0.0, 0.0], rpy=(0.0, pitch, 0.0))
    on_nose = _FakeSim(
        [1000.0 * math.cos(pitch), 0.0, 1000.0 * math.sin(pitch)],
        [200.0, 0.0, 0.0],
    )
    row_on_nose = extract_relative_state(pitched, on_nose)
    assert row_on_nose[0] > 999.0
    assert abs(row_on_nose[2]) < 1e-6
    assert abs(row_on_nose[8]) < 1e-6


def test_body_x_q_los_geometry_cardinal_cases():
    pos = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])

    assert math.isclose(compute_body_x_q_los(pos, rpy, [1000.0, 0.0, 0.0]), 0.0)
    assert math.isclose(compute_body_x_q_los(pos, rpy, [-1000.0, 0.0, 0.0]), math.pi)
    assert math.isclose(
        compute_body_x_q_los(pos, rpy, [0.0, 1000.0, 0.0]),
        math.pi / 2.0,
    )


def test_radar_fov_uses_body_frame_with_roll_pitch_heading():
    env = UavCombatEnv()
    env.RADAR_K = 1e9
    try:
        rpy = np.array([math.radians(70.0), math.radians(20.0), math.radians(35.0)])
        ego = _FakeSim([0.0, 0.0, 0.0], [250.0, 0.0, 0.0], rpy=rpy)

        def target_at_body_angles(az_deg, el_deg):
            az = math.radians(az_deg)
            el = math.radians(el_deg)
            body_ned = np.array([
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                -math.sin(el),
            ]) * 1000.0
            inertial_ned = PIDController.body_to_ned_matrix(*rpy) @ body_ned
            pos_neu = np.array([inertial_ned[0], inertial_ned[1], -inertial_ned[2]])
            return _FakeSim(pos_neu, [200.0, 0.0, 0.0])

        assert env._is_detected_by_radar(ego, target_at_body_angles(59.9, 0.0))
        assert not env._is_detected_by_radar(ego, target_at_body_angles(60.1, 0.0))
        assert env._is_detected_by_radar(ego, target_at_body_angles(0.0, 31.9))
        assert not env._is_detected_by_radar(ego, target_at_body_angles(0.0, 32.1))
        assert env._is_detected_by_radar(ego, target_at_body_angles(0.0, -9.9))
        assert not env._is_detected_by_radar(ego, target_at_body_angles(0.0, -10.1))
    finally:
        env.close()


def test_launch_geometry_uses_shooter_and_target_headings_for_rear_hemisphere():
    target_pos = np.array([0.0, 0.0, 0.0])
    target_heading = 0.0

    ao, ta, distance = get2d_heading_AO_TA_R(
        np.array([-1000.0, 0.0, 0.0]), 0.0, target_pos, target_heading)
    assert math.isclose(ao, 0.0, abs_tol=1e-9)
    assert math.isclose(ta, math.pi, abs_tol=1e-9)
    assert math.isclose(distance, 1000.0, abs_tol=1e-9)

    _ao, ta_front, _distance = get2d_heading_AO_TA_R(
        np.array([1000.0, 0.0, 0.0]), math.pi, target_pos, target_heading)
    assert math.isclose(ta_front, 0.0, abs_tol=1e-9)


def test_missile_hit_probability_follows_velocity_to_target_direction(monkeypatch):
    missile = MissileSimulator(dt=1.0 / 60.0)
    missile._position[:] = np.array([0.0, 0.0, 0.0])
    missile.target_aircraft = _FakeSim([1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: 0.5)

    missile._velocity[:] = np.array([300.0, 0.0, 0.0])
    assert missile._roll_hit_probability()
    missile._velocity[:] = np.array([-300.0, 0.0, 0.0])
    assert not missile._roll_hit_probability()


def test_pid_channels_rudder_and_reset_match_paper_interface():
    pid = PIDController(dt=1.0 / 60.0)
    aileron, elevator, rudder, throttle = pid.compute_control(
        current_rpy=np.zeros(3),
        current_velocity=200.0,
        target_pitch=math.radians(10.0),
        target_heading=math.radians(30.0),
        target_velocity=250.0,
        ned_velocity=np.array([200.0, 0.0, 0.0]),
    )
    assert aileron > 0.0
    assert elevator < 0.0
    assert rudder == 0.0
    assert 0.0 <= throttle <= 1.0
    pid.compute_control(
        current_rpy=np.zeros(3),
        current_velocity=200.0,
        target_pitch=math.radians(10.0),
        target_heading=math.radians(30.0),
        target_velocity=250.0,
        ned_velocity=np.array([200.0, 0.0, 0.0]),
    )
    assert pid._roll_pid._integral != 0.0 or pid._pitch_pid._integral != 0.0

    pid.reset()
    assert pid._roll_pid._integral == 0.0
    assert pid._pitch_pid._integral == 0.0
    assert pid._velocity_pid._integral == 0.0
    assert pid._prev_target_heading is None


def test_critic_global_state_contains_only_red_native_ego_states():
    local_a = np.arange(132, dtype=np.float32)
    local_b = np.arange(132, dtype=np.float32) + 1000.0
    state = _global_state_from_local_obs_flats(
        [local_a, local_b], obs_mode="paper_strict")

    assert state.shape == (_compute_global_state_dim(2, "paper_strict"),)
    assert np.array_equal(state[:10], local_a[:10])
    assert np.array_equal(state[10:], local_b[:10])


def test_standard_mappo_actor_critic_shapes_for_paper_6v6():
    obs_dim = _compute_obs_dim(6, 6, is_red=True, obs_mode="paper_strict")
    global_dim = _compute_global_state_dim(6, "paper_strict")
    actor = VanillaActor(obs_dim=obs_dim, action_dim=3, hidden=16, rnn_hidden=8)
    critic = CentralizedCritic(global_obs_dim=global_dim, hidden=16)

    distribution, hidden = actor(
        torch.zeros((6, obs_dim), dtype=torch.float32),
        torch.zeros((6, 8), dtype=torch.float32),
    )
    values = critic(torch.zeros((2, global_dim), dtype=torch.float32))

    assert distribution.mean.shape == (6, 3)
    assert hidden.shape == (6, 8)
    assert values.shape == (2, 1)


def test_timeout_outcome_compares_survivor_counts():
    assert _episode_outcome(red_alive=4, blue_alive=3) == "red"
    assert _episode_outcome(red_alive=3, blue_alive=4) == "blue"
    assert _episode_outcome(red_alive=3, blue_alive=3) == "draw"


def test_reset_creates_head_on_approximately_10km_initial_geometry():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1, suppress_jsbsim_output=True)
    try:
        env.reset()
        red = env.red_planes["red_0"]
        blue = env.blue_planes["blue_0"]
        separation = np.linalg.norm(red.get_position() - blue.get_position())
        heading_delta = abs((red.get_rpy()[2] - blue.get_rpy()[2] + math.pi) % (2 * math.pi) - math.pi)
        assert math.isclose(separation, 10000.0, rel_tol=0.01)
        assert math.isclose(heading_delta, math.pi, abs_tol=math.radians(1.0))
    finally:
        env.close()


def test_dead_entities_are_zeroed_in_paper_strict_observation():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1, suppress_jsbsim_output=True)
    try:
        env.reset()
        env.blue_planes["blue_0"].shotdown()
        obs = env._get_obs()
        assert np.all(obs["red_0"]["enemy_states"][0] == 0.0)
        assert obs["red_0"]["death_mask"][0] == 0
        assert np.all(obs["blue_0"]["ego_state"] == 0.0)
    finally:
        env.close()


class _LaunchSim(_FakeSim):
    def __init__(self, pos, heading, uid, color):
        super().__init__(pos, [250.0, 0.0, 0.0], rpy=(0.0, 0.0, heading),
                         uid=uid, color=color)
        self.num_left_missiles = 1

    def close(self):
        pass


def test_launch_lock_resets_on_target_switch_and_loss():
    env = UavCombatEnv(max_num_red=1, max_num_blue=2)
    shooter = _LaunchSim([-1000.0, 0.0, 0.0], 0.0, "red_0", "Red")
    target_a = _LaunchSim([0.0, 0.0, 0.0], 0.0, "blue_0", "Blue")
    target_b = _LaunchSim([100.0, 0.0, 0.0], 0.0, "blue_1", "Blue")
    env.red_planes = {"red_0": shooter}
    env.blue_planes = {"blue_0": target_a, "blue_1": target_b}
    env.agent_ids = ["red_0"]
    env._missile_cooldown = {"red_0": 0}
    env._lock_timer = {"red_0": 0}
    env._lock_target = {"red_0": None}
    env.missile_lock_delay_frames = 999
    try:
        env._check_missile_launch()
        assert env._lock_target["red_0"] == "blue_0"
        assert env._lock_timer["red_0"] == 1

        target_a.is_alive = False
        env._check_missile_launch()
        assert env._lock_target["red_0"] == "blue_1"
        assert env._lock_timer["red_0"] == 1

        target_b.is_alive = False
        env._check_missile_launch()
        assert env._lock_target["red_0"] is None
        assert env._lock_timer["red_0"] == 0
    finally:
        env.close()


def test_launch_sensor_range_uses_3d_distance():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1)
    shooter = _LaunchSim([0.0, 0.0, 0.0], 0.0, "red_0", "Red")
    target = _LaunchSim([9000.0, 0.0, 6000.0], 0.0, "blue_0", "Blue")
    env.red_planes = {"red_0": shooter}
    env.blue_planes = {"blue_0": target}
    env.agent_ids = ["red_0"]
    env._missile_cooldown = {"red_0": 0}
    env._lock_timer = {"red_0": 0}
    env._lock_target = {"red_0": None}
    try:
        env._check_missile_launch()
        assert env._lock_target["red_0"] is None
        assert env._lock_timer["red_0"] == 0
    finally:
        env.close()


def test_engaged_target_is_released_when_missile_is_no_longer_alive():
    env = UavCombatEnv()
    missile = type("Missile", (), {"is_alive": True, "_target_id": "blue_0"})()
    env._missiles_in_flight = {"m0": missile}
    try:
        assert env.refresh_engaged_targets() == {"blue_0"}
        missile.is_alive = False
        assert env.refresh_engaged_targets() == set()
    finally:
        env.close()


def test_safe_pursuit_excludes_inflight_engaged_target_without_mutating_set(monkeypatch):
    forced_targets = []

    def fake_action(_obs, _num_blue, _num_red, _blue_id,
                    forced_target_idx=None, **_kwargs):
        forced_targets.append(forced_target_idx)
        return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(rule_agent, "_blue_simple_pursuit_action_impl", fake_action)
    obs = {
        "blue_0": {
            "enemy_states": np.array([
                [1000.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 1000.0],
                [2000.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 2000.0],
            ], dtype=np.float32),
            "death_mask": np.ones(3, dtype=np.float32),
        },
    }
    engaged = {"red_0"}

    rule_agent.blue_coordinated_actions(
        obs, num_blue=1, num_red=2, engaged_targets=engaged)

    assert forced_targets == [1]
    assert engaged == {"red_0"}


def test_zero_missile_shooter_cannot_launch():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1)
    shooter = _LaunchSim([-1000.0, 0.0, 0.0], 0.0, "red_0", "Red")
    target = _LaunchSim([0.0, 0.0, 0.0], 0.0, "blue_0", "Blue")
    shooter.num_left_missiles = 0
    env.red_planes = {"red_0": shooter}
    env.blue_planes = {"blue_0": target}
    env.agent_ids = ["red_0"]
    env._missile_cooldown = {"red_0": 0}
    env._lock_timer = {"red_0": env.missile_lock_delay_frames}
    env._lock_target = {"red_0": "blue_0"}
    try:
        env._check_missile_launch()
        assert env._missiles_in_flight == {}
    finally:
        env.close()


def test_situation_reward_sums_every_alive_enemy():
    env = UavCombatEnv(max_num_red=1, max_num_blue=2)
    ego = _LaunchSim([0.0, 0.0, 0.0], 0.0, "red_0", "Red")
    enemy_a = _LaunchSim([1000.0, 0.0, 0.0], math.pi, "blue_0", "Blue")
    enemy_b = _LaunchSim([2000.0, 500.0, 0.0], math.pi, "blue_1", "Blue")
    env.red_planes = {"red_0": ego}
    env.blue_planes = {"blue_0": enemy_a, "blue_1": enemy_b}
    try:
        both = env._situation_reward(ego)
        enemy_b.is_alive = False
        only_a = env._situation_reward(ego)
        enemy_b.is_alive = True
        enemy_a.is_alive = False
        only_b = env._situation_reward(ego)
        assert math.isclose(both, only_a + only_b, rel_tol=1e-9, abs_tol=1e-9)
    finally:
        env.close()


def test_terminal_reward_sums_to_paper_team_scale():
    env = UavCombatEnv(max_num_red=2, max_num_blue=2, suppress_jsbsim_output=True)
    try:
        env.reset()
        env.blue_planes["blue_0"].shotdown()
        env.current_step = env.max_steps
        _rewards, components = env._compute_rewards()
        red_total = sum(components[aid].get("r_end", 0.0) for aid in env.red_ids)
        blue_total = sum(components[aid].get("r_end", 0.0) for aid in env.blue_ids)
        assert math.isclose(red_total, 30.0)
        assert math.isclose(blue_total, -30.0)
    finally:
        env.close()


def test_death_reason_categories_are_mutually_classified():
    assert _classify_death_reason("Missile_Kill") == "missile"
    assert _classify_death_reason("Crash_LowAlt") == "crash"
    assert _classify_death_reason("Crash_OverG") == "crash"


def test_training_evaluation_and_acmi_defaults_use_paper_strict_6v6():
    evaluate_source = Path("evaluate_vanilla_mappo.py").read_text(encoding="utf-8")
    acmi_source = Path("eval_acmi.py").read_text(encoding="utf-8")
    preset_source = Path("configs/experiment_presets.py").read_text(encoding="utf-8")
    assert 'default=6' in evaluate_source
    assert 'default=1400' in evaluate_source
    assert 'default="paper_strict"' in evaluate_source
    assert 'num_red: int = 6, num_blue: int = 6, max_steps: int = 1400' in acmi_source
    assert 'obs_mode: str = "paper_strict"' in acmi_source
    assert '"vanilla_6v6_paper_main"' in preset_source
