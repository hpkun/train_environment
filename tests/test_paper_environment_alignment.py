import math
import ast
from pathlib import Path

import numpy as np

from my_uav_env import UavCombatEnv
from my_uav_env.alignment.los_geometry import compute_body_x_q_los
from my_uav_env.alignment.launch_quality import LAUNCH_QUALITY_FIELDS, make_launch_quality_record
from my_uav_env.alignment.reward_utils import (
    REWARD_VERSION,
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
from train_vanilla_mappo import Config, _compute_obs_dim, _flatten_obs


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
        assert env.obs_mode == "paper_strict"
    finally:
        env.close()


def test_paper_default_disables_legacy_post_hit_kill_gates():
    env = UavCombatEnv()
    try:
        assert env.enable_kill_cooldown_gate is False
        assert env.enable_single_kill_per_step_gate is False
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
