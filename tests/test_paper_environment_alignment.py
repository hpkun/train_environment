import math
import ast
import csv
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import rule_based_agent as rule_agent

from my_uav_env import UavCombatEnv
from my_uav_env.alignment.los_geometry import compute_body_x_q_los
from my_uav_env.alignment.launch_quality import LAUNCH_QUALITY_FIELDS, make_launch_quality_record
from my_uav_env.pid_controller import PIDController
from my_uav_env.simulator import (
    MissileSimulator,
    compute_los_angles_and_rates,
    compute_paper_eq9_overloads,
)
from my_uav_env.utils import get2d_heading_AO_TA_R
from my_uav_env.alignment.reward_utils import (
    AltitudeRewardConfig,
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
    altitude_reward_paper_eq17,
    pitch_penalty_current,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)
from my_uav_env.alignment.state_extractor import (
    build_strict_paper_entity_observation,
    extract_relative_state,
    ordered_entity_slots,
)
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
    ACTION_DISTRIBUTION_VERSION,
    ACTION_LOG_STD_INIT,
    CHECKPOINT_SCHEMA_VERSION,
    Config,
    CentralizedCritic,
    REWARD_COMPONENT_LOG_FIELDS,
    RolloutBuffer,
    VanillaActor,
    _build_actor_segments,
    _accumulate_action_bound_totals,
    _action_bound_metrics,
    _checkpoint_metadata,
    _classify_death_reason,
    _compute_global_state_dim,
    _compute_joint_gae_by_env,
    _compute_obs_dim,
    _episode_outcome,
    _empty_action_bound_totals,
    _flatten_obs,
    _global_state_from_local_obs_flats,
    _joint_team_reward_once,
    _normalize_paper_fixed_v1,
    _reward_component_log_metrics,
    _recompute_segment_log_probs,
    _ratio_with_denominator_zero,
    _rollout_layout,
    _unpack_and_validate_checkpoint,
    ppo_update,
)
import train_vanilla_mappo as tvm
import torch


def test_evaluation_csv_header_covers_joint_reward_and_environment_semantics():
    from evaluate_vanilla_mappo import (
        EVALUATION_FIELDNAMES,
        EVALUATION_SUMMARY_FIELDNAMES,
    )

    required = {
        "EpisodeRewardRed", "RewardVersion", "RewardMode",
        "ObsNormalization", "PIDProfile", "PIDThrottleBase",
        "MissileGuidanceMode", "CheckpointSchema", "ActionDistribution",
        "AltitudeRewardConfigVersion", "AltitudeRewardConfig",
    }
    assert required.issubset(EVALUATION_FIELDNAMES)
    assert len(EVALUATION_FIELDNAMES) == len(set(EVALUATION_FIELDNAMES))
    assert {
        "RWR", "RWRDenominatorZero", "KD_Red_AllDeaths",
        "KD_Red_MissileOnly",
    }.issubset(EVALUATION_SUMMARY_FIELDNAMES)


def test_ta_uses_paper_eq20_scale():
    samples = {
        0.0: 10.0,
        4.0: 1.0 + 2.0 * (15.0 - 4.0) / 15.0,
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


def test_pitch_penalty_uses_literal_discontinuous_eq15():
    assert pitch_penalty_current(math.pi / 4.0) == 0.0
    assert math.isclose(
        pitch_penalty_current(math.radians(50.0)),
        -(50.0 / 180.0 - 0.25) / 12.0,
    )
    assert math.isclose(
        pitch_penalty_current(math.pi / 3.0 - 1e-9),
        -1.0 / 144.0,
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert pitch_penalty_current(math.pi / 3.0 + 1e-9) == -1.0


def test_ta_preserves_literal_jump_at_four_degrees():
    assert ta_angle_advantage_fixed(4.0 - 1e-9) == 10.0
    assert math.isclose(
        ta_angle_advantage_fixed(4.0),
        1.0 + 2.0 * 11.0 / 15.0,
    )


def test_reward_version_names_paper_eq20_default():
    assert "eq15" in REWARD_VERSION
    assert "eq20" in REWARD_VERSION
    assert "fixed_ta" not in REWARD_VERSION


def test_main_training_defaults_match_main_3v3_scale():
    cfg = Config()
    assert cfg.num_red == 3
    assert cfg.num_blue == 3
    assert cfg.max_episode_length == 1400
    assert cfg.obs_mode == "paper_strict"
    assert cfg.obs_normalization == "paper_fixed_v1"
    assert cfg.pid_profile == "paper"
    assert cfg.reward_mode == "paper_joint"
    assert cfg.missile_guidance_mode == "paper_eq9"


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


def test_training_rollout_main_has_no_post_sample_clamp():
    tree = ast.parse(Path("train_vanilla_mappo.py").read_text(encoding="utf-8"))
    main_func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    clamp_calls = [
        node for node in ast.walk(main_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clamp"
    ]
    assert not clamp_calls


def test_evaluation_and_acmi_use_shared_joint_reward_reader():
    for path, function_name in (
        ("evaluate_vanilla_mappo.py", "run_one_episode"),
        ("eval_acmi.py", "run_acmi"),
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        func = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        called_names = {
            node.func.id for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_joint_team_reward_once" in called_names


def test_evaluation_executes_clipped_normal_mean(monkeypatch):
    import evaluate_vanilla_mappo as evaluation

    def observation():
        return {
            "ego_state": np.array(
                [0, 0, 6000, 250, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "ally_states": np.zeros((0, 10), dtype=np.float32),
            "enemy_states": np.zeros((1, 10), dtype=np.float32),
            "alive_mask": np.ones(2, dtype=np.int64),
            "death_mask": np.ones(2, dtype=np.int64),
        }

    class MeanOnlyDistribution:
        @property
        def mean(self):
            return torch.tensor([[1.25, -1.5, 0.75]], dtype=torch.float32)

    class FakeActor:
        def __call__(self, obs, hidden):
            return MeanOnlyDistribution(), hidden

    class FakeEnv:
        last_red_action = None

        def __init__(self, **_kwargs):
            self.agent_ids = ["blue_0", "red_0"]

        def reset(self):
            obs = {"blue_0": observation(), "red_0": observation()}
            return obs, {}

        def refresh_engaged_targets(self):
            return set()

        def get_blue_own_kinematics(self):
            return {}

        def step(self, actions):
            FakeEnv.last_red_action = np.asarray(actions["red_0"])
            obs = {"blue_0": observation(), "red_0": observation()}
            rewards = {"blue_0": 0.0, "red_0": 0.0}
            terminated = {"blue_0": True, "red_0": True}
            truncated = {"blue_0": False, "red_0": False}
            info = {
                "blue_0": {"alive": True}, "red_0": {"alive": True},
            }
            return obs, rewards, terminated, truncated, info

        def close(self):
            pass

    monkeypatch.setattr(evaluation, "UavCombatEnv", FakeEnv)
    monkeypatch.setattr(
        evaluation, "blue_coordinated_actions",
        lambda *_args, **_kwargs: {"blue_0": np.zeros(3, dtype=np.float32)})

    evaluation.run_one_episode(
        actor=FakeActor(), rnn_hidden_size=4, num_red=1, num_blue=1,
        max_steps=1, device=torch.device("cpu"), episode_idx=1,
        enable_blue_gcas=False, obs_mode="paper_strict")

    assert np.allclose(FakeEnv.last_red_action, [1.0, -1.0, 0.75])


def test_formal_evaluation_defaults_to_100_episodes(monkeypatch):
    import evaluate_vanilla_mappo as evaluation

    monkeypatch.setattr("sys.argv", ["evaluate_vanilla_mappo.py"])
    assert evaluation.parse_args().episodes == 100


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
    assert flat.shape == (20,)
    assert not np.any(flat == 1.0)


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
    # qLOS follows velocity rather than body-x; a level velocity vector sees
    # this pitched-up LOS at 30 degrees.
    assert math.isclose(row_on_nose[8], pitch, rel_tol=0.0, abs_tol=1e-6)


def test_paper_q_los_uses_velocity_direction_not_body_axis():
    observer = _FakeSim(
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
        rpy=(0.0, 0.0, math.pi / 2.0),
    )
    target = _FakeSim([1000.0, 0.0, 0.0], [100.0, 0.0, 0.0])
    row = extract_relative_state(observer, target)
    assert math.isclose(row[8], 0.0, abs_tol=1e-6)


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


def test_missile_hit_probability_zero_los_and_zero_speed_are_distinct(monkeypatch):
    missile = MissileSimulator(dt=1.0 / 60.0)
    missile._position[:] = np.array([0.0, 0.0, 0.0])
    missile._velocity[:] = np.array([0.0, 0.0, 0.0])

    missile.target_aircraft = _FakeSim(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: 0.99)
    assert missile._roll_hit_probability()  # geometric contact -> p=1

    missile.target_aircraft = _FakeSim(
        [1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: 0.10)
    assert not missile._roll_hit_probability()  # zero speed -> base p=0.05


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


def test_paper_pid_eq13_direction_errors_use_x_denominator():
    zero = np.zeros(3)
    roll_error, pitch_error, *_ = PIDController.paper_direction_errors(
        zero, 0.0, 0.0)
    assert math.isclose(roll_error, 0.0, abs_tol=1e-9)
    assert math.isclose(pitch_error, 0.0, abs_tol=1e-9)

    roll_error, pitch_error, *_ = PIDController.paper_direction_errors(
        zero, 0.0, math.radians(30.0))
    assert math.isclose(roll_error, math.radians(30.0), abs_tol=1e-9)
    assert math.isclose(pitch_error, 0.0, abs_tol=1e-9)

    roll_error, pitch_error, *_ = PIDController.paper_direction_errors(
        zero, 0.0, math.radians(-30.0))
    assert math.isclose(roll_error, math.radians(-30.0), abs_tol=1e-9)

    roll_error, pitch_error, *_ = PIDController.paper_direction_errors(
        zero, math.radians(10.0), 0.0)
    assert math.isclose(roll_error, 0.0, abs_tol=1e-9)
    assert math.isclose(pitch_error, math.radians(10.0), abs_tol=1e-9)


def test_paper_pid_rotation_round_trip_and_current_nose_zero_error():
    rpy = np.array([0.4, -0.3, 1.2])
    r_ib = PIDController.body_to_ned_matrix(*rpy)
    r_bi = PIDController.ned_to_body_matrix(*rpy)
    assert np.allclose(r_bi @ r_ib, np.eye(3), atol=1e-12)

    roll_error, pitch_error, *_ = PIDController.paper_direction_errors(
        rpy, target_pitch=float(rpy[1]), target_heading=float(rpy[2]))
    assert math.isclose(roll_error, 0.0, abs_tol=1e-9)
    assert math.isclose(pitch_error, 0.0, abs_tol=1e-9)


def test_pid_profiles_separate_paper_from_engineering_protections():
    paper = PIDController(1.0 / 60.0, profile="paper")
    engineering = PIDController(1.0 / 60.0, profile="engineering_safe")
    args = dict(
        current_rpy=np.array([0.0, math.radians(86.0), 0.0]),
        current_velocity=200.0,
        target_pitch=0.0,
        target_heading=0.0,
        target_velocity=250.0,
        ned_velocity=np.array([20.0, 0.0, -199.0]),
    )
    paper_control = paper.compute_control(**args)
    engineering_control = engineering.compute_control(**args)
    assert np.isfinite(paper_control).all()
    assert not np.allclose(paper_control, np.zeros(4))
    assert np.allclose(engineering_control, np.zeros(4))
    assert paper._prev_target_heading is None
    assert paper_control[2] == 0.0


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
    assert isinstance(distribution, torch.distributions.Normal)
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
        assert obs["red_0"]["death_mask"].tolist() == [1, 0]
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
        obs, num_blue=1, num_red=2, engaged_targets=engaged,
        pursuit_mode="safe_pursuit")

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
        rewards, components = env._compute_rewards()
        summary = env._reward_summary_step
        assert math.isclose(summary["red_team_terminal_reward"], 30.0)
        assert math.isclose(summary["blue_team_terminal_reward"], -30.0)
        assert len({rewards[aid] for aid in env.red_ids}) == 1
        assert len({rewards[aid] for aid in env.blue_ids}) == 1
        assert math.isclose(
            rewards["red_0"],
            summary["red_local_reward_sum"] + summary["red_team_terminal_reward"])
        assert all(components[aid]["r_death"] == 0.0 for aid in env.agent_ids)
    finally:
        env.close()


def test_paper_joint_crash_has_no_extra_death_penalty():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1, suppress_jsbsim_output=True)
    try:
        env.reset()
        env.red_planes["red_0"].shotdown()
        env._crashed_this_step.add("red_0")
        rewards, components = env._compute_rewards()
        assert rewards["red_0"] == -30.0
        assert components["red_0"]["r_death"] == 0.0
        assert env._reward_summary_step["red_team_terminal_reward"] == -30.0
    finally:
        env.close()


def test_death_reason_categories_are_mutually_classified():
    assert _classify_death_reason("Missile_Kill") == "missile"
    assert _classify_death_reason("Crash_LowAlt") == "crash"
    assert _classify_death_reason("Crash_OverG") == "crash"


def test_training_evaluation_and_acmi_defaults_use_paper_strict_3v3():
    evaluate_source = Path("evaluate_vanilla_mappo.py").read_text(encoding="utf-8")
    acmi_source = Path("eval_acmi.py").read_text(encoding="utf-8")
    preset_source = Path("configs/experiment_presets.py").read_text(encoding="utf-8")
    assert 'default=3' in evaluate_source
    assert 'default=1400' in evaluate_source
    assert 'default="paper_strict"' in evaluate_source
    assert 'num_red: int = 3, num_blue: int = 3, max_steps: int = 1400' in acmi_source
    assert 'obs_mode: str = "paper_strict"' in acmi_source
    assert '"main_3v3_mappo_mlp"' in preset_source


def test_table2_angles_use_up_positive_right_positive_convention():
    observer = _FakeSim([0.0, 0.0, 0.0], [200.0, 0.0, 0.0])
    above_right = _FakeSim([1000.0, 200.0, 300.0], [200.0, 20.0, 30.0])
    below_left = _FakeSim([1000.0, -200.0, -300.0], [200.0, -20.0, -30.0])

    upper = extract_relative_state(observer, above_right)
    lower = extract_relative_state(observer, below_left)
    assert upper[2] < 0.0 and upper[6] > 0.0 and upper[7] > 0.0
    assert upper[3] > 0.0 and upper[4] > 0.0
    assert lower[2] > 0.0 and lower[6] < 0.0 and lower[7] < 0.0
    assert lower[3] < 0.0 and lower[4] < 0.0
    assert 0.0 <= upper[8] <= math.pi
    assert 0.0 <= lower[8] <= math.pi


def test_alive_mask_alias_and_entity_mask_have_explicit_opposite_semantics():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1, suppress_jsbsim_output=True)
    try:
        env.reset()
        env.blue_planes["blue_0"].shotdown()
        obs = env._get_obs()["red_0"]
        _entities, entity_mask, _meta = build_strict_paper_entity_observation(
            env, "red_0")
        assert np.array_equal(obs["alive_mask"], obs["death_mask"])
        assert obs["alive_mask"].tolist() == [1, 0]
        assert np.array_equal(entity_mask, 1 - obs["alive_mask"])
        assert np.all(obs["enemy_states"][0] == 0.0)
    finally:
        env.close()


def test_3v2_masks_match_each_agents_ego_ally_enemy_slots_in_both_obs_modes():
    for obs_mode in ("paper_strict", "engineering"):
        env = UavCombatEnv(
            max_num_blue=3, max_num_red=2, obs_mode=obs_mode,
            suppress_jsbsim_output=True)
        try:
            env.reset()
            env.blue_planes["blue_1"].shotdown()
            env.red_planes["red_0"].shotdown()
            observations = env._get_obs()

            for aid in ("blue_0", "blue_1", "blue_2", "red_0", "red_1"):
                ego, allies, enemies = ordered_entity_slots(env, aid)
                slots = ego + allies + enemies
                expected_ids = [slot_id for slot_id, _sim in slots]
                if aid.startswith("blue"):
                    expected_layout = [aid] + [
                        bid for bid in env.blue_ids if bid != aid
                    ] + env.red_ids
                else:
                    expected_layout = [aid] + [
                        rid for rid in env.red_ids if rid != aid
                    ] + env.blue_ids
                assert expected_ids == expected_layout

                ego_alive = bool(slots[0][1].is_alive)
                expected_alive = np.asarray([
                    int(ego_alive and sim.is_alive) for _slot_id, sim in slots
                ], dtype=np.int64)
                obs = observations[aid]
                assert np.array_equal(obs["alive_mask"], expected_alive)
                assert np.array_equal(obs["death_mask"], expected_alive)

                _entities, entity_mask, _meta = \
                    build_strict_paper_entity_observation(env, aid)
                assert np.array_equal(entity_mask, 1 - obs["alive_mask"])

                rows = [obs["ego_state"], *obs["ally_states"], *obs["enemy_states"]]
                for slot_idx, valid in enumerate(expected_alive):
                    if not valid:
                        assert np.allclose(rows[slot_idx], 0.0)
                if aid.startswith("blue"):
                    candidates = rule_agent._simple_valid_targets(
                        obs, num_blue=3, num_red=2)
                    expected_targets = [] if not ego_alive else [1]
                    assert [candidate[1] for candidate in candidates] == expected_targets
        finally:
            env.close()


def test_paper_fixed_v1_normalization_uses_declared_physical_scales():
    obs = {
        "ego_state": np.array([
            40000.0, -40000.0, 10000.0, 600.0,
            math.pi, -math.pi, math.pi, 0.5 * math.pi, -0.5 * math.pi, -600.0,
        ], dtype=np.float32),
        "ally_states": np.array([[
            40000.0, -40000.0, 10000.0, math.pi, -math.pi,
            600.0, math.pi, -math.pi, math.pi, 100000.0,
        ]], dtype=np.float32),
        "enemy_states": np.zeros((0, 10), dtype=np.float32),
    }
    ego, allies, enemies = _normalize_paper_fixed_v1(obs)
    assert np.allclose(ego, [1, -1, 1, 1, 1, -1, 1, 0.5, -0.5, -1])
    assert np.allclose(allies[0], [1, -1, 1, 1, -1, 1, 1, -1, 1, 1])
    assert enemies.shape == (0, 10)


def test_joint_reward_is_read_once_and_rejects_agent_disagreement():
    rewards = {"red_0": 12.5, "red_1": 12.5}
    assert _joint_team_reward_once(rewards, ["red_0", "red_1"]) == 12.5
    with np.testing.assert_raises(ValueError):
        _joint_team_reward_once(
            {"red_0": 12.5, "red_1": 13.0}, ["red_0", "red_1"])


def test_reward_component_logs_distinguish_team_sum_and_per_agent_mean():
    metrics = _reward_component_log_metrics(
        {"r_pitch": -6.0, "r_adv": 12.0, "r_end": 30.0}, team_size=6)
    assert metrics["r_pitch_team_sum"] == -6.0
    assert metrics["r_pitch_per_agent_mean"] == -1.0
    assert metrics["r_adv_team_sum"] == 12.0
    assert metrics["r_adv_per_agent_mean"] == 2.0
    assert metrics["r_end_team"] == 30.0
    assert set(metrics) == set(REWARD_COMPONENT_LOG_FIELDS)


def test_eq17_all_boundaries_finite_and_unbounded_tail_modes():
    finite = AltitudeRewardConfig(
        version="eq17_engineering_thresholds_finite_tail_test",
        h_min_m=0.0, h_att_m=2.0, h_adv_m=5.0, h_max_m=10.0,
        d_att_max_m=12.0, high_altitude_tail=0.1)
    eps = 1e-4
    assert altitude_reward_paper_eq17(0.0, finite) == 0.0
    assert altitude_reward_paper_eq17(0.0 + eps, finite) > 0.0
    assert altitude_reward_paper_eq17(2.0 - eps, finite) <= 1.0
    assert altitude_reward_paper_eq17(2.0, finite) == 1.0
    assert altitude_reward_paper_eq17(5.0, finite) == 1.0
    assert altitude_reward_paper_eq17(5.0 + eps, finite) < 1.0
    assert math.isclose(altitude_reward_paper_eq17(10.0, finite), 0.1)
    assert altitude_reward_paper_eq17(10.0 + eps, finite) == 0.1
    assert altitude_reward_paper_eq17(12.0 - eps, finite) == 0.1
    assert altitude_reward_paper_eq17(12.0, finite) == 0.0

    assert DEFAULT_ALTITUDE_REWARD_CONFIG.d_att_max_m is None
    assert "unbounded_tail" in DEFAULT_ALTITUDE_REWARD_CONFIG.version
    assert altitude_reward_paper_eq17(
        1e9, DEFAULT_ALTITUDE_REWARD_CONFIG) == \
        DEFAULT_ALTITUDE_REWARD_CONFIG.high_altitude_tail


def test_eq17_rejects_invalid_thresholds_and_tail():
    with np.testing.assert_raises_regex(ValueError, "h_min_m"):
        AltitudeRewardConfig(h_min_m=2.0, h_att_m=1.0)
    with np.testing.assert_raises_regex(ValueError, "d_att_max_m"):
        AltitudeRewardConfig(d_att_max_m=9000.0)
    with np.testing.assert_raises_regex(ValueError, "high_altitude_tail"):
        AltitudeRewardConfig(high_altitude_tail=1.1)
    with np.testing.assert_raises_regex(ValueError, "unbounded"):
        AltitudeRewardConfig(version="ambiguous", d_att_max_m=None)


def test_pid_throttle_base_is_explicit_correction_offset():
    common = dict(
        current_rpy=np.zeros(3), current_velocity=250.0,
        target_pitch=0.0, target_heading=0.0, target_velocity=250.0,
        ned_velocity=np.array([250.0, 0.0, 0.0]))
    zero_base = PIDController(1.0 / 60.0, throttle_base=0.0)
    nonzero_base = PIDController(1.0 / 60.0, throttle_base=0.3)
    assert zero_base.compute_control(**common)[3] == 0.0
    assert math.isclose(nonzero_base.compute_control(**common)[3], 0.3)
    faster = dict(common, target_velocity=1000.0)
    slower = dict(common, target_velocity=0.0)
    assert nonzero_base.compute_control(**faster)[3] == 1.0
    nonzero_base.reset()
    assert nonzero_base.compute_control(**slower)[3] == 0.0


def test_joint_gae_is_one_trajectory_per_env_and_terminal_bootstrap_is_zeroed():
    buffer = RolloutBuffer(
        num_steps=3, num_envs=1, num_red=2, action_dim=3, rnn_hidden_size=4)
    buffer.joint_rewards[:, 0] = [1.0, 2.0, 3.0]
    buffer.team_values[:, 0] = [0.0, 0.0, 0.0]
    buffer.episode_dones[:, 0] = [0.0, 0.0, 1.0]
    buffer.bootstrap_value[0] = 0.0

    advantages, returns = _compute_joint_gae_by_env(
        buffer, gamma=1.0, lam=1.0, device=torch.device("cpu"))
    assert len(advantages) == 1 and len(returns) == 1
    assert torch.allclose(advantages[0], torch.tensor([6.0, 5.0, 3.0]))
    assert torch.allclose(returns[0], torch.tensor([6.0, 5.0, 3.0]))


def test_vanilla_actor_matches_paper_mlp_gru_normal_layout():
    actor = VanillaActor(obs_dim=60)
    distribution, hidden = actor(torch.zeros(2, 60), torch.zeros(2, 128))

    assert actor.fc_in.in_features == 60 and actor.fc_in.out_features == 128
    assert actor.fc_hidden.in_features == 128 and actor.fc_hidden.out_features == 128
    assert actor.rnn.input_size == 128 and actor.rnn.hidden_size == 128
    assert actor.action_head.in_features == 128 and actor.action_head.out_features == 3
    assert isinstance(distribution, torch.distributions.Normal)
    assert distribution.mean.shape == (2, 3) and hidden.shape == (2, 128)
    assert torch.allclose(distribution.scale, torch.full((2, 3), 0.3))
    assert torch.count_nonzero(actor.action_log_std_head.weight) == 0
    assert torch.all(distribution.scale >= 0.05)
    assert torch.all(distribution.scale <= 0.6)


def test_raw_action_is_buffered_while_environment_action_is_clipped():
    raw = np.array([[1.4, -1.2, 0.25]], dtype=np.float32)
    env_action = np.clip(raw, -1.0, 1.0)
    buffer = RolloutBuffer(1, 1, 1, 3, 4)
    buffer.store_step(0, 0, 0, np.zeros(2, dtype=np.float32), raw[0], 0.0,
                      True, np.zeros(4, dtype=np.float32), True)
    totals = _empty_action_bound_totals()
    _accumulate_action_bound_totals(totals, raw, env_action)
    metrics = _action_bound_metrics(totals)

    assert np.array_equal(buffer.actions[0, 0, 0], raw[0])
    assert np.array_equal(env_action[0], [1.0, -1.0, 0.25])
    assert math.isclose(metrics["RawActionOutOfBoundsFrac"], 2.0 / 3.0)
    assert math.isclose(metrics["EnvActionNearBoundFrac"], 2.0 / 3.0)


def _fill_actor_step(buffer, actor, step, obs, hidden, sequence_start):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    hidden_t = torch.as_tensor(hidden, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        distribution, next_hidden = actor(obs_t, hidden_t)
        action = distribution.mean
        log_prob = distribution.log_prob(action).sum(dim=-1)
    buffer.store_step(
        step, 0, 0, obs, action[0].numpy(), float(log_prob.item()), True,
        actor_rnn_state_before=hidden,
        sequence_start=sequence_start)
    return next_hidden[0].numpy()


def test_recurrent_segments_split_two_episodes_and_recompute_old_log_probs():
    torch.manual_seed(5)
    actor = VanillaActor(obs_dim=4, action_dim=3, hidden=8, rnn_hidden=6)
    buffer = RolloutBuffer(6, 1, 1, 3, 6)
    hidden = np.zeros(6, dtype=np.float32)
    for step in range(6):
        if step == 3:
            hidden = np.zeros(6, dtype=np.float32)
        obs = np.full(4, 0.1 * (step + 1), dtype=np.float32)
        hidden = _fill_actor_step(
            buffer, actor, step, obs, hidden,
            sequence_start=step in (0, 3))
        buffer.global_states[step][0] = np.zeros(10, dtype=np.float32)
    buffer.episode_dones[:, 0] = [0, 0, 1, 0, 0, 1]
    advantages = [torch.arange(6, dtype=torch.float32)]

    segments = _build_actor_segments(buffer, advantages)

    assert [segment["steps"].tolist() for segment in segments] == [
        [0, 1, 2], [3, 4, 5]]
    assert np.allclose(segments[1]["initial_hidden"], 0.0)
    for segment in segments:
        recomputed, _entropy = _recompute_segment_log_probs(
            actor, segment, torch.device("cpu"))
        assert torch.allclose(
            recomputed,
            torch.as_tensor(segment["old_log_probs"]),
            atol=1e-6, rtol=1e-6)


def test_recurrent_segments_do_not_cross_individual_death_gap():
    actor = VanillaActor(obs_dim=3, action_dim=3, hidden=8, rnn_hidden=5)
    buffer = RolloutBuffer(5, 1, 1, 3, 5)
    hidden = np.zeros(5, dtype=np.float32)
    for step in (0, 1, 3, 4):
        if step == 3:
            hidden = np.zeros(5, dtype=np.float32)
        hidden = _fill_actor_step(
            buffer, actor, step,
            np.full(3, step + 1, dtype=np.float32), hidden,
            sequence_start=step in (0, 3))
    buffer.alive[2, 0, 0] = False
    buffer.episode_dones[-1, 0] = 1.0

    segments = _build_actor_segments(
        buffer, [torch.ones(5, dtype=torch.float32)])

    assert [segment["steps"].tolist() for segment in segments] == [[0, 1], [3, 4]]
    assert np.allclose(segments[1]["initial_hidden"], 0.0)


def test_missile_eq10_eq11_and_eq9_match_direct_geometry():
    beta, epsilon, beta_dot, epsilon_dot = compute_los_angles_and_rates(
        [1000.0, 0.0, 0.0], [0.0, 100.0, 0.0])
    assert beta == 0.0 and epsilon == 0.0
    assert math.isclose(beta_dot, 0.1)
    assert epsilon_dot == 0.0

    n_mc, n_mh = compute_paper_eq9_overloads(
        [1000.0, 0.0, 0.0], [0.0, 100.0, 0.0], [300.0, 0.0, 0.0],
        navigation_constant=3.0, gravity=9.81)
    assert math.isclose(n_mc, 3.0 * 300.0 / 9.81 * 0.1)
    assert n_mh == 0.0


def test_missile_eq9_vertical_and_general_3d_match_literal_formula():
    for relative_position, relative_velocity, missile_velocity in (
        ([1000.0, 0.0, 100.0], [0.0, 0.0, 10.0], [300.0, 0.0, 0.0]),
        ([1000.0, 400.0, 200.0], [-20.0, 80.0, 30.0], [280.0, 40.0, 25.0]),
    ):
        beta, epsilon, beta_dot, epsilon_dot = compute_los_angles_and_rates(
            relative_position, relative_velocity)
        speed = float(np.linalg.norm(missile_velocity))
        gamma = math.asin(missile_velocity[2] / speed)
        expected_n_mc = (
            3.0 * speed * math.cos(gamma) / 9.81
            * (beta_dot + math.tan(epsilon)
               * math.tan(epsilon + beta) * epsilon_dot)
        )
        expected_n_mh = (
            speed / 9.81 * 3.0 / math.cos(epsilon + beta) * epsilon_dot
        )
        actual_n_mc, actual_n_mh = compute_paper_eq9_overloads(
            relative_position, relative_velocity, missile_velocity)
        assert math.isclose(actual_n_mc, expected_n_mc, rel_tol=1e-12)
        assert math.isclose(actual_n_mh, expected_n_mh, rel_tol=1e-12)


def test_missile_eq9_vertical_and_singular_geometries_are_finite_and_clipped():
    direct = compute_paper_eq9_overloads(
        [1000.0, 0.0, 100.0], [0.0, 0.0, 10.0], [300.0, 0.0, 0.0])
    singular = compute_paper_eq9_overloads(
        [0.0, 0.0, 1000.0], [1.0, 2.0, 3.0], [0.0, 0.0, 300.0])
    assert np.isfinite(direct).all()
    assert np.isfinite(singular).all()

    missile = MissileSimulator(dt=1.0 / 60.0, guidance_mode="paper_eq9")
    missile._position[:] = 0.0
    missile._velocity[:] = [300.0, 0.0, 0.0]
    missile.target_aircraft = _FakeSim(
        [1000.0, 0.0, 100.0], [0.0, 100000.0, 100000.0])
    overload, _distance = missile._guidance()
    assert np.isfinite(overload).all()
    assert np.max(np.abs(overload)) <= 30.0


def test_missile_probability_failure_terminates_with_raw_reason():
    missile = MissileSimulator(dt=1.0 / 60.0)
    missile.target_aircraft = _FakeSim(
        [100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    missile._status = MissileSimulator.LAUNCHED
    missile._t = 0.0
    missile._distance_pre = math.inf
    missile._distance_increment = deque(maxlen=5)
    missile._t_arm = 0.0
    missile._guidance = lambda: (np.zeros(2, dtype=np.float64), 1.0)
    missile._roll_hit_probability = lambda: False

    missile.run()

    assert missile.is_done
    assert not missile.is_success
    assert missile._termination_reason == "p_hit_fail"


def test_checkpoint_metadata_rejects_legacy_and_mismatched_semantics():
    cfg = Config()
    obs_dim = _compute_obs_dim(3, 3, is_red=True, obs_mode="paper_strict")
    global_dim = _compute_global_state_dim(3, "paper_strict")
    metadata = _checkpoint_metadata(cfg, obs_dim, global_dim)
    state = VanillaActor(obs_dim).state_dict()
    assert metadata["schema_version"] == "vanilla_mappo_paper_env_v5"
    assert metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert metadata["action_distribution"] == ACTION_DISTRIBUTION_VERSION
    assert metadata["actor_obs_dim"] == 60
    assert metadata["global_state_dim"] == 30
    assert metadata["actor_hidden_sizes"] == [128, 128]
    assert metadata["actor_rnn_hidden_size"] == 128
    assert metadata["recurrent_n"] == 1
    assert metadata["pid_throttle_base"] == 0.0
    assert metadata["requested_replay_buffer_size"] == cfg.replay_buffer_size
    assert metadata["rollout_horizon_per_env"] == cfg.replay_buffer_size // cfg.num_envs
    assert metadata["transitions_per_update"] == metadata["rollout_horizon_per_env"] * cfg.num_envs
    assert metadata["unused_replay_slots"] == cfg.replay_buffer_size - metadata["transitions_per_update"]
    assert metadata["total_env_steps"] == cfg.total_env_steps
    assert metadata["entropy_coef"] == cfg.entropy_coef
    assert metadata["actor_lr"] == cfg.actor_lr
    assert metadata["critic_lr"] == cfg.critic_lr
    assert metadata["n_update_epochs"] == cfg.n_update_epochs
    assert metadata["n_minibatches"] == cfg.n_minibatches
    assert metadata["gamma"] == cfg.gamma
    assert metadata["gae_lambda"] == cfg.gae_lambda
    assert metadata["clip_epsilon"] == cfg.clip_epsilon

    with np.testing.assert_raises(ValueError):
        _unpack_and_validate_checkpoint(state, metadata, "actor")
    payload = {"state_dict": state, "metadata": dict(metadata), "model_kind": "actor"}
    payload["metadata"]["obs_normalization"] = "none"
    with np.testing.assert_raises(ValueError):
        _unpack_and_validate_checkpoint(payload, metadata, "actor")

    payload = {"state_dict": state, "metadata": dict(metadata), "model_kind": "actor"}
    payload["metadata"]["schema_version"] = "vanilla_mappo_paper_env_v1"
    with np.testing.assert_raises_regex(ValueError, "schema_version"):
        _unpack_and_validate_checkpoint(payload, metadata, "actor")

    payload = {"state_dict": state, "metadata": dict(metadata), "model_kind": "actor"}
    payload["metadata"]["action_distribution"] = "tanh_squashed_normal_v1"
    with np.testing.assert_raises_regex(ValueError, "action_distribution"):
        _unpack_and_validate_checkpoint(payload, metadata, "actor")

    payload = {"state_dict": state, "metadata": dict(metadata), "model_kind": "actor"}
    payload["metadata"]["altitude_reward_config"] = dict(
        metadata["altitude_reward_config"])
    payload["metadata"]["altitude_reward_config"]["h_max_m"] += 1.0
    with np.testing.assert_raises_regex(ValueError, "altitude_reward_config"):
        _unpack_and_validate_checkpoint(payload, metadata, "actor")


def test_vanilla_3v3_paper_presets_and_rollout_layout():
    from configs.experiment_presets import EXPERIMENT_PRESETS

    main = EXPERIMENT_PRESETS["vanilla_3v3_paper_main"]
    diag = EXPERIMENT_PRESETS["vanilla_3v3_paper_100k_diag"]
    assert main["total_env_steps"] == 15_000_000
    for preset in (main, diag):
        assert preset["num_red"] == preset["num_blue"] == 3
        assert preset["num_envs"] == 32
        assert preset["replay_buffer_size"] == 2_000
        assert preset["max_episode_length"] == 1_400
        assert preset["actor_lr"] == 2e-4
        assert preset["critic_lr"] == 5e-4
        assert preset["entropy_coef"] == 0.05
        assert preset["mlp_hidden"] == 128
        assert preset["rnn_hidden_size"] == 128
        assert preset["obs_mode"] == "paper_strict"
        assert preset["obs_normalization"] == "paper_fixed_v1"
        assert preset["pid_profile"] == "paper"
        assert preset["pid_throttle_base"] == 0.0
        assert preset["reward_mode"] == "paper_joint"
        assert preset["missile_guidance_mode"] == "paper_eq9"
        assert preset["blue_policy_profile"] == "paper_pursuit"
        assert preset["enable_blue_gcas"] is False
    assert diag["total_env_steps"] == 100_000
    assert diag["device"] == "auto"
    assert diag["checkpoint_dir"] == "checkpoints/vanilla_3v3_paper_100k_diag"
    assert diag["log_file"] == "logs/vanilla_3v3_paper_100k_diag.csv"
    assert diag["results_file"] == "results/vanilla_3v3_paper_100k_diag.csv"

    layout = _rollout_layout(2_000, 32)
    assert layout == {
        "requested_replay_buffer_size": 2_000,
        "rollout_horizon_per_env": 62,
        "transitions_per_update": 1_984,
        "unused_replay_slots": 16,
    }
    with pytest.raises(ValueError):
        _rollout_layout(31, 32)


def _evaluation_row(**overrides):
    row = {
        "RedWin": 0, "BlueWin": 0, "Draw": 1,
        "RedAlive": 1, "BlueAlive": 1,
        "RedDeathsMissile": 0, "RedDeathsCrash": 0, "RedDeathsOther": 0,
        "BlueDeathsMissile": 0, "BlueDeathsCrash": 0, "BlueDeathsOther": 0,
        "RedMissilesFired": 0, "BlueMissilesFired": 0,
        "RedMissileHits": 0, "BlueMissileHits": 0,
    }
    row.update(overrides)
    return row


def test_evaluation_summary_uses_aggregate_kd_and_true_rwr():
    from evaluate_vanilla_mappo import _aggregate_evaluation_summary

    rows = [
        _evaluation_row(
            RedWin=1, BlueWin=0, Draw=0,
            RedDeathsMissile=1, BlueDeathsMissile=3),
        _evaluation_row(
            RedWin=0, BlueWin=0, Draw=1,
            RedDeathsCrash=3, BlueDeathsMissile=0),
    ]
    summary = _aggregate_evaluation_summary(rows)
    assert math.isclose(summary["KD_Red_AllDeaths"], 3.0 / 4.0)
    assert not math.isclose(summary["KD_Red_AllDeaths"], (3.0 + 0.0) / 2.0)
    assert math.isinf(summary["RWR"])
    assert summary["RWRDenominatorZero"] is True

    all_draw = _aggregate_evaluation_summary([_evaluation_row()])
    assert all_draw["RWR"] == 0.0
    assert all_draw["RWRDenominatorZero"] is True

    finite, zero = _ratio_with_denominator_zero(2, 4)
    assert finite == 0.5 and zero is False


def test_attention_presets_distinguish_2v2_probes_from_6v6_canonical():
    from configs.experiment_presets import EXPERIMENT_PRESETS

    assert "attention_2v2_brma_paper_main" not in EXPERIMENT_PRESETS
    assert "attention_2v2_attn_nobrma_paper_baseline" not in EXPERIMENT_PRESETS
    assert EXPERIMENT_PRESETS["attention_2v2_brma_probe"]["num_red"] == 2
    for name in (
        "attention_6v6_brma_paper_main",
        "attention_6v6_attn_nobrma_paper_baseline",
    ):
        preset = EXPERIMENT_PRESETS[name]
        assert preset["num_red"] == preset["num_blue"] == 6
        assert preset["num_envs"] == 32
        assert preset["max_episode_length"] == 1400
        assert preset["total_env_steps"] == 10_000_000
        assert preset["reward_mode"] == "paper_joint"
        assert preset["pid_profile"] == "paper"
        assert preset["missile_guidance_mode"] == "paper_eq9"


def test_joint_mappo_short_buffer_forward_and_backward_is_finite():
    cfg = Config()
    cfg.n_update_epochs = 1
    cfg.n_minibatches = 1
    cfg.num_red = 2
    cfg.num_blue = 1
    cfg.obs_mode = "paper_strict"
    obs_dim = _compute_obs_dim(2, 1, is_red=True, obs_mode="paper_strict")
    global_dim = _compute_global_state_dim(2, "paper_strict")
    actor = VanillaActor(obs_dim, hidden=16, rnn_hidden=8)
    critic = CentralizedCritic(global_dim, hidden=16)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    buffer = RolloutBuffer(3, 1, 2, 3, 8)

    for t in range(3):
        buffer.global_states[t][0] = np.zeros(global_dim, dtype=np.float32)
        buffer.joint_rewards[t, 0] = float(t + 1)
        buffer.team_values[t, 0] = 0.0
        for agent_idx in range(2):
            alive = not (agent_idx == 1 and t > 0)
            buffer.obs[t][0][agent_idx] = np.zeros(obs_dim, dtype=np.float32)
            buffer.actions[t, 0, agent_idx] = 0.0
            buffer.log_probs[t, 0, agent_idx] = 0.0
            buffer.alive[t, 0, agent_idx] = alive
    buffer.episode_dones[-1, 0] = 1.0
    buffer.bootstrap_value[0] = 0.0

    stats = ppo_update(
        actor, critic, actor_opt, critic_opt, buffer, cfg, torch.device("cpu"))
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["entropy_bonus"])
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])
    assert np.isfinite(stats["entropy"])
    assert stats["ActorUpdateAttempts"] == 1
    assert stats["ActorUpdatesApplied"] == 1
    assert stats["ActorUpdatesSkipped"] == 0
    assert stats["CriticUpdateAttempts"] == 1
    assert stats["CriticUpdatesApplied"] == 1
    assert stats["CriticUpdatesSkipped"] == 0


def test_ppo_update_records_skipped_actor_and_critic_updates(monkeypatch):
    cfg = Config()
    cfg.n_update_epochs = 1
    cfg.n_minibatches = 1
    cfg.num_red = 1
    cfg.num_blue = 1
    obs_dim = _compute_obs_dim(1, 1, is_red=True, obs_mode="paper_strict")
    global_dim = _compute_global_state_dim(1, "paper_strict")
    actor = VanillaActor(obs_dim, hidden=16, rnn_hidden=8)
    critic = CentralizedCritic(global_dim, hidden=16)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    buffer = RolloutBuffer(2, 1, 1, 3, 8)
    for t in range(2):
        buffer.global_states[t][0] = np.zeros(global_dim, dtype=np.float32)
        buffer.joint_rewards[t, 0] = 1.0
        buffer.team_values[t, 0] = 0.0
        buffer.obs[t][0][0] = np.zeros(obs_dim, dtype=np.float32)
        buffer.actions[t, 0, 0] = 0.0
        buffer.log_probs[t, 0, 0] = 0.0
        buffer.alive[t, 0, 0] = True
    buffer.episode_dones[-1, 0] = 1.0
    buffer.bootstrap_value[0] = 0.0

    calls = {"count": 0}

    def always_skip(_module):
        calls["count"] += 1
        return True

    monkeypatch.setattr(tvm, "_grad_has_nan", always_skip)
    stats = ppo_update(
        actor, critic, actor_opt, critic_opt, buffer, cfg, torch.device("cpu"))
    assert stats["ActorUpdateAttempts"] == 1
    assert stats["ActorUpdatesApplied"] == 0
    assert stats["ActorUpdatesSkipped"] == 1
    assert stats["CriticUpdateAttempts"] == 1
    assert stats["CriticUpdatesApplied"] == 0
    assert stats["CriticUpdatesSkipped"] == 1


def _write_diag_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _diag_rows(**overrides):
    base = []
    for step in (100, 200, 300, 400, 500):
        row = {
            "Step": step,
            "ActorLoss": 0.1,
            "PolicyLoss": 0.2,
            "EntropyBonus": 0.1,
            "PolicyEntropy": 3.5 - step * 0.0001,
            "CriticLoss": 10.0 - step * 0.01,
            "ActionStdMean": 0.3,
            "ActionStdMin": 0.3,
            "ActionStdMax": 0.3,
            "ActionLogStdMean": math.log(0.3),
            "ActionStdDeltaFromInit": 0.0,
            "ActionStdGrowthRatio": 1.0,
            "StateDependentStdMean": 0.3,
            "StateDependentStdMin": 0.3,
            "StateDependentStdMax": 0.3,
            "StateDependentStdLowerBoundFrac": 0.0,
            "StateDependentStdUpperBoundFrac": 0.0,
            "RawActionOutOfBoundsFrac": 0.0,
            "RawActionOutOfBoundsFracPitch": 0.0,
            "RawActionOutOfBoundsFracHeading": 0.0,
            "RawActionOutOfBoundsFracVelocity": 0.0,
            "EnvActionNearBoundFrac": 0.0,
            "EnvActionNearBoundFracPitch": 0.0,
            "EnvActionNearBoundFracHeading": 0.0,
            "EnvActionNearBoundFracVelocity": 0.0,
            "ActorUpdatesSkipped": 0,
            "ActorUpdateAttempts": 1,
            "ActorUpdatesApplied": 1,
            "CriticUpdatesSkipped": 0,
            "CriticUpdateAttempts": 1,
            "CriticUpdatesApplied": 1,
            "InvalidNumericalEpisodes": 0,
            "InvalidTransitionsDropped": 0,
            "RedMeanReward": float(step) * 0.01,
            "WinRateRecent": 0.5,
            "WinRateCumul": 0.5,
            "Episodes": step // 100,
            "RedWins": step // 200,
            "BlueWins": step // 300,
            "Draws": 0,
            "RedMissiles": 1,
            "BlueMissiles": 1,
            "LaunchDiagRedGeometryOk": 1,
            "LaunchDiagBlueGeometryOk": 1,
            "LaunchDiagRedLockMature": 1,
            "LaunchDiagBlueLockMature": 1,
            "LaunchDiagRedLaunches": 1,
            "LaunchDiagBlueLaunches": 1,
            "RedMissileHitRate": 0.1,
            "BlueMissileHitRate": 0.1,
            "RedDeathsMissile": 0,
            "RedDeathsCrash": 0,
            "BlueDeathsMissile": 0,
            "BlueDeathsCrash": 0,
        }
        row.update(overrides)
        base.append(row)
    return base


def test_diagnostic_report_generates_pass_review_and_fail(tmp_path):
    from scripts.report_vanilla_mappo_diagnostic import (
        analyze,
        load_training_csv,
        write_plots,
        write_report,
    )

    pass_csv = tmp_path / "pass.csv"
    _write_diag_csv(pass_csv, _diag_rows())
    rows = load_training_csv(pass_csv)
    status, fail, review, summaries = analyze(rows, expected_steps=500)
    assert status == "PASS"
    assert not fail
    assert any("10k" in item for item in review)
    plot_dir = tmp_path / "plots"
    write_plots(rows, plot_dir)
    assert (plot_dir / "action_std.png").exists()

    report_md = tmp_path / "report.md"
    write_report(report_md, status, fail, review, summaries, rows, {}, {})
    text = report_md.read_text(encoding="utf-8")
    assert text.startswith("# PASS")
    assert "- Episodes: 5" in text
    assert "- Episodes: 15" not in text

    review_csv = tmp_path / "review.csv"
    _write_diag_csv(review_csv, _diag_rows(
        ActionStdMean=0.39,
        ActionStdDeltaFromInit=0.09,
        ActionStdGrowthRatio=1.3,
    ))
    status, fail, review, _summaries = analyze(
        load_training_csv(review_csv), expected_steps=500)
    assert status == "PASS"
    assert not fail
    assert any("10k" in item for item in review)

    fail_csv = tmp_path / "fail.csv"
    _write_diag_csv(fail_csv, _diag_rows(
        ActorLoss="NaN",
        ActorUpdatesSkipped=1,
        RawActionOutOfBoundsFrac=0.06,
    ))
    status, fail, _review, _summaries = analyze(
        load_training_csv(fail_csv), expected_steps=500)
    assert status == "FAIL"
    assert any("NaN" in item or "Inf" in item for item in fail)
    assert any("Actor" in item and "跳过" in item for item in fail)


def _diag_10k_rows(**overrides):
    template = _diag_rows()[0]
    rows = []
    for index, step in enumerate(range(1000, 10001, 1000), start=1):
        row = dict(template)
        row.update({
            "Step": step,
            "Episodes": index,
            "RedWins": 0,
            "BlueWins": index,
            "WinRateRecent": 0.0,
            "WinRateCumul": 0.0,
            "RedMeanReward": -10.0,
            "LaunchDiagRedGeometryOk": 100 if step <= 3000 else 10,
            "LaunchDiagRedLockMature": 50 if step <= 3000 else 5,
            "LaunchDiagRedLaunches": 1,
            "RedMissileHits": 0,
            "RedAliveMean": 1.0,
            "BlueAliveMean": 3.0,
        })
        row.update(overrides)
        rows.append(row)
    return rows


def test_zero_early_wins_do_not_fail_environment_health():
    from scripts.report_vanilla_mappo_diagnostic import analyze_diagnostics
    result = analyze_diagnostics(_diag_rows(
        RedWins=0, WinRateRecent=0.0, WinRateCumul=0.0), expected_steps=500)
    assert result["EnvironmentHealthStatus"] == "PASS"
    assert result["LearningEvidenceStatus"] == "INSUFFICIENT_DATA"


def test_burn_in_is_excluded_and_health_can_pass_without_clear_trend():
    from scripts.report_vanilla_mappo_diagnostic import analyze_diagnostics
    result = analyze_diagnostics(_diag_10k_rows(), expected_steps=10_000)
    assert result["EnvironmentHealthStatus"] == "PASS"
    assert result["LearningEvidenceStatus"] == "NO_CLEAR_TREND_YET"
    assert result["window_row_counts"]["burn_in"] == 3
    assert result["window_row_counts"]["middle"] > 0
    assert result["window_row_counts"]["late"] > 0


def test_health_pass_and_optimization_warning_are_independent():
    from scripts.report_vanilla_mappo_diagnostic import analyze_diagnostics
    result = analyze_diagnostics(_diag_10k_rows(
        StateDependentStdMean=0.58,
        StateDependentStdMin=0.55,
        StateDependentStdMax=0.59,
        StateDependentStdUpperBoundFrac=0.5,
    ), expected_steps=10_000)
    assert result["EnvironmentHealthStatus"] == "PASS"
    assert result["LearningEvidenceStatus"] == "OPTIMIZATION_WARNING"


def test_speed_projection_failure_fails_health_not_learning_semantics():
    from scripts.report_vanilla_mappo_diagnostic import analyze_diagnostics
    result = analyze_diagnostics(_diag_rows(
        MaximumSpeedAfterLimiterMps=600.02), expected_steps=500)
    assert result["EnvironmentHealthStatus"] == "FAIL"
    assert any("600.01" in reason
               for reason in result["EnvironmentHealthReasons"])


def test_short_jsbsim_paper_pid_steps_reduce_heading_pitch_and_speed_error():
    def run_case(kind, frames):
        env = UavCombatEnv(
            max_num_red=1, max_num_blue=1, pid_profile="paper",
            suppress_jsbsim_output=True)
        try:
            env.reset()
            sim = env.red_planes["red_0"]
            pid = env.pid_controllers["red_0"]
            initial_rpy = sim.get_rpy().copy()
            initial_speed = float(np.linalg.norm(sim.get_velocity()))
            target_pitch = float(initial_rpy[1])
            target_heading = float(initial_rpy[2])
            target_speed = initial_speed
            if kind == "heading":
                target_heading += math.radians(5.0)
            elif kind == "pitch":
                target_pitch += math.radians(2.0)
            else:
                target_speed += 10.0

            def error():
                if kind == "heading":
                    return abs((target_heading - sim.get_rpy()[2] + math.pi)
                               % (2 * math.pi) - math.pi)
                if kind == "pitch":
                    return abs(target_pitch - sim.get_rpy()[1])
                return abs(target_speed - np.linalg.norm(sim.get_velocity()))

            initial_error = float(error())
            controls = []
            for _ in range(frames):
                velocity = sim.get_velocity()
                control = pid.compute_control(
                    sim.get_rpy(), float(np.linalg.norm(velocity)),
                    target_pitch, target_heading, target_speed,
                    np.array([velocity[0], velocity[1], -velocity[2]]))
                controls.append(control)
                for prop, value in zip((
                        "fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
                        "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"), control):
                    sim.set_property_value(prop, value)
                sim.run()
            controls = np.asarray(controls)
            assert np.isfinite(controls).all()
            assert np.max(np.abs(controls)) <= 1.0
            assert float(error()) < initial_error
        finally:
            env.close()

    run_case("heading", 180)
    run_case("pitch", 180)
    run_case("speed", 600)
