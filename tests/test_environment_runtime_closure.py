import math
import inspect

import numpy as np

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    environment_config_snapshot,
)
from my_uav_env import UavCombatEnv
from rule_based_agent import _blue_simple_pursuit_action_impl
from my_uav_env.sensors import (
    bilinear_rcs_m2,
    radar_diagnostic,
    select_most_dangerous_missile,
    target_body_aspect,
)


class FakeAircraft:
    def __init__(self, uid, position, velocity=(0, 0, 0), rpy=(0, 0, 0)):
        self.uid = uid
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_rpy(self):
        return self._rpy


class FakeMissile(FakeAircraft):
    def __init__(self, uid, position, velocity, target):
        super().__init__(uid, position, velocity)
        self.target_aircraft = target
        self.is_alive = True


def test_environment_config_snapshot_fingerprint_is_stable_and_scale_sensitive():
    a = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, num_red=3, num_blue=3,
        sim_freq=60, agent_interaction_steps=12, seed=7)
    b = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, num_red=3, num_blue=3,
        sim_freq=60, agent_interaction_steps=12, seed=7)
    c = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, num_red=6, num_blue=6,
        sim_freq=60, agent_interaction_steps=12, seed=7)
    assert a["environment_config_fingerprint"] == b["environment_config_fingerprint"]
    assert a["environment_config_fingerprint"] != c["environment_config_fingerprint"]


def test_physical_arena_and_reward_boundary_are_separate():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1)
    assert env.arena_half_width_m == 50_000.0
    assert env.reward_boundary_half_width_m == 40_000.0
    assert env.arena_altitude_min_m == 0.0
    assert env.arena_altitude_max_m == 10_000.0


def test_rcs_nodes_midpoint_and_fourth_root_range():
    cfg = DEFAULT_PAPER_ENVIRONMENT_CONFIG
    front = bilinear_rcs_m2(0.0, 0.0, cfg.rcs.azimuth_grid_deg.value,
                            cfg.rcs.elevation_grid_deg.value, cfg.rcs.table_m2.value)
    side = bilinear_rcs_m2(math.pi / 2, 0.0, cfg.rcs.azimuth_grid_deg.value,
                           cfg.rcs.elevation_grid_deg.value, cfg.rcs.table_m2.value)
    middle = bilinear_rcs_m2(math.pi / 4, 0.0, cfg.rcs.azimuth_grid_deg.value,
                             cfg.rcs.elevation_grid_deg.value, cfg.rcs.table_m2.value)
    assert math.isclose(front, 0.1)
    assert math.isclose(side, 2.0)
    assert front < middle < side
    observer = FakeAircraft("o", [0, 0, 0])
    target = FakeAircraft("t", [1000, 0, 0], rpy=(0, 0, math.pi))
    diag = radar_diagnostic(observer, target, cfg.radar, cfg.rcs)
    assert math.isclose(
        diag["radar_max_range_m"],
        cfg.rcs.range_constant.value * diag["interpolated_rcs_m2"] ** 0.25)


def test_target_body_aspect_distinguishes_front_and_side():
    target = FakeAircraft("t", [0, 0, 0])
    front_az, _ = target_body_aspect([1000, 0, 0], target.get_position(), target.get_rpy())
    side_az, _ = target_body_aspect([0, 1000, 0], target.get_position(), target.get_rpy())
    assert math.isclose(front_az, 0.0)
    assert math.isclose(side_az, math.pi / 2)


def test_mws_selects_minimum_positive_ttc_and_never_falls_back_to_receding():
    target = FakeAircraft("target", [0, 0, 0])
    slow = FakeMissile("slow", [1000, 0, 0], [-100, 0, 0], target)
    fast = FakeMissile("fast", [1500, 0, 0], [-500, 0, 0], target)
    selected, diag = select_most_dangerous_missile(target, [slow, fast])
    assert selected.uid == "fast"
    assert diag["time_to_closest_approach_s"] < 4.0
    receding_near = FakeMissile("near", [100, 0, 0], [100, 0, 0], target)
    receding_far = FakeMissile("far", [500, 0, 0], [100, 0, 0], target)
    selected, diag = select_most_dangerous_missile(
        target, [receding_far, receding_near])
    assert selected is None
    assert diag is None


def test_mws_filters_receding_invalid_and_non_targeting_missiles():
    target = FakeAircraft("target", [0, 0, 0])
    other = FakeAircraft("other", [0, 0, 0])
    approaching = FakeMissile(
        "approaching", [1000, 0, 0], [-200, 0, 0], target)
    receding = FakeMissile(
        "receding", [100, 0, 0], [100, 0, 0], target)
    invalid = FakeMissile(
        "invalid", [float("nan"), 0, 0], [-100, 0, 0], target)
    wrong_target = FakeMissile(
        "wrong", [50, 0, 0], [-500, 0, 0], other)
    selected, diag = select_most_dangerous_missile(
        target, [receding, invalid, wrong_target, approaching])
    assert selected is approaching
    assert diag["closing_speed_mps"] > 0.0
    assert diag["time_to_closest_approach_s"] >= 0.0
    assert diag["candidate_is_approaching"] is True


def test_mws_rejects_negative_ttc_and_exits_when_missile_turns_away():
    target = FakeAircraft("target", [0, 0, 0])
    missile = FakeMissile(
        "missile", [1000, 0, 0], [-200, 0, 0], target)
    selected, _diag = select_most_dangerous_missile(target, [missile])
    assert selected is missile
    missile._velocity[:] = [200, 0, 0]
    selected, diag = select_most_dangerous_missile(target, [missile])
    assert selected is None
    assert diag is None


def test_mws_multiple_incoming_tie_breaks_by_ttc_distance_then_uid():
    target = FakeAircraft("target", [0, 0, 0])
    later = FakeMissile("later", [1000, 0, 0], [-100, 0, 0], target)
    nearer_uid_b = FakeMissile("b", [500, 0, 0], [-100, 0, 0], target)
    nearer_uid_a = FakeMissile("a", [500, 0, 0], [-100, 0, 0], target)
    selected, diag = select_most_dangerous_missile(
        target, [later, nearer_uid_b, nearer_uid_a])
    assert selected is nearer_uid_a
    assert diag["time_to_closest_approach_s"] == 5.0


def test_reset_seed_reproduces_awacs_coarse_track_and_snapshot():
    env = UavCombatEnv(max_num_red=1, max_num_blue=1)
    try:
        env.reset(seed=123)
        env.red_planes["red_0"]._posture[2] = 0.0
        first = env._get_sensor_track(env.red_planes["red_0"], env.blue_planes["blue_0"])
        assert first.source in ("awacs_coarse", "rws_awacs_fused")
        assert not np.allclose(
            first.position_estimate, env.blue_planes["blue_0"].get_position())
        first_position = first.position_estimate.copy()
        snapshot = env._get_info()["__environment_config__"]
        env.reset(seed=123)
        env.red_planes["red_0"]._posture[2] = 0.0
        second = env._get_sensor_track(env.red_planes["red_0"], env.blue_planes["blue_0"])
        assert np.allclose(first_position, second.position_estimate)
        assert snapshot["environment_config_fingerprint"] == env._get_info()[
            "__environment_config__"]["environment_config_fingerprint"]
    finally:
        env.close()


def test_initial_centroid_range_and_headings_for_3v3():
    env = UavCombatEnv(max_num_red=3, max_num_blue=3)
    try:
        env.reset(seed=5)
        red = np.mean([s.get_position() for s in env.red_planes.values()], axis=0)
        blue = np.mean([s.get_position() for s in env.blue_planes.values()], axis=0)
        assert math.isclose(np.linalg.norm(red - blue), 10_000.0, rel_tol=0.0, abs_tol=30.0)
        red_heading = next(iter(env.red_planes.values())).get_rpy()[2]
        blue_heading = next(iter(env.blue_planes.values())).get_rpy()[2]
        delta = abs((red_heading - blue_heading + math.pi) % (2 * math.pi) - math.pi)
        assert math.isclose(delta, math.pi, abs_tol=1e-5)
    finally:
        env.close()


def test_paper_pursuit_same_altitude_has_zero_pitch_and_stable_cruise_speed():
    obs = {
        "ego_state": np.array([0, 0, 6096, 250, 0, 0, 0, 0, 0, 0], np.float32),
        "ally_states": np.zeros((0, 10), np.float32),
        "enemy_states": np.array([[5000, 0, 0, 0, 0, 250, 0, 0, 0, 5000]], np.float32),
        "alive_mask": np.array([1, 1]),
        "velocity": np.array([250, 0, 0], np.float32),
        "altitude": np.array([6096], np.float32),
    }
    action = _blue_simple_pursuit_action_impl(
        obs, 1, 1, 0, 0, own_heading=0.0, paper_profile=True)
    assert abs(action[0]) < 1e-6
    expected_speed = 2.0 * (250.0 - 102.0) / 306.0 - 1.0
    assert math.isclose(action[2], expected_speed, abs_tol=1e-6)


def test_paper_constraint_does_not_write_jsbsim_velocity_state():
    source = inspect.getsource(UavCombatEnv._enforce_aircraft_constraints)
    assert "velocities/u-fps" not in source
    assert "velocities/v-fps" not in source
    assert "velocities/w-fps" not in source
