from __future__ import annotations

import copy

import numpy as np

from my_uav_env.blue_policy_profiles import (
    BLUE_POLICY_PROFILES,
    BluePolicyController,
)
from rule_based_agent import blue_coordinated_actions
from rule_based_agent import _paper_cruise_speed_action
from my_uav_env.env import UavCombatEnv
from evaluate_blue_profile_matrix import MATRIX_FIELDS
from evaluate_vanilla_mappo import EVALUATION_FIELDNAMES
from train_vanilla_mappo import (
    BLUE_POLICY_DIAG_CSV_FIELDS,
    Config,
    _compute_global_state_dim,
    _compute_obs_dim,
    _unpack_and_validate_checkpoint,
)
from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    environment_config_snapshot,
)
import pytest


def _obs(enemy_ranges=(10000.0, 12000.0, 14000.0), alive=(1, 1, 1),
         heading=0.0, altitude=6000.0, v_up=0.0, warning=0.0):
    ego = np.zeros(10, dtype=np.float32)
    ego[2] = altitude
    ego[3] = 250.0
    ego[6] = heading
    enemies = np.zeros((3, 10), dtype=np.float32)
    for index, distance in enumerate(enemy_ranges):
        enemies[index, 0] = distance
        enemies[index, 5] = 250.0
        enemies[index, 9] = distance
    return {
        "ego_state": ego,
        "ally_states": np.zeros((2, 10), dtype=np.float32),
        "enemy_states": enemies,
        "alive_mask": np.asarray((1, 1, 1, *alive), dtype=np.int64),
        "death_mask": np.asarray((1, 1, 1, *alive), dtype=np.int64),
        "missile_warning": np.asarray([warning], dtype=np.float32),
        "altitude": np.asarray([altitude], dtype=np.float32),
        "velocity": np.asarray([250.0, 0.0, v_up], dtype=np.float32),
    }


def _blue_obs(**kwargs):
    return {f"blue_{index}": _obs(**kwargs) for index in range(3)}


def _reset(controller):
    controller.reset(
        ["blue_0", "blue_1", "blue_2"],
        ["red_0", "red_1", "red_2"],
        {"blue_0": 0.0, "blue_1": 0.2, "blue_2": -0.2},
        {"blue_0": 6000.0, "blue_1": 6000.0, "blue_2": 6000.0},
    )


def _act(controller, obs, step=0, engaged=None, own_alive=None):
    headings = {"blue_0": 0.0, "blue_1": 0.2, "blue_2": -0.2}
    positions = {key: np.zeros(3) for key in headings}
    return controller.act(
        obs, 3, 3, engaged or set(), positions, headings, step,
        own_alive=own_alive)


def test_profiles_and_default_remain_paper_pursuit():
    assert BLUE_POLICY_PROFILES == (
        "paper_pursuit", "fixed_pair_pursuit_v1", "fixed_pair_no_mws_v1",
        "fixed_pair_hold_after_kill_v1", "frozen_route_blue_v1",
        "paper_minimal_fixed_pair_v1", "paper_minimal_straight_patrol_v1")
    assert Config.blue_policy_profile == "paper_pursuit"


def test_paper_pursuit_controller_delegates_exact_baseline_actions():
    obs = _blue_obs(enemy_ranges=(9000.0, 11000.0, 13000.0))
    controller = BluePolicyController("paper_pursuit")
    _reset(controller)
    positions = {f"blue_{i}": np.zeros(3) for i in range(3)}
    headings = {"blue_0": 0.0, "blue_1": 0.2, "blue_2": -0.2}
    expected = blue_coordinated_actions(
        obs, 3, 3, engaged_targets={"red_0"}, own_positions=positions,
        own_headings=headings, pursuit_mode="paper_pursuit")
    actual = controller.act(
        obs, 3, 3, {"red_0"}, positions, headings, current_step=0)
    for blue_id in expected:
        np.testing.assert_array_equal(actual[blue_id], expected[blue_id])


def test_fixed_pair_mapping_and_distance_or_engaged_do_not_switch():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    _act(controller, _blue_obs(), engaged={"red_0", "red_1", "red_2"})
    first = controller.snapshot_episode_diagnostics()
    assert [row["assigned_target_id"] for row in first["per_blue"]] == [
        "red_0", "red_1", "red_2"]
    _act(controller, _blue_obs(enemy_ranges=(15000.0, 500.0, 700.0)))
    diag = controller.snapshot_episode_diagnostics()
    assert [row["assigned_target_id"] for row in diag["per_blue"]] == [
        "red_0", "red_1", "red_2"]
    assert diag["blue_target_switches_total"] == 0
    assert diag["blue_distance_triggered_switches"] == 0
    assert diag["blue_engaged_triggered_switches"] == 0


def test_fixed_pair_awacs_track_keeps_target_and_target_death_cycles():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    obs = _blue_obs()
    obs["blue_0"]["enemy_states"][0, 5:9] = 0.0
    _act(controller, obs)
    assert controller.current_targets["blue_0"] == "red_0"
    dead = _blue_obs(alive=(0, 1, 1))
    _act(controller, dead)
    diag = controller.snapshot_episode_diagnostics()
    row = next(row for row in diag["per_blue"] if row["blue_id"] == "blue_0")
    assert row["assigned_target_id"] == "red_1"
    assert row["last_switch_reason"] == "target_dead"
    assert row["target_switch_count"] == 1
    assert diag["blue_target_dead_switches"] == 1


def test_fixed_pair_reset_restores_initial_mapping():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    _act(controller, _blue_obs(alive=(0, 1, 1), warning=1.0))
    controller.record_executed_heading("blue_0", 1.0, "mws_override")
    generation = controller.episode_generation
    _reset(controller)
    assert controller.episode_generation == generation + 1
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1", "blue_2": "red_2"}
    assert sum(controller.target_switch_counts.values()) == 0
    assert controller.mws_detected_agent_decisions == 0
    assert controller.mws_override_agent_decisions == 0
    assert controller.last_base_commands == {}
    assert controller.last_executed_headings == {}


def test_no_mws_records_detection_without_action_override():
    no_mws = BluePolicyController("fixed_pair_no_mws_v1")
    pursuit = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(no_mws)
    _reset(pursuit)
    obs = _blue_obs(warning=1.0)
    a = _act(no_mws, copy.deepcopy(obs))
    b = _act(pursuit, copy.deepcopy(obs))
    for blue_id in a:
        np.testing.assert_array_equal(a[blue_id], b[blue_id])
    diag = no_mws.snapshot_episode_diagnostics()
    assert diag["blue_mws_detected_agent_decisions"] == 3
    assert diag["blue_mws_override_agent_decisions"] == 0
    assert not no_mws.blue_mws_override_enabled
    assert pursuit.blue_mws_override_enabled
    for blue_id in ("blue_0", "blue_1", "blue_2"):
        pursuit.record_executed_heading(blue_id, 0.0, "mws_override")
    pursuit_diag = pursuit.snapshot_episode_diagnostics()
    assert pursuit_diag["blue_mws_detected_agent_decisions"] == 3
    assert pursuit_diag["blue_mws_override_agent_decisions"] == 3


def test_hold_after_kill_does_not_reassign_and_holds_cruise():
    controller = BluePolicyController("fixed_pair_hold_after_kill_v1")
    _reset(controller)
    actions = _act(controller, _blue_obs(alive=(0, 1, 1)))
    diag = controller.snapshot_episode_diagnostics()
    row = next(row for row in diag["per_blue"] if row["blue_id"] == "blue_0")
    assert row["assigned_target_id"] is None
    assert row["target_switch_count"] == 0
    assert row["target_speed_cmd_mps"] == 250.0
    assert actions["blue_0"][0] == 0.0


def test_fixed_target_valid_uses_only_requested_target():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    obs = _blue_obs(enemy_ranges=(10000.0, 500.0, 700.0))
    obs["blue_0"]["enemy_states"][0, :3] = (10000.0, 10000.0, 0.0)
    actions = _act(controller, obs)
    assert actions["blue_0"][1] > 0.2
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["assigned_target_id"] == "red_0"
    assert row["assignment_reason"] == "fixed_pair"


def test_fixed_target_track_unavailable_holds_without_nearest_fallback_then_recovers():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    obs = _blue_obs(enemy_ranges=(10000.0, 500.0, 700.0), heading=0.4)
    obs["blue_0"]["enemy_states"][0] = 0.0
    headings = {"blue_0": 0.4, "blue_1": 0.2, "blue_2": -0.2}
    positions = {key: np.zeros(3) for key in headings}
    actions = controller.act(obs, 3, 3, set(), positions, headings, 0)
    np.testing.assert_allclose(
        actions["blue_0"],
        np.asarray([0.0, 0.4 / np.pi, _paper_cruise_speed_action(250.0)]),
        rtol=0.0, atol=1e-7)
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["assigned_target_id"] == "red_0"
    assert row["target_switch_count"] == 0
    assert row["assignment_reason"] == "track_unavailable_hold"

    recovered = _blue_obs(enemy_ranges=(10000.0, 500.0, 700.0), heading=0.4)
    recovered["blue_0"]["enemy_states"][0, :3] = (10000.0, 10000.0, 0.0)
    actions = controller.act(recovered, 3, 3, set(), positions, headings, 1)
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["assigned_target_id"] == "red_0"
    assert row["target_switch_count"] == 0
    assert row["assignment_reason"] == "fixed_pair"
    assert actions["blue_0"][1] > 0.2


def test_frozen_route_phases_wrap_and_have_no_target_or_mws_override():
    controller = BluePolicyController("frozen_route_blue_v1")
    controller.reset(
        ["blue_0", "blue_1", "blue_2"], ["red_0", "red_1", "red_2"],
        {"blue_0": 0.0, "blue_1": np.deg2rad(170),
         "blue_2": np.deg2rad(-170)},
        {"blue_0": 6000.0, "blue_1": 6000.0, "blue_2": 6000.0})
    expected = {
        0: ("A", np.deg2rad(170)),
        76: ("B", np.deg2rad(-145)),
        176: ("C", np.deg2rad(125)),
        276: ("D", np.deg2rad(170)),
    }
    for step, (phase, heading) in expected.items():
        actions = _act(controller, _blue_obs(warning=1.0), step=step)
        diag = controller.snapshot_episode_diagnostics()
        row = next(row for row in diag["per_blue"] if row["blue_id"] == "blue_1")
        assert row["route_phase"] == phase
        assert np.isclose(actions["blue_1"][1] * np.pi, heading)
        assert row["assigned_target_id"] is None
        assert row["target_switch_count"] == 0
        assert not row["mws_action_override"]


def test_frozen_route_is_invariant_to_enemy_observations():
    a = BluePolicyController("frozen_route_blue_v1")
    b = BluePolicyController("frozen_route_blue_v1")
    _reset(a)
    _reset(b)
    obs_a = _blue_obs(enemy_ranges=(500.0, 700.0, 900.0), alive=(1, 1, 1))
    obs_b = _blue_obs(enemy_ranges=(90000.0, 80000.0, 70000.0), alive=(0, 1, 0),
                      warning=1.0)
    for step in (0, 76, 176, 276):
        actions_a = _act(a, copy.deepcopy(obs_a), step=step,
                         engaged={"red_0"})
        actions_b = _act(b, copy.deepcopy(obs_b), step=step,
                         engaged={"red_2"})
        for blue_id in actions_a:
            np.testing.assert_array_equal(actions_a[blue_id], actions_b[blue_id])


def test_frozen_route_does_not_read_enemy_or_mws_observation_keys():
    class Forbidden(dict):
        def get(self, key, default=None):
            if key in {"enemy_states", "missile_warning"}:
                raise AssertionError(f"forbidden observation read: {key}")
            return super().get(key, default)

    controller = BluePolicyController("frozen_route_blue_v1")
    _reset(controller)
    obs = {blue_id: Forbidden(value) for blue_id, value in _blue_obs().items()}
    actions = _act(controller, obs, step=76, engaged={"red_0"})
    assert all(np.isfinite(action).all() for action in actions.values())


def test_frozen_altitude_recovery_uses_only_ownship_fields():
    controller = BluePolicyController("frozen_route_blue_v1")
    _reset(controller)
    high = _blue_obs(altitude=9000.0, v_up=20.0)
    action = _act(controller, high)["blue_0"]
    assert action[0] < 0.0
    diag = controller.snapshot_episode_diagnostics()
    assert diag["blue_altitude_recovery_frames"] == 3


def test_eight_controller_instances_are_isolated_and_reproducible():
    controllers = [BluePolicyController("fixed_pair_pursuit_v1") for _ in range(8)]
    for controller in controllers:
        _reset(controller)
    _act(controllers[0], _blue_obs(alive=(0, 1, 1)))
    assert controllers[0].current_targets["blue_0"] == "red_1"
    assert all(c.current_targets["blue_0"] == "red_0" for c in controllers[1:])
    outputs = [_act(c, _blue_obs()) for c in controllers[1:]]
    for other in outputs[1:]:
        for blue_id in outputs[0]:
            np.testing.assert_array_equal(outputs[0][blue_id], other[blue_id])


def test_dimensions_and_training_diagnostic_fields_are_unchanged_or_present():
    assert _compute_obs_dim(3, 3, True, obs_mode="paper_strict") == 60
    assert _compute_global_state_dim(3, "paper_strict") == 30
    assert "blue_mws_override_agent_decisions" in BLUE_POLICY_DIAG_CSV_FIELDS
    assert "blue_mws_detected_agent_decisions" in BLUE_POLICY_DIAG_CSV_FIELDS
    assert "blue_base_heading_command_discontinuities" in BLUE_POLICY_DIAG_CSV_FIELDS
    assert "blue_executed_heading_command_discontinuities" in BLUE_POLICY_DIAG_CSV_FIELDS
    assert not any("mws" in field and "frames" in field
                   for field in BLUE_POLICY_DIAG_CSV_FIELDS)
    assert "blue_distance_triggered_switches" in BLUE_POLICY_DIAG_CSV_FIELDS


class _FakeIncomingMissile:
    uid = "red_missile_0"

    def get_position(self):
        return np.asarray([0.0, 1000.0, 6000.0])


class _FakeBlueSim:
    is_alive = True

    def __init__(self):
        self.incoming = None
        self.altitude = 6000.0

    def get_rpy(self):
        return np.zeros(3)

    def check_missile_warning(self):
        return self.incoming

    def get_geodetic(self):
        return (0.0, 0.0, self.altitude)

    def get_position(self):
        return np.asarray([0.0, 0.0, 6000.0])

    def get_velocity(self):
        return np.asarray([250.0, 0.0, 0.0])


def _diagnostic_env_and_action():
    env = UavCombatEnv.__new__(UavCombatEnv)
    sim = _FakeBlueSim()
    env.blue_planes = {"blue_0": sim}
    env.red_planes = {}
    env.blue_policy_controller = BluePolicyController("fixed_pair_pursuit_v1")
    env.blue_policy_controller.reset(
        ["blue_0"], ["red_0"], {"blue_0": 0.0}, {"blue_0": 6000.0})
    env._evasion_diagnostics = {
        "blue_0": {"activations": 0, "active_frames": 0, "active": False}}
    env.enable_gcas_for_blue = False
    obs = _obs(enemy_ranges=(10000.0, 12000.0, 14000.0))
    action = env.blue_policy_controller.act(
        {"blue_0": obs}, 1, 1, set(), {"blue_0": np.zeros(3)},
        {"blue_0": 0.0}, 0, mws_detected={"blue_0": False})["blue_0"]
    return env, sim, obs, action


def test_executed_heading_tracks_base_policy_without_mws():
    env, _sim, _obs_one, action = _diagnostic_env_and_action()
    targets = env._parse_actions({"blue_0": action})
    row = env.blue_policy_controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["executed_command_source"] == "base_policy"
    assert row["executed_heading_command_rad"] == pytest.approx(
        row["base_heading_command_rad"])
    assert targets["blue_0"][1] == pytest.approx(row["executed_heading_command_rad"])


def test_mws_changes_executed_not_base_heading_discontinuity():
    env, sim, obs, action = _diagnostic_env_and_action()
    env._parse_actions({"blue_0": action})
    sim.incoming = _FakeIncomingMissile()
    obs["missile_warning"][:] = 1.0
    action = env.blue_policy_controller.act(
        {"blue_0": obs}, 1, 1, set(), {"blue_0": np.zeros(3)},
        {"blue_0": 0.0}, 1, selected_missiles={"blue_0": sim.incoming.uid},
        mws_detected={"blue_0": True})["blue_0"]
    targets = env._parse_actions({"blue_0": action})
    diag = env.blue_policy_controller.snapshot_episode_diagnostics()
    row = diag["per_blue"][0]
    assert diag["blue_base_heading_command_discontinuities"] == 0
    assert diag["blue_executed_heading_command_discontinuities"] == 1
    assert diag["blue_mws_detected_agent_decisions"] == 1
    assert diag["blue_mws_override_agent_decisions"] == 1
    assert row["executed_command_source"] == "mws_override"
    assert row["executed_heading_command_rad"] == pytest.approx(np.deg2rad(60.0))
    assert targets["blue_0"][1] == pytest.approx(row["executed_heading_command_rad"])


def test_gcas_executed_command_source_is_recorded():
    env, sim, _obs_one, action = _diagnostic_env_and_action()
    env.enable_gcas_for_blue = True
    sim.altitude = 2000.0
    env._parse_actions({"blue_0": action})
    row = env.blue_policy_controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["executed_command_source"] == "gcas_override"
    assert row["executed_heading_command_rad"] == pytest.approx(0.0)


def test_matrix_formal_fields_use_agent_decisions_and_two_heading_layers():
    assert "blue_mws_detected_agent_decisions" in MATRIX_FIELDS
    assert "blue_mws_override_agent_decisions" in MATRIX_FIELDS
    assert "blue_base_heading_command_discontinuities" in MATRIX_FIELDS
    assert "blue_executed_heading_command_discontinuities" in MATRIX_FIELDS
    assert not any("mws" in field and "frames" in field for field in MATRIX_FIELDS)
    assert "blue_mws_detected_agent_decisions" in EVALUATION_FIELDNAMES
    assert "blue_mws_override_agent_decisions" in EVALUATION_FIELDNAMES
    assert "blue_base_heading_command_discontinuities" in EVALUATION_FIELDNAMES
    assert "blue_executed_heading_command_discontinuities" in EVALUATION_FIELDNAMES
    assert not any("mws" in field and "frames" in field
                   for field in EVALUATION_FIELDNAMES)


def test_dead_fixed_ownship_does_not_reassign_when_target_dies():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    _act(controller, _blue_obs())
    targets_before = dict(controller.current_targets)
    switches_before = controller.snapshot_episode_diagnostics()[
        "blue_target_switches_total"]
    actions = _act(
        controller, _blue_obs(alive=(0, 1, 1)),
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    diag = controller.snapshot_episode_diagnostics()
    row = diag["per_blue"][0]
    np.testing.assert_array_equal(actions["blue_0"], np.zeros(3, dtype=np.float32))
    assert controller.current_targets["blue_0"] == targets_before["blue_0"]
    assert diag["blue_target_switches_total"] == switches_before
    assert diag["blue_target_dead_switches"] == 0
    assert row["alive"] is False
    assert row["assignment_reason"] == "dead_ownship_ignored"
    assert row["assigned_target_id"] == "red_0"
    assert row["executed_command_source"] == "not_executed_dead"


def test_dead_fixed_ownship_ignores_nearest_target_and_residual_mws():
    controller = BluePolicyController("fixed_pair_no_mws_v1")
    _reset(controller)
    obs = _blue_obs(enemy_ranges=(15000.0, 100.0, 200.0), warning=1.0)
    actions = _act(
        controller, obs,
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    diag = controller.snapshot_episode_diagnostics()
    row = diag["per_blue"][0]
    np.testing.assert_array_equal(actions["blue_0"], np.zeros(3, dtype=np.float32))
    assert controller.current_targets["blue_0"] == "red_0"
    assert diag["blue_distance_triggered_switches"] == 0
    assert diag["blue_mws_detected_agent_decisions"] == 2
    assert diag["blue_mws_override_agent_decisions"] == 0
    assert row["mws_detected"] is False
    assert row["mws_action_override"] is False


def test_dead_frozen_ownship_does_not_advance_route_or_read_recovery():
    controller = BluePolicyController("frozen_route_blue_v1")
    _reset(controller)
    _act(controller, _blue_obs(), step=75)
    before = controller.snapshot_episode_diagnostics()
    actions = _act(
        controller, _blue_obs(altitude=9000.0, v_up=50.0, warning=1.0),
        step=176,
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    after = controller.snapshot_episode_diagnostics()
    np.testing.assert_array_equal(actions["blue_0"], np.zeros(3, dtype=np.float32))
    assert controller.route_phases["blue_0"] == "A"
    assert after["blue_route_phase_changes"] - before["blue_route_phase_changes"] == 2
    assert (after["blue_base_heading_command_discontinuities"]
            - before["blue_base_heading_command_discontinuities"]) == 2
    assert after["blue_altitude_recovery_frames"] - before[
        "blue_altitude_recovery_frames"] == 2
    assert after["per_blue"][0]["command_heading_delta_rad"] is None


def test_dead_or_unmatched_executed_record_cannot_pollute_counters():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    _act(
        controller, _blue_obs(warning=1.0),
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    before_history = dict(controller.last_executed_headings)
    before_count = controller.mws_override_agent_decisions
    controller.record_executed_heading("blue_0", 1.0, "mws_override")
    controller.record_executed_heading("blue_missing", 1.0, "mws_override")
    assert controller.last_executed_headings == before_history
    assert controller.mws_override_agent_decisions == before_count
    assert controller.executed_heading_command_discontinuities == 0


def test_one_dead_two_alive_only_alive_agents_contribute_diagnostics():
    controller = BluePolicyController("fixed_pair_pursuit_v1")
    _reset(controller)
    actions = _act(
        controller, _blue_obs(warning=1.0),
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    controller.record_executed_heading("blue_0", 1.0, "mws_override")
    controller.record_executed_heading("blue_1", 1.0, "mws_override")
    controller.record_executed_heading("blue_2", -1.0, "mws_override")
    diag = controller.snapshot_episode_diagnostics()
    np.testing.assert_array_equal(actions["blue_0"], np.zeros(3, dtype=np.float32))
    assert set(controller.last_base_commands) == {"blue_1", "blue_2"}
    assert set(controller.last_executed_headings) == {"blue_1", "blue_2"}
    assert diag["blue_mws_detected_agent_decisions"] == 2
    assert diag["blue_mws_override_agent_decisions"] == 2


def test_paper_pursuit_dead_blue_filters_only_diagnostics_not_alive_actions():
    obs = _blue_obs(enemy_ranges=(9000.0, 11000.0, 13000.0))
    positions = {f"blue_{i}": np.zeros(3) for i in range(3)}
    headings = {"blue_0": 0.0, "blue_1": 0.2, "blue_2": -0.2}
    expected = blue_coordinated_actions(
        obs, 3, 3, engaged_targets={"red_1"}, own_positions=positions,
        own_headings=headings, pursuit_mode="paper_pursuit")
    controller = BluePolicyController("paper_pursuit")
    _reset(controller)
    actual = controller.act(
        obs, 3, 3, {"red_1"}, positions, headings, 0,
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    np.testing.assert_array_equal(actual["blue_1"], expected["blue_1"])
    np.testing.assert_array_equal(actual["blue_2"], expected["blue_2"])
    row = controller.snapshot_episode_diagnostics()["per_blue"][0]
    assert row["alive"] is False
    assert row["assignment_reason"] == "dead_ownship_ignored"
    assert "blue_0" not in controller.last_base_commands


def test_dead_state_is_isolated_across_eight_controllers_and_reset_clears_rows():
    controllers = [BluePolicyController("fixed_pair_pursuit_v1") for _ in range(8)]
    for controller in controllers:
        _reset(controller)
    _act(
        controllers[0], _blue_obs(warning=1.0),
        own_alive={"blue_0": False, "blue_1": True, "blue_2": True})
    assert controllers[0].snapshot_episode_diagnostics()["per_blue"][0][
        "alive"] is False
    assert all(controller.latest_per_blue == [] for controller in controllers[1:])
    _reset(controllers[0])
    assert controllers[0].latest_per_blue == []
    assert controllers[0].last_base_commands == {}
    assert controllers[0].last_executed_headings == {}
    assert controllers[0].mws_detected_agent_decisions == 0
    assert controllers[0].mws_override_agent_decisions == 0


def test_environment_passes_authoritative_simulator_own_alive():
    captured = {}

    class CaptureController:
        def act(self, *args, **kwargs):
            captured.update(kwargs["own_alive"])
            return {f"blue_{index}": np.zeros(3, dtype=np.float32)
                    for index in range(3)}

    env = UavCombatEnv.__new__(UavCombatEnv)
    sims = {f"blue_{index}": _FakeBlueSim() for index in range(3)}
    sims["blue_1"].is_alive = False
    env.blue_planes = sims
    env.blue_policy_profile = "fixed_pair_pursuit_v1"
    env.blue_policy_controller = CaptureController()
    env.max_num_blue = 3
    env.max_num_red = 3
    env.current_step = 0
    env.refresh_engaged_targets = lambda: set()
    env.get_blue_own_kinematics = lambda: {
        blue_id: {"position": np.zeros(3), "heading": 0.0}
        for blue_id, sim in sims.items() if sim.is_alive}
    env.blue_policy_actions(_blue_obs())
    assert captured == {"blue_0": True, "blue_1": False, "blue_2": True}


def test_environment_fingerprint_includes_blue_policy_profile():
    kwargs = dict(num_red=3, num_blue=3, sim_freq=60,
                  agent_interaction_steps=12, seed=1)
    paper = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, **kwargs,
        blue_policy_profile="paper_pursuit")
    fixed = environment_config_snapshot(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG, **kwargs,
        blue_policy_profile="fixed_pair_pursuit_v1")
    assert paper["environment_config_fingerprint"] != fixed[
        "environment_config_fingerprint"]


def test_legacy_checkpoint_profile_compatibility_is_paper_only(capsys):
    payload = {
        "state_dict": {"weight": np.asarray([1.0])},
        "metadata": {"schema_version": "test"},
        "model_kind": "actor",
    }
    state = _unpack_and_validate_checkpoint(
        payload, {"schema_version": "test",
                  "blue_policy_profile": "paper_pursuit"}, "actor")
    assert state is payload["state_dict"]
    assert "interpreting it as paper_pursuit" in capsys.readouterr().out
    with pytest.raises(ValueError, match="blue_policy_profile"):
        _unpack_and_validate_checkpoint(
            payload, {"schema_version": "test",
                      "blue_policy_profile": "fixed_pair_pursuit_v1"}, "actor")
