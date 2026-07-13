from __future__ import annotations

import copy

import numpy as np

from my_uav_env.blue_policy_profiles import (
    BLUE_POLICY_PROFILES,
    BluePolicyController,
)
from rule_based_agent import blue_coordinated_actions
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


def _act(controller, obs, step=0, engaged=None):
    headings = {"blue_0": 0.0, "blue_1": 0.2, "blue_2": -0.2}
    positions = {key: np.zeros(3) for key in headings}
    return controller.act(
        obs, 3, 3, engaged or set(), positions, headings, step)


def test_profiles_and_default_remain_paper_pursuit():
    assert BLUE_POLICY_PROFILES == (
        "paper_pursuit", "fixed_pair_pursuit_v1", "fixed_pair_no_mws_v1",
        "fixed_pair_hold_after_kill_v1", "frozen_route_blue_v1")
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
    _act(controller, _blue_obs(alive=(0, 1, 1)))
    generation = controller.episode_generation
    _reset(controller)
    assert controller.episode_generation == generation + 1
    assert controller.current_targets == {
        "blue_0": "red_0", "blue_1": "red_1", "blue_2": "red_2"}
    assert sum(controller.target_switch_counts.values()) == 0
    assert controller.mws_detected_frames == 0


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
    assert diag["blue_mws_detected_frames"] == 3
    assert diag["blue_mws_override_frames"] == 0
    assert not no_mws.blue_mws_override_enabled
    assert pursuit.blue_mws_override_enabled


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
    assert _compute_obs_dim(3, 3, True, obs_mode="paper_strict") == 66
    assert _compute_global_state_dim(3, "paper_strict") == 30
    assert "blue_mws_override_frames" in BLUE_POLICY_DIAG_CSV_FIELDS
    assert "blue_distance_triggered_switches" in BLUE_POLICY_DIAG_CSV_FIELDS


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
