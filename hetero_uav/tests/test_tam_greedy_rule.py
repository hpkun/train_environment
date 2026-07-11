from __future__ import annotations

import numpy as np

from algorithms.mappo.opponent_policy import OpponentPolicy
from algorithms.mappo.tam_greedy_rule import (
    CANDIDATE_MANEUVERS, PROTOCOL_VERSION, SCORE_WEIGHTS, TamGreedyRule,
)
from scripts.experiment_logging_schema import FINAL_PURE_HAPPO_COLUMNS, SCALE_V2_TRAIN_METRIC_COLUMNS
from scripts.train_happo_reference import _aggregate_scale_v2_step


def _obs(*, bearing=0.25, elevation=0.0, distance=0.5, altitude=0.6, warning=0.0):
    return {
        "ego_geo_state": np.array([0, 0, altitude, 0.45, 0, 0, 0], dtype=np.float32),
        "enemy_relative_pos_xyz": np.array([[distance, 0, 0]], dtype=np.float32),
        "enemy_bearing_elevation": np.array([[bearing, elevation]], dtype=np.float32),
        "enemy_full_geo_valid_mask": np.array([1], dtype=np.float32),
        "enemy_observed_mask": np.array([1], dtype=np.float32),
        "enemy_alive_mask": np.array([1], dtype=np.float32),
        "missile_warning": np.array([warning], dtype=np.float32),
    }


class _OwnshipOnlyEnv:
    VELOCITY_MIN = 102.0
    VELOCITY_MAX = 408.0

    def refresh_engaged_targets(self):
        return set()

    def get_blue_own_kinematics(self):
        return {"blue_0": {"heading": 0.0, "position": np.array([0.0, 0.0, 6000.0])}}

    def get_blue_own_positions(self):
        return {"blue_0": np.array([0.0, 0.0, 6000.0])}

    def __getattr__(self, name):
        if name.startswith("red") or "reward" in name or "action" in name:
            raise AssertionError(f"hidden red data accessed: {name}")
        raise AttributeError(name)


def test_tam_rule_constructs_and_actions_are_legal_without_hidden_red_access():
    assert "tam_greedy_rule" in OpponentPolicy.MODES
    policy = OpponentPolicy("tam_greedy_rule", seed=1)
    action = policy.act({"blue_0": _obs()}, ["blue_0"], env=_OwnshipOnlyEnv())["blue_0"]
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert policy.last_states["blue_0"] in CANDIDATE_MANEUVERS


def test_absolute_heading_mapping_and_no_target_are_stable():
    policy = OpponentPolicy("tam_greedy_rule")
    action = policy.act({"blue_0": _obs(bearing=0.5)}, ["blue_0"], env=_OwnshipOnlyEnv())["blue_0"]
    if policy.last_states["blue_0"] == "pursue_current_target":
        assert action[1] == np.float32(0.5)
    empty = {"ego_geo_state": np.array([0, 0, .6, .4, 0, .2, 0], dtype=np.float32)}
    no_target = policy.act({"blue_0": empty}, ["blue_0"], env=_OwnshipOnlyEnv())["blue_0"]
    assert np.isfinite(no_target).all()
    assert -1.0 <= no_target[1] <= 1.0


def test_target_assignment_deconflicts_and_reset_clears_memory():
    rule = TamGreedyRule()
    obs = _obs()
    obs["enemy_relative_pos_xyz"] = np.array([[.2, 0, 0], [.4, 0, 0]], dtype=np.float32)
    obs["enemy_bearing_elevation"] = np.array([[.1, 0], [-.1, 0]], dtype=np.float32)
    obs["enemy_full_geo_valid_mask"] = obs["enemy_observed_mask"] = obs["enemy_alive_mask"] = np.ones(2, dtype=np.float32)
    assigned = rule.assign_targets({"blue_0": obs, "blue_1": obs}, ["blue_0", "blue_1"], set())
    assert assigned["blue_0"] != assigned["blue_1"]
    rule.reset()
    assert rule.targets == {}
    assert rule.selected_counts == {}


def test_safety_scoring_blocks_descent_and_prefers_warning_break():
    rule = TamGreedyRule()
    low_action, low_name = rule.action("blue_0", _obs(altitude=.2), {}, 0, 102, 408)
    assert low_action[0] >= 0.0
    warning_action, warning_name = rule.action("blue_0", _obs(warning=1.0), {}, 0, 102, 408)
    assert warning_name in {"break_left", "break_right"}
    assert np.isfinite(warning_action).all()


def test_paper_weight_ratio_and_candidate_dependent_components():
    assert PROTOCOL_VERSION == "tam_greedy_rule_v2_paper_weighted"
    ratio = np.array([SCORE_WEIGHTS[k] for k in ("height", "speed", "angle", "distance", "avoidance")])
    assert np.allclose(ratio / ratio[0], [1.0, 1.0, 1.5, 1.0, 3.0])
    rule = TamGreedyRule()
    rule.action("blue_0", _obs(bearing=.3, elevation=.1, distance=.4), {}, 0, 102, 408)
    scores = rule.last_candidate_scores["blue_0"]
    assert len({round(v["height"], 8) for v in scores.values()}) > 1
    assert len({round(v["speed"], 8) for v in scores.values()}) > 1
    assert len({round(v["angle"], 8) for v in scores.values()}) > 1
    assert len({round(v["distance"], 8) for v in scores.values()}) > 1


def test_boundary_pressure_selects_explicit_return_center():
    rule = TamGreedyRule()
    action, name = rule.action(
        "blue_0", _obs(bearing=0.2), {"position": np.array([39000.0, 0.0, 6000.0])},
        0, 102, 408,
    )
    assert name == "return_center"
    assert abs(abs(float(action[1])) - 1.0) < 1e-6


def test_v2_and_final_metrics_are_part_of_formal_schema():
    assert len(SCALE_V2_TRAIN_METRIC_COLUMNS) == 23
    assert len(FINAL_PURE_HAPPO_COLUMNS) == 16
    for name in ("final_ratio_std_mav", "final_ratio_p95_uav", "final_ratio_p99_mav"):
        assert name in FINAL_PURE_HAPPO_COLUMNS


def test_v2_step_aggregation_uses_alive_before_team_weighting():
    components = {
        "red_0": {"scale_v2_flight_raw_total": 10, "scale_v2_flight_scaled_total": 1,
                  "scale_v2_mav_role": 2, "scale_v2_event": 0, "scale_v2_terminal": 3, "total": 6},
        "red_1": {"scale_v2_flight_raw_total": 20, "scale_v2_flight_scaled_total": 2,
                  "scale_v2_progress": 4, "scale_v2_event": 1, "scale_v2_terminal": 3, "total": 10},
        "red_2": {"scale_v2_flight_raw_total": 999, "scale_v2_flight_scaled_total": 999,
                  "scale_v2_progress": 999, "scale_v2_event": 999, "scale_v2_terminal": 999, "total": 3996},
    }
    out = _aggregate_scale_v2_step(
        ["red_0", "red_1", "red_2"],
        {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"},
        np.array([1.0, 1.0, 0.0]), components,
    )
    assert out["scale_v2_active_red_count"] == 2.0
    assert out["effective_scale_v2_mav_flight_scaled"] == 0.5
    assert out["effective_scale_v2_uav_flight_scaled"] == 1.0
    assert out["effective_scale_v2_total"] == 8.0
    assert abs(out["effective_scale_v2_identity_error"]) <= 1e-12
