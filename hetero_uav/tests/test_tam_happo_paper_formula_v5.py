from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_tam_v5_unknown_constants import solve as solve_unknown_constants
from scripts.experiment_logging_schema import EPISODE_REWARD_COMPONENTS_COLUMNS, REWARD_COMPONENT_COLUMNS
from uav_env import make_env
from uav_env.JSBSim.envs.paper_formula_v5 import (
    GLOBAL_REWARD_SCALE,
    V5_COMPONENT_FIELDS,
    brma_angle_situation,
    brma_distance_situation,
    brma_end_reference,
    collect_v5_effective_samples,
    identity_error,
    paper_target_score,
    reset_v5_state,
    tam_angle_reward,
    tam_distance_reward,
    tam_speed_reward,
    _update_missile_events,
    _uav_reward,
)


ROOT = Path(__file__).resolve().parents[1]
CFG3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"
CFG5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"


class Sim:
    def __init__(self, pos, vel, alive=True):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.is_alive = alive

    def get_position(self): return self._pos
    def get_velocity(self): return self._vel


class GeometryEnv:
    @staticmethod
    def _brma_tam_3d_geometry(attacker, target):
        delta = target.get_position() - attacker.get_position()
        distance = np.linalg.norm(delta); los = delta / distance
        av = attacker.get_velocity() / np.linalg.norm(attacker.get_velocity())
        tv = target.get_velocity() / np.linalg.norm(target.get_velocity())
        return {"tam_ata_rad": float(np.arccos(np.clip(np.dot(los, av), -1, 1))),
                "tam_aa_rad": float(np.arccos(np.clip(np.dot(los, tv), -1, 1))),
                "target_distance_m": float(distance)}


def test_speed_formula_segments_and_invalid():
    assert tam_speed_reward(200, 99)["speed_raw"] == pytest.approx(1.0)
    assert tam_speed_reward(200, 100)["speed_raw"] == pytest.approx(1.0)
    assert tam_speed_reward(200, 200)["speed_raw"] == pytest.approx(0.0)
    assert tam_speed_reward(200, 300)["speed_raw"] == pytest.approx(-1.0)
    assert tam_speed_reward(200, 301)["speed_raw"] == pytest.approx(-1.0)
    invalid = tam_speed_reward(0, 200)
    assert invalid["speed_raw"] == -1.0 and invalid["speed_invalid"] == 1.0


def test_distance_formula_boundaries():
    assert tam_distance_reward(5000 - 1)["distance_raw"] == 1.0
    assert tam_distance_reward(5000)["distance_raw"] == 1.0
    assert 0.0 < tam_distance_reward(5000 + 1)["distance_raw"] < 1.0
    assert tam_distance_reward(10000 - 1)["distance_raw"] > 0.0
    assert tam_distance_reward(10000)["distance_raw"] == -1.0
    assert tam_distance_reward(10000 + 1)["distance_raw"] == -1.0


def test_angle_geometry_order():
    tail = tam_angle_reward(0.0, 0.0)
    side_rear = tam_angle_reward(math.pi / 4, math.pi / 4)
    head_on = tam_angle_reward(0.0, math.pi)
    being_tailed = tam_angle_reward(math.pi, 0.0)
    assert tail > side_rear > head_on
    assert head_on == pytest.approx(being_tailed)


def test_published_uav_weights_and_global_scale():
    raw = 10 * 1 + 10 * 1 + 15 * 1 + 10 * 1 + 30 * 1 + 200
    assert GLOBAL_REWARD_SCALE * raw == pytest.approx(1.375)


@pytest.mark.parametrize("degree, expected", [(4, 10), (15, 1), (35, 0)])
def test_brma_angle_boundaries(degree, expected):
    assert brma_angle_situation(math.radians(degree)) == pytest.approx(expected)


def test_brma_angle_boundary_sides():
    eps = 1e-8
    assert brma_angle_situation(math.radians(4) - eps) == 10.0
    assert brma_angle_situation(math.radians(4) + eps) < 3.0
    assert brma_angle_situation(math.radians(15) - eps) > 1.0
    assert brma_angle_situation(math.radians(15) + eps) < 1.0
    assert brma_angle_situation(math.radians(35) + eps) == 0.0


def test_brma_distance_and_end():
    assert brma_distance_situation(15000) == 1.0
    assert brma_distance_situation(15001) < 1.0
    assert brma_end_reference(2, 2) == 0.0
    assert brma_end_reference(3, 1) == 60.0


def test_target_score_is_not_nearest_only():
    env = GeometryEnv()
    own = Sim((0, 0, 6000), (250, 0, 0))
    near_bad = Sim((-3000, 0, 6000), (250, 0, 0))
    far_good = Sim((8000, 0, 5000), (150, 0, 0))
    cfg = {"target_assessment": {"engagement_range_m": 14000,
                                  "relative_altitude_norm_m": 10000,
                                  "relative_speed_norm_mps": 408}}
    assert paper_target_score(env, own, far_good, cfg)["score"] > paper_target_score(env, own, near_bad, cfg)["score"]


def test_identity_is_independent_and_detects_tamper():
    parts = [0.1, -0.2, 0.3]
    assert abs(identity_error(sum(parts), parts)) <= 1e-12
    assert abs(identity_error(sum(parts), [0.1, -0.1, 0.3])) > 1e-8


def test_effective_metrics_use_alive_before_role_denominators():
    comps = {
        "red_0": {"alive_before": 1, "safety_raw": 2, "support_raw": 4, "mav_event_raw": 6},
        "red_1": {"alive_before": 1, "height_raw": 1, "speed_raw": 2, "angle_raw": 3,
                  "distance_raw": 4, "dodge_raw": 5, "kill_event_raw": 200,
                  "death_event_raw": 0, "oob_event_raw": 0},
        "red_2": {"alive_before": 0, "height_raw": 999, "speed_raw": 999, "angle_raw": 999,
                  "distance_raw": 999, "dodge_raw": 999},
    }
    sums, counts = collect_v5_effective_samples(comps, {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"})
    assert sums["v5_uav_angle_mean"] / counts["v5_uav_angle_mean"] == 3
    assert sums["v5_mav_support_mean"] / counts["v5_mav_support_mean"] == 4


def test_unknown_constants_are_not_inferred_without_shared_kill_coverage():
    frame = pd.DataFrame([
        {"censored": 0, "terminal_observed": 1, "mav_alive_final": 0,
         "blue_alive_final": 2, "shared_kill_raw": 0, "team_kill_alive_raw": 0},
        {"censored": 0, "terminal_observed": 1, "mav_alive_final": 1,
         "blue_alive_final": 2, "shared_kill_raw": 0, "team_kill_alive_raw": 0},
    ])
    feasible, notes = solve_unknown_constants(frame)
    assert feasible.empty
    assert any("MAV-shared-only kill" in note for note in notes)


def test_schema_is_complete():
    assert set(V5_COMPONENT_FIELDS) <= set(REWARD_COMPONENT_COLUMNS)
    assert {f"{key}_sum" for key in V5_COMPONENT_FIELDS} <= set(EPISODE_REWARD_COMPONENTS_COLUMNS)


@pytest.mark.parametrize("cfg_path, red_count, blue_count", [(CFG3, 3, 2), (CFG5, 5, 4)])
def test_real_config_reset_and_step(cfg_path, red_count, blue_count):
    env = make_env(cfg_path, max_steps=5)
    try:
        obs, _ = env.reset(seed=7)
        assert len(env.red_ids) == red_count and len(env.blue_ids) == blue_count
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _, rewards, _, _, info = env.step(actions)
        assert info["reward_mode"] == "tam_happo_paper_formula_v5"
        for rid in env.red_ids:
            comp = info["reward_components"][rid]
            assert set(V5_COMPONENT_FIELDS) <= set(comp)
            numeric = [value for value in comp.values() if isinstance(value, (int, float, np.number))]
            assert np.isfinite(numeric).all()
            assert abs(comp["identity_error"]) <= 1e-8
            assert rewards[rid] == pytest.approx(comp["total"])
            assert comp["brma_overlay_enabled"] == 0.0
    finally:
        env.close()


def test_reset_clears_v5_event_state():
    env = make_env(CFG3, max_steps=5)
    try:
        env.reset(seed=1)
        env._tam_v5_state["mav_death_seen"] = True
        env._tam_v5_state["mav_event_credit_used"] = 123.0
        env._tam_v5_state["red_launch"].add("m1")
        env.reset(seed=2)
        assert env._tam_v5_state["mav_death_seen"] is False
        assert env._tam_v5_state["mav_event_credit_used"] == 0.0
        assert not env._tam_v5_state["red_launch"]
    finally:
        env.close()


def test_missile_events_are_unique_and_hit_requires_launch():
    class EventEnv:
        _launch_quality_step_records = []
        _launch_quality_done_step_records = []

    env = EventEnv()
    reset_v5_state(env)
    env._launch_quality_step_records = [
        {"missile_id": "m1", "shooter_id": "red_1", "launch_track_source": "mav_shared"},
        {"missile_id": "m1", "shooter_id": "red_1", "launch_track_source": "mav_shared"},
    ]
    env._launch_quality_done_step_records = [
        {"missile_id": "m1", "raw_termination_reason": "hit"},
        {"missile_id": "orphan", "raw_termination_reason": "hit"},
    ]
    _update_missile_events(env, env._tam_v5_state)
    assert env._tam_v5_state["red_launch"] == {"m1"}
    assert env._tam_v5_state["red_hit"] == {"m1"}
    assert len(env._tam_v5_state["red_hit"]) <= len(env._tam_v5_state["red_launch"])

    _update_missile_events(env, env._tam_v5_state)
    assert env._tam_v5_state["red_launch"] == {"m1"}
    assert env._tam_v5_state["red_hit"] == {"m1"}


@pytest.mark.parametrize(
    "death_reason, expected_death, expected_oob",
    [("Out_of_zone", 0.0, -100.0), ("MissileHit", -200.0, 0.0)],
)
def test_uav_death_and_oob_events_are_mutually_exclusive(death_reason, expected_death, expected_oob):
    class EventRewardEnv:
        blue_ids = []
        blue_planes = {}
        _step_kill_count = {}
        _lock_target = {}
        _launch_quality_step_records = []

        @staticmethod
        def _tam_table1_uav_height_raw(_sim, _cfg):
            return 0.0, {"tam_table1_uav_height_pv": 0.0, "tam_table1_uav_height_ph": 0.0}

        @staticmethod
        def _brma_tam_death_reason(_aid):
            return death_reason

        @staticmethod
        def _brma_tam_horizontal_oob(_sim):
            return True

    env = EventRewardEnv()
    reset_v5_state(env)
    sim = Sim((0, 0, 6000), (250, 0, 0), alive=False)
    total, values = _uav_reward(env, "red_1", sim, env._tam_v5_state, {}, alive_before=True)
    assert values["death_event_raw"] == expected_death
    assert values["oob_event_raw"] == expected_oob
    assert not (values["death_event_raw"] and values["oob_event_raw"])
    expected_speed = 10.0  # No alive target gives the published low-target-speed branch R_V=1.
    assert total == pytest.approx(GLOBAL_REWARD_SCALE * (expected_speed + expected_death + expected_oob))

    total_again, values_again = _uav_reward(env, "red_1", sim, env._tam_v5_state, {}, alive_before=False)
    assert total_again == 0.0
    assert all(float(value) == 0.0 for value in values_again.values() if isinstance(value, (int, float)))
