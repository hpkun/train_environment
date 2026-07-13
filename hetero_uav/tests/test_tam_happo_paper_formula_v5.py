from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.calibrate_tam_v5_unknown_constants import coverage_statuses, solve as solve_unknown_constants
from scripts.experiment_logging_schema import (
    EPISODE_REWARD_COMPONENTS_COLUMNS, REWARD_COMPONENT_COLUMNS, TRAIN_METRICS_COLUMNS,
)
from scripts.summarize_tam_v5_probes import INSUFFICIENT, readiness
from scripts.rich_logging import RichExperimentLogger
from uav_env import make_env
from uav_env.JSBSim.envs.paper_formula_v5 import (
    GLOBAL_REWARD_SCALE,
    V5_COMPONENT_FIELDS,
    V5_TRAIN_FIELDS,
    accumulate_v5_episode_step,
    brma_angle_situation,
    brma_distance_situation,
    brma_end_reference,
    collect_v5_effective_samples,
    identity_error,
    paper_target_score,
    paper_mav_awareness,
    marginal_shared_awareness,
    reset_v5_state,
    tam_angle_reward,
    tam_distance_reward,
    tam_speed_reward,
    validate_global_reward_scale,
    _update_missile_events,
    _mav_reward,
    _uav_reward,
)


ROOT = Path(__file__).resolve().parents[1]
CFG3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"
CFG5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"
DIAGNOSTIC_CFG = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_paper_formula_v5_shared_diagnostic.yaml"


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


def test_no_target_speed_is_zero_in_production_path():
    class NoTargetEnv:
        blue_ids = []
        blue_planes = {}
        _step_kill_count = {}
        _lock_target = {}
        _launch_quality_step_records = []
        @staticmethod
        def _tam_table1_uav_height_raw(_sim, _cfg):
            return 0.0, {"tam_table1_uav_height_pv": 0.0, "tam_table1_uav_height_ph": 0.0}
        @staticmethod
        def _brma_tam_horizontal_oob(_sim): return False
        @staticmethod
        def _brma_tam_death_reason(_aid): return ""
    env = NoTargetEnv(); reset_v5_state(env)
    total, values = _uav_reward(env, "red_1", Sim((0, 0, 6000), (250, 0, 0)),
                                env._tam_v5_state, {}, alive_before=True)
    assert values["speed_target_valid"] == 0.0
    assert values["speed_raw"] == 0.0
    assert total == 0.0


def test_global_scale_contract_rejects_other_values():
    assert validate_global_reward_scale({"global_reward_scale": 0.005}) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="must equal 1/200"):
        validate_global_reward_scale({"global_reward_scale": 0.01})


def test_environment_rejects_wrong_global_scale(tmp_path):
    config = yaml.safe_load(CFG3.read_text(encoding="utf-8"))
    config["tam_happo_paper_formula_v5"]["global_reward_scale"] = 0.01
    path = tmp_path / "bad_scale.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal 1/200"):
        make_env(path)


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
    assert "NO_SHARED_ONLY_KILL_COVERAGE" in notes
    assert "NO_BLUE_LOSS_COVERAGE" in coverage_statuses(frame)


def test_schema_is_complete():
    assert set(V5_COMPONENT_FIELDS) <= set(REWARD_COMPONENT_COLUMNS)
    assert set(V5_TRAIN_FIELDS) <= set(TRAIN_METRICS_COLUMNS)
    assert "true_final_j_last" in EPISODE_REWARD_COMPONENTS_COLUMNS
    assert "identity_error_max_abs" in EPISODE_REWARD_COMPONENTS_COLUMNS


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
            assert comp["true_final_j"] == 0.0
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
    assert total == pytest.approx(GLOBAL_REWARD_SCALE * (expected_death + expected_oob))

    total_again, values_again = _uav_reward(env, "red_1", sim, env._tam_v5_state, {}, alive_before=False)
    assert total_again == 0.0
    assert all(float(value) == 0.0 for value in values_again.values() if isinstance(value, (int, float)))


def test_paper_awareness_counts_each_blue_once_and_marginal_is_separate():
    class AwarenessEnv(GeometryEnv):
        red_ids = ["red_0", "red_1", "red_2"]
        agent_roles = {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"}
        red_planes = {"red_1": Sim((0, 0, 0), (1, 0, 0)), "red_2": Sim((0, 0, 0), (1, 0, 0))}
        @staticmethod
        def _mav_shared_track_state(observer, _target):
            return {"direct_visible": observer == "red_0", "mav_shared_visible": observer != "red_0"}
    env = AwarenessEnv(); mav = Sim((0, 0, 0), (1, 0, 0))
    blue = [("blue_0", Sim((1000, 0, 0), (1, 0, 0)))]
    paper = paper_mav_awareness(env, "red_0", mav, blue)
    marginal = marginal_shared_awareness(env, mav, blue)
    assert paper["awareness_observed_blue_count"] == 1
    assert paper["awareness_raw"] == pytest.approx(0.3)
    assert marginal["marginal_shared_pair_count"] == 2
    assert marginal["marginal_shared_awareness_raw"] == pytest.approx(0.6)


def test_hit_is_not_kill_without_real_kill_attribution():
    class EventEnv:
        red_ids = ["red_1"]
        agent_roles = {"red_1": "attack_uav"}
        blue_planes = {"blue_0": Sim((0, 0, 0), (0, 0, 0), alive=True)}
        _step_kill_count = {"red_1": 0}
        _launch_quality_step_records = [{"missile_id": "m1", "shooter_id": "red_1",
                                         "target_id": "blue_0", "launch_track_source": "mav_shared"}]
        _launch_quality_done_step_records = [{"missile_id": "m1", "raw_termination_reason": "hit"}]
        @staticmethod
        def _brma_tam_death_reason(_aid): return ""
    env = EventEnv(); reset_v5_state(env); _update_missile_events(env, env._tam_v5_state)
    assert env._tam_v5_state["new_red_hit"] == {"m1"}
    assert env._tam_v5_state["new_red_kill"] == set()


def test_real_kill_attribution_uses_shooter_target_and_transition():
    class EventEnv:
        red_ids = ["red_1"]
        agent_roles = {"red_1": "attack_uav"}
        blue_planes = {"blue_0": Sim((0, 0, 0), (0, 0, 0), alive=False)}
        _step_kill_count = {"red_1": 1}
        _launch_quality_step_records = [{"missile_id": "m1", "shooter_id": "red_1",
                                         "target_id": "blue_0", "launch_track_source": "mav_shared"}]
        _launch_quality_done_step_records = [{"missile_id": "m1", "raw_termination_reason": "hit"}]
        @staticmethod
        def _brma_tam_death_reason(_aid): return "Missile_Kill"
    env = EventEnv(); reset_v5_state(env); _update_missile_events(env, env._tam_v5_state)
    assert env._tam_v5_state["new_red_kill"] == {"m1"}
    assert env._tam_v5_state["unattributed_kill_count"] == 0


def test_mav_death_transition_receives_no_team_credit():
    class MavEnv(GeometryEnv):
        blue_ids = []
        blue_planes = {}
        red_ids = ["red_0", "red_1"]
        agent_roles = {"red_0": "mav", "red_1": "attack_uav"}
        red_planes = {"red_1": Sim((0, 0, 0), (1, 0, 0), alive=True)}
        _step_kill_count = {"red_1": 1}
        @staticmethod
        def _mav_shared_track_state(_observer, _target):
            return {"direct_visible": False, "mav_shared_visible": False}
    env = MavEnv(); reset_v5_state(env)
    mav = Sim((0, 0, 6000), (200, 0, 0), alive=False)
    _total, values = _mav_reward(
        env, "red_0", mav, env._tam_v5_state,
        {"unknown_constants": {"mav_death_penalty": 200, "mav_team_credit_per_kill": 100,
                               "mav_team_credit_cap": 200}}, alive_before=True,
    )
    assert values["death_event_raw"] == -200
    assert values["event_credit_delta"] == 0
    assert values["team_kill_after_mav_death_raw"] == 1


def test_identity_rollout_max_and_episode_last_values_are_not_averaged():
    sums = {key: 0.0 for key in V5_TRAIN_FIELDS}
    for error in [1e-12] + [0.0] * 255:
        step_sums, _counts = collect_v5_effective_samples(
            {"red_1": {"alive_before": 1, "identity_error": error}}, {"red_1": "attack_uav"})
        sums["v5_identity_max_abs"] = max(sums["v5_identity_max_abs"], step_sums["v5_identity_max_abs"])
    assert sums["v5_identity_max_abs"] == pytest.approx(1e-12)
    env0, env1 = {}, {}
    accumulate_v5_episode_step(env0, {"true_final_j": 0.0, "red_alive_final": 3, "identity_error": 1e-12})
    accumulate_v5_episode_step(env0, {"true_final_j": -1.0, "red_alive_final": 0, "identity_error": 0.0})
    accumulate_v5_episode_step(env1, {"true_final_j": 0.5, "red_alive_final": 2, "identity_error": 2e-12})
    assert env0["true_final_j_last"] == -1.0 and env0["red_alive_final_last"] == 0
    assert env0["identity_error_max_abs"] == pytest.approx(1e-12)
    assert env1["true_final_j_last"] == 0.5 and env1["identity_error_max_abs"] == pytest.approx(2e-12)


def test_probe_readiness_rejects_single_seed_launch():
    rows = pd.DataFrame([
        {"candidate": "mid", "seed": seed, "technical_pass": True,
         "red_launch": 1 if seed == 0 else 0, "blue_loss": 0,
         "uav_angle": 0.1, "uav_distance": 0.1, "geometry_rate": 0.0}
        for seed in (0, 1, 2)
    ])
    baseline = pd.DataFrame([
        {"seed": seed, "red_launch": 0, "blue_loss": 0,
         "uav_angle": 0.0, "uav_distance": 0.0, "geometry_rate": 0.0}
        for seed in (0, 1, 2)
    ])
    status, _reasons = readiness(rows, ["mid"], baseline)
    assert status != "TAM_HAPPO_PAPER_FORMULA_V5_READY_FOR_200K_PROBE"


@pytest.mark.parametrize("mode", ["summary", "full"])
def test_v5_train_metrics_are_written_to_real_rich_csv(tmp_path, mode):
    directory = tmp_path / mode
    logger = RichExperimentLogger(
        directory=directory, run_id="v5-test", method_name="pure_happo",
        scenario_name="3v2", device="cpu", num_envs=1,
        rollout_length_per_env=4, transitions_per_rollout=4, mode=mode,
    )
    expected = {field: float(index + 1) / 10.0 for index, field in enumerate(V5_TRAIN_FIELDS)}
    try:
        logger.write_train_metrics({"train_steps": 4, "total_env_steps_actual": 4, **expected})
    finally:
        logger.close()
    frame = pd.read_csv(directory / "train_metrics.csv")
    assert len(frame) == 1
    assert set(V5_TRAIN_FIELDS) <= set(frame.columns)
    assert np.isfinite(frame[list(V5_TRAIN_FIELDS)].to_numpy(dtype=float)).all()
    for field, value in expected.items():
        assert frame.loc[0, field] == pytest.approx(value)


def test_diagnostic_fixture_preserves_formal_reward_and_missile_contract():
    formal = yaml.safe_load(CFG3.read_text(encoding="utf-8"))
    diagnostic = yaml.safe_load(DIAGNOSTIC_CFG.read_text(encoding="utf-8"))
    assert diagnostic["tam_happo_paper_formula_v5"]["diagnostic_fixture_claim"] == (
        "diagnostic-only observability fixture, not training configuration"
    )
    diagnostic_reward = dict(diagnostic["tam_happo_paper_formula_v5"])
    diagnostic_reward.pop("diagnostic_fixture_claim")
    assert diagnostic_reward == formal["tam_happo_paper_formula_v5"]
    for key in ("hetero_reward_mode", "missile_launch_range_m", "missile_attack_interval_sec",
                "missile_guidance", "missile_protocol"):
        assert diagnostic[key] == formal[key]


def test_probe_readiness_accepts_two_of_three_consistent_seeds():
    baseline = pd.DataFrame([
        {"seed": seed, "red_launch": 0, "blue_loss": 0, "uav_angle": 0.0,
         "uav_distance": 0.0, "geometry_rate": 0.0}
        for seed in (0, 1, 2)
    ])
    rows = pd.DataFrame([
        {"candidate": "mid", "seed": seed, "technical_pass": True,
         "red_launch": 1 if seed in (0, 1) else 0, "blue_loss": 0,
         "uav_angle": 0.1 if seed in (0, 1) else 0.0,
         "uav_distance": 0.0, "geometry_rate": 0.1 if seed in (0, 1) else 0.0}
        for seed in (0, 1, 2)
    ])
    status, reasons = readiness(rows, ["mid"], baseline)
    assert status == "TAM_HAPPO_PAPER_FORMULA_V5_READY_FOR_200K_PROBE", reasons
