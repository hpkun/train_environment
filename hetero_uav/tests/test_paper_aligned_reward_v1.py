"""Tests for the TAM/BRMA paper-aligned reward v1 candidate."""

from __future__ import annotations

import math
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"
)


class FakeSim:
    def __init__(
        self,
        *,
        pos=(0.0, 0.0, 6000.0),
        rpy=(0.0, 0.0, 0.0),
        alive=True,
        warning=False,
    ):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.is_alive = bool(alive)
        self._warning = warning

    def get_position(self):
        return self._pos

    def get_rpy(self):
        return self._rpy

    def check_missile_warning(self):
        return object() if self._warning else None


def _bare_env() -> HeteroUavCombatEnv:
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = ["blue_0", "blue_1"]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {
        "red_0": "mav",
        "red_1": "attack_uav",
        "red_2": "attack_uav",
        "blue_0": "attack_uav",
        "blue_1": "attack_uav",
    }
    env.red_planes = {
        "red_0": FakeSim(pos=(0.0, 0.0, 6500.0), rpy=(0.0, 0.0, 0.0)),
        "red_1": FakeSim(pos=(8000.0, 0.0, 6000.0), rpy=(0.0, 0.0, 0.0)),
        "red_2": FakeSim(pos=(8000.0, 4000.0, 6000.0), rpy=(0.0, 0.0, 0.0)),
    }
    env.blue_planes = {
        "blue_0": FakeSim(pos=(16000.0, 0.0, 6000.0), rpy=(0.0, 0.0, math.pi)),
        "blue_1": FakeSim(pos=(16000.0, 4000.0, 6000.0), rpy=(0.0, 0.0, math.pi)),
    }
    env._last_step_obs = {
        "red_0": {
            "enemy_observed_mask": np.array([1.0, 1.0], dtype=np.float32),
            "enemy_track_source": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        },
        "red_1": {
            "enemy_track_source": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        },
        "red_2": {
            "enemy_track_source": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        },
    }
    env._launch_quality_step_records = [
        {"shooter_id": "red_1", "launch_track_source": "mav_shared"},
        {"shooter_id": "red_2", "launch_track_source": "direct"},
    ]
    env._launch_quality_done_step_records = [
        {
            "shooter_id": "red_1",
            "launch_track_source": "mav_shared",
            "raw_termination_reason": "hit",
        }
    ]
    env._step_kill_count = {"red_1": 2, "red_2": 1}
    env.tam_brma_paper_aligned_v1_config = {
        "mav_reward_scale": 0.05,
        "mav_safety": {
            "dist_weight": 0.5,
            "threat_weight": 0.3,
            "aspect_weight": 0.2,
            "d_danger_m": 8000.0,
            "d_safe_m": 15000.0,
        },
        "mav_support": {
            "pos_weight": 0.6,
            "aware_weight": 0.4,
            "d_opt_m": 8000.0,
            "d_max_m": 25000.0,
        },
        "mav_event": {
            "death_penalty_raw": 200.0,
            "team_credit_per_kill_raw": 100.0,
            "team_credit_cap_raw": 200.0,
        },
        "uav": {"include_r_death": False},
    }
    env._tam_brma_paper_v1_reset_episode_state()
    return env


def _base_rewards_and_components():
    rewards = {"red_0": 10.0, "red_1": 11.0, "red_2": 12.0}
    components = {
        "red_0": {
            "r_pitch": 1.0,
            "r_roll": 2.0,
            "r_alt": 3.0,
            "r_bound": 4.0,
            "r_vel": 5.0,
            "r_adv": 6.0,
            "r_end": 7.0,
            "r_death": -99.0,
            "total": 10.0,
        },
        "red_1": {
            "r_pitch": 1.0,
            "r_roll": 2.0,
            "r_alt": 3.0,
            "r_bound": 4.0,
            "r_vel": 5.0,
            "r_adv": 6.0,
            "r_end": 7.0,
            "r_death": -99.0,
            "total": 11.0,
        },
        "red_2": {
            "r_pitch": 2.0,
            "r_roll": 3.0,
            "r_alt": 4.0,
            "r_bound": 5.0,
            "r_vel": 6.0,
            "r_adv": 7.0,
            "r_end": 8.0,
            "total": 12.0,
        },
    }
    return rewards, components


def test_reward_mode_registered_and_configs_load():
    from uav_env import make_env

    for cfg, max_red, max_blue in [(CFG_3V2, 3, 2), (CFG_5V4, 5, 4)]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "tam_brma_paper_aligned_v1"
            assert env.observation_mode == "mav_shared_geo"
            assert env.agent_roles["red_0"] == "mav"
            assert env.aircraft_type_params["mav"]["aircraft_model"] == "f16"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.aircraft_type_params["attack_uav"]["num_missiles"] == 2
            assert env.red_target_selection_mode == "closest"
            assert env.max_num_red == max_red
            assert env.max_num_blue == max_blue
        finally:
            env.close()


def test_configs_do_not_override_missile_or_geometry_contracts():
    forbidden = {
        "MISSILE_LAUNCH_RANGE_THRESH",
        "MISSILE_LAUNCH_AO_THRESH",
        "MISSILE_LAUNCH_TA_THRESH",
        "MISSILE_LOCK_DELAY_SEC",
        "MISSILE_LAUNCH_COOLDOWN_SEC",
        "red_initial_state",
        "blue_initial_state",
    }
    for cfg in [CFG_3V2, CFG_5V4]:
        data = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        assert data["hetero_reward_mode"] == "tam_brma_paper_aligned_v1"
        assert data["red_target_selection_mode"] == "closest"
        assert not (forbidden & set(data))
        assert "tam_paper_reward_v7_role_aligned" not in data
        assert "happo_ref_v1_mav_support" not in data


def test_attack_uav_keeps_only_brma_trunk_terms_by_default():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()

    rewards, components = env._compute_tam_brma_paper_aligned_v1(rewards, components)
    comp = components["red_1"]

    assert rewards["red_1"] == pytest.approx(28.0)
    assert comp["paper_v1_uav_flight"] == pytest.approx(15.0)
    assert comp["paper_v1_uav_adv"] == pytest.approx(6.0)
    assert comp["paper_v1_uav_end"] == pytest.approx(7.0)
    assert comp["paper_v1_uav_r_death_log"] == pytest.approx(-99.0)
    assert comp["paper_v1_uav_total"] == pytest.approx(28.0)
    for key in comp:
        assert "uav_fire" not in key
        assert "uav_hit" not in key
        assert "uav_dodge" not in key
        assert "launch_reward" not in key
        assert "shared_track_hit_reward" not in key


def test_mav_removes_attack_adv_and_terminal_then_adds_scaled_role_terms():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()

    rewards, components = env._compute_tam_brma_paper_aligned_v1(rewards, components)
    comp = components["red_0"]

    assert comp["paper_v1_mav_removed_r_adv"] == pytest.approx(6.0)
    assert comp["paper_v1_mav_removed_r_end"] == pytest.approx(7.0)
    assert comp["paper_v1_mav_r_death_log"] == pytest.approx(-99.0)
    assert comp["r_adv"] == pytest.approx(0.0)
    assert comp["r_end"] == pytest.approx(0.0)
    assert comp["paper_v1_mav_flight"] == pytest.approx(15.0)
    expected = comp["paper_v1_mav_flight"] + 0.05 * (
        comp["paper_v1_mav_safety"]
        + comp["paper_v1_mav_support"]
        + comp["paper_v1_mav_event_raw"]
    )
    assert rewards["red_0"] == pytest.approx(expected)
    assert comp["paper_v1_mav_total"] == pytest.approx(expected)
    assert "tam_v7_mav_safety" not in comp
    assert "v1_mav_safety" not in comp


def test_mav_event_death_once_team_credit_cap_and_no_credit_after_death():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()

    _, components = env._compute_tam_brma_paper_aligned_v1(rewards, components)
    assert components["red_0"]["paper_v1_mav_event_death_raw"] == pytest.approx(0.0)
    assert components["red_0"]["paper_v1_mav_event_team_credit_delta_raw"] == pytest.approx(200.0)
    assert components["red_0"]["paper_v1_mav_event_team_credit_used_raw"] == pytest.approx(200.0)

    env.red_planes["red_0"].is_alive = False
    rewards, components = _base_rewards_and_components()
    _, components = env._compute_tam_brma_paper_aligned_v1(rewards, components)
    assert components["red_0"]["paper_v1_mav_event_death_raw"] == pytest.approx(-200.0)
    assert components["red_0"]["paper_v1_mav_event_team_credit_delta_raw"] == pytest.approx(0.0)

    rewards, components = _base_rewards_and_components()
    _, components = env._compute_tam_brma_paper_aligned_v1(rewards, components)
    assert components["red_0"]["paper_v1_mav_event_death_raw"] == pytest.approx(0.0)


def test_mav_position_support_geometry_center_excludes_mav():
    env = _bare_env()
    cfg = env.tam_brma_paper_aligned_v1_config
    mav = env.red_planes["red_0"]

    mav._pos = np.array([12000.0, 2000.0, 6500.0])
    pos, logs = env._paper_v1_mav_position_support(mav, cfg)
    assert logs["paper_v1_mav_battlefield_center_x"] == pytest.approx(12000.0)
    assert logs["paper_v1_mav_battlefield_center_y"] == pytest.approx(2000.0)
    assert logs["paper_v1_mav_pos_distance_m"] == pytest.approx(0.0)
    assert pos == pytest.approx(-1.0)

    mav._pos = np.array([20000.0, 2000.0, 6500.0])
    pos, logs = env._paper_v1_mav_position_support(mav, cfg)
    assert logs["paper_v1_mav_pos_distance_m"] == pytest.approx(8000.0)
    assert pos == pytest.approx(1.0)

    mav._pos = np.array([50000.0, 2000.0, 6500.0])
    pos, _ = env._paper_v1_mav_position_support(mav, cfg)
    assert pos == pytest.approx(-0.5)


def test_mav_awareness_and_aspect_use_geometry_not_launch_gate():
    env = _bare_env()
    cfg = env.tam_brma_paper_aligned_v1_config
    mav = env.red_planes["red_0"]

    env.blue_planes["blue_0"]._pos = np.array([10000.0, 0.0, 6500.0])
    mav._rpy = np.array([0.0, 0.0, 0.0])
    aware, _logs = env._paper_v1_mav_awareness(mav, cfg)
    assert aware > 0.0

    mav._rpy = np.array([0.0, 0.0, math.pi])
    aware_back, _logs = env._paper_v1_mav_awareness(mav, cfg)
    assert aware_back == pytest.approx(0.0)

    blue = env.blue_planes["blue_0"]
    blue._rpy = np.array([0.0, 0.0, math.pi])
    aspect = env._paper_v1_blue_aspect_threat(blue, mav)
    assert aspect < 0.0

    blue._rpy = np.array([0.0, 0.0, 0.0])
    aspect_away = env._paper_v1_blue_aspect_threat(blue, mav)
    assert aspect_away == pytest.approx(0.0)


def test_logging_schema_contains_paper_v1_fields_and_summary_mode_skips_big_files(tmp_path):
    from scripts.experiment_logging_schema import (
        EPISODE_REWARD_COMPONENTS_COLUMNS,
        REWARD_COMPONENT_COLUMNS,
    )
    from scripts.rich_logging import RichExperimentLogger

    for key in [
        "paper_v1_uav_total",
        "paper_v1_mav_safety",
        "paper_v1_mav_support",
        "paper_v1_mav_event_raw",
        "paper_v1_mav_total",
    ]:
        assert key in REWARD_COMPONENT_COLUMNS
        assert key + "_sum" in EPISODE_REWARD_COMPONENTS_COLUMNS

    logger = RichExperimentLogger(
        tmp_path,
        run_id="paper-v1",
        method_name="method",
        scenario_name="scenario",
        device="cpu",
        num_envs=1,
        rollout_length_per_env=1,
        transitions_per_rollout=1,
        mode="summary",
    )
    logger.write_reward_components(
        {"reward_components": {"red_0": {"paper_v1_mav_total": 1.0}}},
        scenario="test",
        episode_id=0,
        step=1,
        sim_time=0.0,
    )
    logger.write_aircraft_timeseries({}, scenario="test", episode_id=0, step=1, sim_time=0.0)
    logger.close()
    for filename in ("reward_components.csv", "aircraft_timeseries.csv"):
        rows = list(csv.reader((tmp_path / filename).open(newline="", encoding="utf-8")))
        assert len(rows) == 1
    assert (tmp_path / "episode_reward_components.csv").exists()


def test_paper_v1_reset_clears_episode_state():
    """Real env.reset must clear _paper_aligned_v1_mav_death_penalized and team_credit."""
    from uav_env import make_env
    config = (
        "uav_env/JSBSim/configs/"
        "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"
    )
    env = make_env(config, max_steps=10)
    try:
        obs, _info = env.reset(seed=0)
        # Set state as if episode progressed
        env._paper_aligned_v1_mav_death_penalized = True
        env._paper_aligned_v1_mav_team_credit_used = 123.0
        # Re-reset and verify cleared
        obs, _info = env.reset(seed=1)
        assert env._paper_aligned_v1_mav_death_penalized is False, (
            f"death_penalized should be False after reset, got {env._paper_aligned_v1_mav_death_penalized}"
        )
        assert env._paper_aligned_v1_mav_team_credit_used == 0.0, (
            f"team_credit_used should be 0 after reset, got {env._paper_aligned_v1_mav_team_credit_used}"
        )
    finally:
        env.close()
