"""Tests for tam_paper_reward_v7_role_aligned."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_logging_schema import MISSILE_EVENTS_COLUMNS, REWARD_COMPONENT_COLUMNS
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


CFG_PATH = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
    "tam_paper_reward_v7_role_aligned.yaml"
)


class FakeSim:
    def __init__(
        self,
        uid: str,
        pos=(0.0, 0.0, 6000.0),
        vel=(250.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        alive=True,
        missiles=2,
    ):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.is_alive = bool(alive)
        self.num_left_missiles = int(missiles)
        self.under_missiles = []

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return self._rpy

    def get_geodetic(self):
        return np.asarray([0.0, 0.0, self._pos[2]], dtype=np.float64)


def _cfg():
    return {
        "global_scale": 1.0,
        "flight": {
            "pitch_weight": 0.01,
            "roll_weight": 0.002,
            "altitude_weight": 0.04,
            "speed_weight": 0.02,
            "boundary_weight": 0.04,
            "boundary_raw_penalty": -10.0,
        },
        "situation": {
            "weight": 0.15,
            "enemy_threat_weight": 0.8,
            "use_effective_launch_range": True,
        },
        "uav_event": {
            "kill_enemy": 200.0,
            "death": -200.0,
            "first_out_of_zone": -100.0,
        },
        "mav_safety": {
            "dist_weight": 0.5,
            "threat_weight": 0.3,
            "aspect_weight": 0.2,
            "negative_scale": 0.20,
            "positive_scale": 0.05,
        },
        "mav_support": {
            "pos_weight": 0.6,
            "aware_weight": 0.4,
            "scale": 0.10,
            "aware_reduce": "mean",
        },
        "mav_event": {
            "death": -300.0,
            "team_credit_per_uav_kill": 40.0,
            "team_credit_cap": 100.0,
            "require_alive_after_for_team_credit": True,
        },
        "terminal": {
            "coef_per_agent": 30.0,
            "red_loss_mode": "role_weighted",
            "mav_weight_mode": "match_uav_count",
        },
    }


def _bare_env(num_red: int = 3, num_blue: int = 2):
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = [f"red_{i}" for i in range(num_red)]
    env.blue_ids = [f"blue_{i}" for i in range(num_blue)]
    env.agent_roles = {rid: ("mav" if rid == "red_0" else "attack_uav") for rid in env.red_ids}
    env.red_planes = {
        "red_0": FakeSim("red_0", pos=(0.0, -8000.0, 6500.0), vel=(230.0, 0.0, 0.0), missiles=0),
    }
    for i in range(1, num_red):
        env.red_planes[f"red_{i}"] = FakeSim(f"red_{i}", pos=(0.0, i * 2000.0, 6000.0))
    env.blue_planes = {
        f"blue_{i}": FakeSim(
            f"blue_{i}",
            pos=(6000.0, i * 2000.0, 6000.0),
            vel=(250.0, 0.0, 0.0),
            rpy=(0.0, 0.0, np.pi),
            missiles=2,
        )
        for i in range(num_blue)
    }
    env.agent_ids = env.red_ids + env.blue_ids
    env.max_num_red = num_red
    env.max_num_blue = num_blue
    env.max_steps = 1000
    env.current_step = 1
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MIN = 2500.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.VELOCITY_MIN = 102.0
    env.VELOCITY_MAX = 408.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_TA_THRESH = np.pi / 2
    env._missile_launch_range_m_effective = 14000.0
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env.tam_paper_reward_v7_role_aligned_config = _cfg()
    env.mav_observation_range_m = 80000.0
    env._last_step_obs = {}
    env._launch_quality_step_records = []
    env._launch_quality_done_step_records = []
    env._tam_v7_reset_episode_state()
    return env


def test_v7_mode_registered_and_config_loads():
    from uav_env import make_env

    env = make_env(CFG_PATH, max_steps=5)
    try:
        assert env.hetero_reward_mode == "tam_paper_reward_v7_role_aligned"
        assert env.tam_paper_reward_v7_role_aligned_config["situation"]["weight"] == 0.15
        assert env._needs_last_step_obs_cache()
    finally:
        env.close()


def test_missing_v7_config_raises():
    with pytest.raises(ValueError, match="tam_paper_reward_v7_role_aligned"):
        HeteroUavCombatEnv(
            hetero_reward_mode="tam_paper_reward_v7_role_aligned",
            max_num_red=3,
            max_num_blue=2,
            max_steps=10,
        )


def test_v7_distance_uses_effective_launch_range_not_hardcoded():
    env = _bare_env()
    assert env._tam_v7_distance_advantage(14000.0, env.tam_paper_reward_v7_role_aligned_config) == pytest.approx(1.0)
    env._missile_launch_range_m_effective = 20000.0
    assert env._tam_v7_distance_advantage(20000.0, env.tam_paper_reward_v7_role_aligned_config) == pytest.approx(1.0)


def test_v7_speed_raw_uses_velocity_envelope_strong_penalty():
    env = _bare_env()
    assert env._tam_v7_speed_raw(42.0) == pytest.approx(-1.0)
    assert env._tam_v7_speed_raw(59.0) == pytest.approx(-1.0)
    assert env._tam_v7_speed_raw(101.0) == pytest.approx(-1.0)
    assert env._tam_v7_speed_raw(250.0) == pytest.approx(0.0)
    assert env._tam_v7_speed_raw(409.0) == pytest.approx(-1.0)


def test_v7_config_exposes_mav_distance_params():
    cfg = yaml.safe_load((ROOT / CFG_PATH).read_text(encoding="utf-8"))
    reward_cfg = cfg["tam_paper_reward_v7_role_aligned"]
    assert reward_cfg["mav_safety"]["d_danger_m"] == pytest.approx(5000.0)
    assert reward_cfg["mav_safety"]["d_safe_m"] == pytest.approx(14000.0)
    assert reward_cfg["mav_support"]["d_opt_m"] == pytest.approx(8000.0)
    assert reward_cfg["mav_support"]["d_max_m"] == pytest.approx(25000.0)


def test_uav_first_out_of_zone_once_and_mav_no_out_of_zone_event():
    env = _bare_env()
    cfg = env.tam_paper_reward_v7_role_aligned_config
    env.red_planes["red_1"]._pos = np.asarray([41000.0, 0.0, 6000.0])
    _, comp1 = env._tam_v7_uav_reward("red_1", env.red_planes["red_1"], env._tam_v2_alive_blue(), cfg)
    _, comp2 = env._tam_v7_uav_reward("red_1", env.red_planes["red_1"], env._tam_v2_alive_blue(), cfg)
    assert comp1["tam_v7_uav_first_out_of_zone"] == pytest.approx(-100.0)
    assert comp2["tam_v7_uav_first_out_of_zone"] == pytest.approx(0.0)

    env.red_planes["red_0"]._pos = np.asarray([41000.0, -8000.0, 6500.0])
    _, mav_comp = env._tam_v7_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg)
    assert "tam_v7_mav_first_out_of_zone" not in mav_comp
    assert mav_comp["tam_v7_mav_boundary"] < 0.0
    assert "tam_v7_mav_altitude" in mav_comp


def test_mav_death_once_and_team_credit_requires_alive_after():
    env = _bare_env()
    cfg = env.tam_paper_reward_v7_role_aligned_config
    env.red_planes["red_0"].is_alive = False
    _, dead_comp = env._tam_v7_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg)
    env._step_kill_count["red_1"] = 1
    _, second_comp = env._tam_v7_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg)
    assert dead_comp["tam_v7_mav_death"] == pytest.approx(-300.0)
    assert second_comp["tam_v7_mav_death"] == pytest.approx(0.0)
    assert second_comp["tam_v7_mav_team_credit_delta"] == pytest.approx(0.0)


def test_mav_team_credit_cap_and_same_step_death_no_credit():
    env = _bare_env()
    cfg = env.tam_paper_reward_v7_role_aligned_config
    env._step_kill_count["red_1"] = 3
    _, comp = env._tam_v7_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg)
    assert comp["tam_v7_mav_team_credit_delta"] == pytest.approx(100.0)
    assert comp["tam_v7_mav_team_credit_used"] == pytest.approx(100.0)

    env = _bare_env()
    env.red_planes["red_0"].is_alive = False
    env._step_kill_count["red_1"] = 1
    _, comp = env._tam_v7_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg)
    assert comp["tam_v7_mav_death"] == pytest.approx(-300.0)
    assert comp["tam_v7_mav_team_credit_delta"] == pytest.approx(0.0)


def test_terminal_role_weighted_not_divided_by_red_count():
    env3 = _bare_env(num_red=3, num_blue=2)
    cfg3 = env3.tam_paper_reward_v7_role_aligned_config
    env3.red_planes["red_0"].is_alive = False
    assert env3._tam_v7_terminal_outcome(cfg3)[0] == pytest.approx(-15.0)

    env5 = _bare_env(num_red=5, num_blue=4)
    cfg5 = env5.tam_paper_reward_v7_role_aligned_config
    env5.red_planes["red_0"].is_alive = False
    terminal, logs = env5._tam_v7_terminal_outcome(cfg5)
    assert terminal == pytest.approx(-15.0)
    assert logs["tam_v7_red_loss_weighted"] == pytest.approx(0.5)


def test_log_only_fields_do_not_enter_total():
    env = _bare_env()
    env._last_step_obs = {
        "red_1": {"enemy_track_source": [[0, 1], [1, 0]]},
        "red_2": {"enemy_track_source": [[0, 1], [0, 1]]},
    }
    env._launch_quality_step_records = [
        {"shooter_id": "red_1", "launch_track_source": "mav_shared"},
    ]
    env._launch_quality_done_step_records = [
        {"shooter_id": "red_1", "launch_track_source": "mav_shared", "raw_termination_reason": "hit"},
    ]
    rewards, comps = env._compute_tam_paper_reward_v7_role_aligned(
        {rid: 0.0 for rid in env.red_ids},
        {rid: {} for rid in env.red_ids},
    )
    c0 = comps["red_0"]
    expected = (
        c0["tam_v7_mav_flight"]
        + c0["tam_v7_mav_safety"]
        + c0["tam_v7_mav_support"]
        + c0["tam_v7_mav_event"]
        + c0["tam_v7_mav_terminal"]
    )
    assert c0["tam_v7_mav_total"] == pytest.approx(expected)
    assert c0["tam_v7_shared_track_usage_log"] > 0.0
    assert c0["tam_v7_red_fire_with_mav_track_log"] > 0.0
    assert c0["tam_v7_red_hit_with_mav_track_log"] > 0.0
    assert rewards["red_0"] == pytest.approx(expected)


def test_v7_total_and_component_keys_dispatch():
    env = _bare_env()
    rewards, comps = env._compute_tam_paper_reward_v7_role_aligned(
        {rid: 0.0 for rid in env.red_ids},
        {rid: {} for rid in env.red_ids},
    )
    for rid in env.red_ids:
        comp = comps[rid]
        assert comp["tam_v7_total"] == pytest.approx(rewards[rid])
        assert "tam_v7_flight" in comp
        assert "tam_v7_event" in comp
        assert "tam_v7_terminal" in comp
    assert "tam_v7_uav_situation" in comps["red_1"]
    assert "tam_v7_mav_support" in comps["red_0"]


def test_v6v3_and_v4_old_modes_unaffected():
    from uav_env import make_env

    v6 = make_env(
        "uav_env/JSBSim/configs/"
        "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
        "tam_paper_reward_v6_jsbsim_aligned_v3.yaml",
        max_steps=5,
    )
    v4 = make_env(
        "uav_env/JSBSim/configs/"
        "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml",
        max_steps=5,
    )
    try:
        assert v6.hetero_reward_mode == "tam_paper_reward_v6_jsbsim_aligned_v3"
        assert not hasattr(v6, "tam_paper_reward_v7_role_aligned_config") or (
            v6.tam_paper_reward_v7_role_aligned_config == {}
        )
        assert v4.hetero_reward_mode == "tam_paper_reward_v4"
    finally:
        v6.close()
        v4.close()


def test_logger_schema_exposes_v7_and_missile_diagnostic_fields():
    single = (ROOT / "scripts" / "train_happo_reference.py").read_text(encoding="utf-8")
    parallel = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    for source in (single, parallel):
        assert "tam_v7_mav_safety_sum" in source
        assert "tam_v7_uav_situation_sum" in source
        assert "tam_v7_uav_altitude_sum" in source
        assert "tam_v7_mav_boundary_sum" in source
        assert "tam_v7_total_sum" in source
    for field in (
        "shooter_id",
        "shooter_role",
        "target_role",
        "team",
        "hit",
        "launch_track_source",
        "range_3d_m",
        "boresight_3d_rad",
        "raw_termination_reason",
    ):
        assert field in MISSILE_EVENTS_COLUMNS


def test_rich_reward_schema_exposes_v7_component_fields():
    for field in (
        "tam_v7_total",
        "tam_v7_uav_altitude",
        "tam_v7_uav_boundary",
        "tam_v7_uav_first_out_of_zone",
        "tam_v7_mav_safety",
        "tam_v7_mav_support",
        "tam_v7_mav_team_credit_delta",
        "tam_v7_terminal_per_agent",
    ):
        assert field in REWARD_COMPONENT_COLUMNS
