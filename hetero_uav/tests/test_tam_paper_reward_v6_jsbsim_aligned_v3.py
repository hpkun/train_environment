"""Tests for tam_paper_reward_v6_jsbsim_aligned_v3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


CFG_PATH = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
    "tam_paper_reward_v6_jsbsim_aligned_v3.yaml"
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
        "flight_status": {
            "pitch_weight": 0.01,
            "roll_weight": 0.002,
            "altitude_weight": 0.04,
            "speed_weight": 0.02,
            "boundary_weight": 0.04,
            "boundary_raw_penalty": -10.0,
            "altitude_mode": "penalty_only",
            "speed_mode": "low_speed_penalty_only",
            "use_one_shot_out_of_zone_event": False,
        },
        "situation": {
            "weight": 0.15,
            "enemy_threat_weight": 0.8,
            "normalize_by_alive_blue": True,
            "use_rear_factor": True,
            "rear_floor": 0.20,
            "rear_start_deg": 90.0,
            "rear_full_deg": 150.0,
            "log_td15_brma": True,
            "log_td10_without_rear": True,
            "log_td10_with_rear": True,
        },
        "dodge": {"active": False, "log_only": True},
        "terminal": {
            "mode": "weighted_normalized_loss_fraction_per_agent",
            "coef_per_agent": 30.0,
            "mav_loss_weight_mode": "match_uav_count",
            "apply_once": True,
            "divide_by_red_initial": False,
        },
        "uav": {"event": {"kill_enemy": 200.0, "death": -200.0, "out_of_zone_active": False}},
        "mav": {
            "d_danger_m": 5000.0,
            "d_safe_m": 14000.0,
            "d_opt_m": 8000.0,
            "d_max_m": 25000.0,
            "safety_weights": {"dist": 0.5, "threat": 0.3, "aspect": 0.2},
            "safety_aggregation": {"aspect": "worst_case", "aspect_angle_source": "mav_threat_angle_3d_not_launch_ta"},
            "safety_scale": {"negative": 0.20, "positive": 0.05},
            "support_weights": {"pos": 0.6, "aware": 0.4},
            "support_aggregation": {"aware": "mean"},
            "support_scale": 0.10,
            "event": {
                "death_penalty": -300.0,
                "team_kill_credit_per_kill": 40.0,
                "team_kill_credit_cap_episode": 100.0,
                "stop_credit_after_mav_death": True,
            },
        },
        "log_only": {},
    }


def _bare_env(num_blue: int = 2):
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = [f"blue_{i}" for i in range(num_blue)]
    env.agent_roles = {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"}
    env.red_planes = {
        "red_0": FakeSim("red_0", pos=(0.0, -8000.0, 6500.0), vel=(230.0, 0.0, 0.0), missiles=0),
        "red_1": FakeSim("red_1", pos=(0.0, 0.0, 6000.0), vel=(250.0, 0.0, 0.0)),
        "red_2": FakeSim("red_2", pos=(0.0, 2000.0, 6000.0), vel=(250.0, 0.0, 0.0)),
    }
    env.blue_planes = {
        f"blue_{i}": FakeSim(f"blue_{i}", pos=(6000.0, i * 2000.0, 6000.0), vel=(250.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi), missiles=2)
        for i in range(num_blue)
    }
    env.max_num_red = 3
    env.max_num_blue = num_blue
    env.max_steps = 1000
    env.current_step = 1
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MIN = 2500.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_TA_THRESH = np.pi / 2
    env._missile_launch_range_m_effective = 14000.0
    env._step_kill_count = {aid: 0 for aid in env.red_ids + env.blue_ids}
    env.tam_paper_reward_v6_jsbsim_aligned_v3_config = _cfg()
    env.mav_observation_range_m = 80000.0
    env._last_step_obs = {}
    env._tam_v6v3_reset_episode_state()
    return env


def test_v6v3_mode_registered_and_config_loads():
    from uav_env import make_env

    env = make_env(CFG_PATH, max_steps=5)
    try:
        assert env.hetero_reward_mode == "tam_paper_reward_v6_jsbsim_aligned_v3"
        assert env.tam_paper_reward_v6_jsbsim_aligned_v3_config["situation"]["weight"] == 0.15
        assert env._needs_last_step_obs_cache()
    finally:
        env.close()


def test_missing_v6v3_config_raises():
    with pytest.raises(ValueError, match="tam_paper_reward_v6_jsbsim_aligned_v3"):
        HeteroUavCombatEnv(
            hetero_reward_mode="tam_paper_reward_v6_jsbsim_aligned_v3",
            max_num_red=3,
            max_num_blue=2,
            max_steps=10,
        )


def test_td_env_uses_effective_missile_range_and_rear_floor():
    env = _bare_env()
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    assert env._tam_v6v3_td_env(14000.0, cfg) == pytest.approx(1.0)
    env._missile_launch_range_m_effective = 20000.0
    assert env._tam_v6v3_td_env(20000.0, cfg) == pytest.approx(1.0)
    assert env._tam_v6v3_launch_rear_factor(np.deg2rad(60.0), cfg) == pytest.approx(0.2)
    assert env._tam_v6v3_launch_rear_factor(np.deg2rad(150.0), cfg) == pytest.approx(1.0)


def test_uav_situation_uses_rear_factor_and_weight():
    env = _bare_env()
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    sim = env.red_planes["red_1"]
    raw, own, threat, logs = env._tam_v6v3_situation_reward(sim, cfg)
    assert "rear_factor_mean_log" in logs
    assert np.isfinite(raw)
    reward, comp = env._tam_v6v3_uav_reward("red_1", sim, env._tam_v2_alive_blue(), cfg, {})
    assert comp["tam_v6v3_uav_situation"] == pytest.approx(0.15 * comp["tam_v6v3_uav_situation_raw_td10_rear"])
    assert comp["tam_v6v3_uav_dodge_raw_log"] == 0.0
    total_without_logs = (
        comp["tam_v6v3_uav_flight"]
        + comp["tam_v6v3_uav_situation"]
        + comp["tam_v6v3_uav_event"]
        + comp["tam_v6v3_uav_terminal"]
    )
    assert reward == pytest.approx(total_without_logs)


def test_flight_penalties_are_penalty_only_and_boundary_not_event():
    env = _bare_env()
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    assert env._tam_v6v3_altitude_penalty(6000.0, cfg) == 0.0
    assert env._tam_v6v3_altitude_penalty(2600.0, cfg) < 0.0
    assert env._tam_v6v3_speed_penalty(250.0, cfg) == 0.0
    assert env._tam_v6v3_speed_penalty(50.0, cfg) == -1.0
    sim = FakeSim("red_1", pos=(41000.0, 0.0, 6000.0))
    assert env._tam_v6v3_boundary_penalty(sim, cfg) == -10.0


def test_mav_aspect_uses_blue_body_x_to_mav_and_worst_case():
    env = _bare_env()
    mav = env.red_planes["red_0"]
    blue = FakeSim("blue_0", pos=(1000.0, -8000.0, 6500.0), rpy=(0.0, 0.0, np.pi))
    threat = env._tam_v6v3_mav_aspect_threat(mav, blue)
    assert threat > 0.9
    blue_away = FakeSim("blue_0", pos=(1000.0, -8000.0, 6500.0), rpy=(0.0, 0.0, 0.0))
    assert env._tam_v6v3_mav_aspect_threat(mav, blue_away) == pytest.approx(0.0)


def test_mav_team_credit_cap_and_no_credit_after_death_or_same_step_death():
    env = _bare_env()
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    env._step_kill_count["red_1"] = 3
    _, comp = env._tam_v6v3_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg, {})
    assert comp["tam_v6v3_mav_team_credit_delta"] == pytest.approx(100.0)
    assert comp["tam_v6v3_mav_team_credit_used"] == pytest.approx(100.0)
    env._step_kill_count["red_1"] = 1
    _, comp2 = env._tam_v6v3_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg, {})
    assert comp2["tam_v6v3_mav_team_credit_delta"] == pytest.approx(0.0)

    env = _bare_env()
    env.red_planes["red_0"].is_alive = False
    env._step_kill_count["red_1"] = 1
    _, comp3 = env._tam_v6v3_mav_reward("red_0", env.red_planes["red_0"], env._tam_v2_alive_blue(), cfg, {})
    assert comp3["tam_v6v3_mav_death"] == pytest.approx(-300.0)
    assert comp3["tam_v6v3_mav_team_credit_delta"] == pytest.approx(0.0)


def test_terminal_weighted_loss_fraction_3v2_and_5v4_mav_death():
    env = _bare_env(num_blue=2)
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    env.red_planes["red_0"].is_alive = False
    assert env._tam_v6v3_terminal_outcome(cfg) == pytest.approx(-15.0)

    env5 = _bare_env(num_blue=4)
    cfg5 = env5.tam_paper_reward_v6_jsbsim_aligned_v3_config
    env5.red_ids = ["red_0", "red_1", "red_2", "red_3", "red_4"]
    env5.agent_roles.update({f"red_{i}": "attack_uav" for i in range(1, 5)})
    env5.red_planes.update({f"red_{i}": FakeSim(f"red_{i}") for i in range(3, 5)})
    env5.red_planes["red_0"].is_alive = False
    assert env5._tam_v6v3_terminal_outcome(cfg5) == pytest.approx(-15.0)


def test_dispatch_terminal_same_all_red_and_reset_clears_state():
    env = _bare_env()
    cfg = env.tam_paper_reward_v6_jsbsim_aligned_v3_config
    env.current_step = env.max_steps
    rewards, comps = env._compute_tam_paper_reward_v6_jsbsim_aligned_v3(
        {rid: 0.0 for rid in env.red_ids},
        {rid: {} for rid in env.red_ids},
    )
    terminals = [comps[rid]["tam_v6v3_terminal"] for rid in env.red_ids]
    assert terminals[0] == terminals[1] == terminals[2]
    assert env._tam_v6v3_terminal_applied
    env._tam_v6v3_reset_episode_state()
    assert not env._tam_v6v3_terminal_applied
    assert env._tam_v6v3_mav_team_credit_used == 0.0


def test_v4_still_unaffected():
    from uav_env import make_env

    env = make_env(
        "uav_env/JSBSim/configs/"
        "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml",
        max_steps=5,
    )
    try:
        assert env.hetero_reward_mode == "tam_paper_reward_v4"
        assert not hasattr(env, "tam_paper_reward_v6_jsbsim_aligned_v3_config") or (
            env.tam_paper_reward_v6_jsbsim_aligned_v3_config == {}
        )
    finally:
        env.close()

