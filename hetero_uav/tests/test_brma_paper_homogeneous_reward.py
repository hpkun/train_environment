"""Tests for the BRMA paper homogeneous diagnostic reward."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


MAV_CFG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
    "brma_paper_homogeneous_v1.yaml"
)
ALL_ATTACK_CFG = (
    "uav_env/JSBSim/configs/"
    "hetero_3v2_all_attack_uav_brma_paper_homogeneous_v1.yaml"
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


def _bare_env(num_red: int = 3, num_blue: int = 2):
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = [f"red_{i}" for i in range(num_red)]
    env.blue_ids = [f"blue_{i}" for i in range(num_blue)]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {rid: ("mav" if rid == "red_0" else "attack_uav") for rid in env.red_ids}
    env.max_num_red = num_red
    env.max_num_blue = num_blue
    env.max_steps = 1000
    env.current_step = 1
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MIN = 2500.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env._missile_launch_range_m_effective = 10000.0
    env.hetero_reward_mode = "brma_paper_homogeneous_v1"
    env._brma_homo_terminal_applied = False
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env.red_planes = {
        "red_0": FakeSim("red_0", pos=(0.0, -2000.0, 6500.0), vel=(230.0, 0.0, 0.0), missiles=0),
        "red_1": FakeSim("red_1", pos=(0.0, 0.0, 6000.0), vel=(250.0, 0.0, 0.0), missiles=2),
        "red_2": FakeSim("red_2", pos=(0.0, 2000.0, 6000.0), vel=(250.0, 0.0, 0.0), missiles=2),
    }
    env.blue_planes = {
        "blue_0": FakeSim("blue_0", pos=(8000.0, 0.0, 6000.0), vel=(250.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi)),
        "blue_1": FakeSim("blue_1", pos=(16000.0, 2000.0, 6000.0), vel=(250.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi)),
    }
    return env


def test_reward_mode_registered_and_configs_load():
    from uav_env import make_env

    env = make_env(MAV_CFG, max_steps=5)
    try:
        assert env.hetero_reward_mode == "brma_paper_homogeneous_v1"
        assert env.agent_roles["red_0"] == "mav"
        assert env.aircraft_type_params["mav"]["num_missiles"] == 0
    finally:
        env.close()

    all_attack = yaml.safe_load((ROOT / ALL_ATTACK_CFG).read_text(encoding="utf-8"))
    assert all_attack["red_agent_types"] == ["attack_uav", "attack_uav", "attack_uav"]
    assert all_attack["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2
    assert all_attack["hetero_reward_mode"] == "brma_paper_homogeneous_v1"


def test_brma_homogeneous_reward_needs_last_step_obs_cache():
    from uav_env import make_env

    env = make_env(MAV_CFG, max_steps=5)
    try:
        assert env._needs_last_step_obs_cache() is True
    finally:
        env.close()


def test_brma_homogeneous_reward_populates_last_step_obs_after_reset_and_step():
    from uav_env import make_env

    env = make_env(MAV_CFG, max_steps=5)
    try:
        obs, _info = env.reset()
        assert env._last_step_obs
        assert set(env.red_ids + env.blue_ids).issubset(env._last_step_obs)
        assert set(env.red_ids + env.blue_ids).issubset(obs)

        zero_actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, *_ = env.step(zero_actions)
        assert env._last_step_obs
        assert set(env.red_ids + env.blue_ids).issubset(env._last_step_obs)
        assert set(env.red_ids + env.blue_ids).issubset(obs)
    finally:
        env.close()


def test_all_red_agents_use_same_component_structure():
    env = _bare_env()
    rewards, components = env._compute_brma_paper_homogeneous_v1({}, {})
    key_sets = [set(components[rid]) for rid in env.red_ids]
    assert key_sets[0] == key_sets[1] == key_sets[2]
    assert all("brma_homo_total" in components[rid] for rid in env.red_ids)
    assert all(rewards[rid] == pytest.approx(components[rid]["brma_homo_total"]) for rid in env.red_ids)


def test_forbidden_role_specific_or_action_event_items_not_active_components():
    env = _bare_env()
    _, components = env._compute_brma_paper_homogeneous_v1({}, {})
    forbidden = {
        "tam_v7_mav_safety",
        "tam_v7_mav_support",
        "tam_v7_mav_team_credit_delta",
        "uav_first_out_of_zone",
        "uav_fire",
        "uav_hit",
        "dodge",
        "shared_track",
        "launch",
        "low_speed_exploit_penalty",
    }
    for comp in components.values():
        assert not any(any(token in key for token in forbidden) for key in comp)


def test_td_uses_brma_15km_not_effective_launch_range():
    env = _bare_env()
    env._missile_launch_range_m_effective = 10000.0
    assert env._brma_homo_td15(10000.0) == pytest.approx(1.0)
    assert env._brma_homo_td15(15000.0) == pytest.approx(1.0)
    assert env._brma_homo_td15(20000.0) == pytest.approx(np.exp(1.0 - 20000.0 / 15000.0))


def test_boundary_only_checks_horizontal_position_not_altitude():
    env = _bare_env()
    sim = FakeSim("red_x", pos=(0.0, 0.0, 12000.0))
    assert env._brma_homo_boundary(sim) == pytest.approx(0.0)
    sim._pos = np.asarray([41000.0, 0.0, 6000.0])
    assert env._brma_homo_boundary(sim) == pytest.approx(-10.0)


def test_terminal_r_end_applied_full_value_to_all_red_agents_once():
    env = _bare_env()
    env.current_step = env.max_steps
    env.red_planes["red_2"].is_alive = False
    # red_alive=2, blue_alive=2 => terminal 0 first.
    rewards, components = env._compute_brma_paper_homogeneous_v1({}, {})
    assert all(components[rid]["brma_homo_r_end"] == pytest.approx(0.0) for rid in env.red_ids)

    env = _bare_env()
    env.current_step = env.max_steps
    env.blue_planes["blue_1"].is_alive = False
    # red_alive=3, blue_alive=1 => 30*(3-1)=60; not divided by red count.
    rewards, components = env._compute_brma_paper_homogeneous_v1({}, {})
    assert all(components[rid]["brma_homo_r_end"] == pytest.approx(60.0) for rid in env.red_ids)
    assert all(components[rid]["brma_homo_terminal_applied"] == pytest.approx(1.0) for rid in env.red_ids)
    _, components2 = env._compute_brma_paper_homogeneous_v1({}, {})
    assert all(components2[rid]["brma_homo_r_end"] == pytest.approx(0.0) for rid in env.red_ids)


def test_fire_launch_dodge_shared_track_do_not_enter_total():
    env = _bare_env()
    env._launch_quality_step_records = [{"launch_track_source": "mav_shared"}]
    env._launch_quality_done_step_records = [{"raw_termination_reason": "hit"}]
    rewards, components = env._compute_brma_paper_homogeneous_v1({}, {})
    for rid, comp in components.items():
        expected = (
            0.01 * comp["brma_homo_r_pitch"]
            + 0.002 * comp["brma_homo_r_roll"]
            + 0.04 * comp["brma_homo_r_altitude"]
            + 0.04 * comp["brma_homo_r_boundary"]
            + 0.02 * comp["brma_homo_r_speed"]
            + 0.15 * comp["brma_homo_r_adv"]
            + comp["brma_homo_r_end"]
        )
        assert rewards[rid] == pytest.approx(expected)
        assert comp["brma_homo_total"] == pytest.approx(expected)
