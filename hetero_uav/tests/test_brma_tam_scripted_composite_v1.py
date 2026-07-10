from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv  # noqa: E402


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)


class FakeSim:
    def __init__(self, uid, *, pos, vel, alive=True, warning=False):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.is_alive = bool(alive)
        self._warning = bool(warning)
        self.under_missiles = []

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return np.zeros(3, dtype=np.float64)

    def get_geodetic(self):
        return np.asarray([0.0, 0.0, self._pos[2]], dtype=np.float64)

    def check_missile_warning(self):
        return object() if self._warning else None


def _cfg():
    data = yaml.safe_load((ROOT / CFG_3V2).read_text(encoding="utf-8"))
    return data["brma_tam_scripted_composite_v1"]


def _bare_env():
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
        "red_0": FakeSim("red_0", pos=(0.0, 0.0, 6500.0), vel=(250.0, 0.0, 0.0)),
        "red_1": FakeSim("red_1", pos=(0.0, 0.0, 6000.0), vel=(300.0, 0.0, 0.0)),
        "red_2": FakeSim("red_2", pos=(0.0, 4000.0, 6000.0), vel=(300.0, 0.0, 0.0)),
    }
    env.blue_planes = {
        "blue_0": FakeSim("blue_0", pos=(4000.0, 0.0, 6000.0), vel=(230.0, 0.0, 0.0)),
        "blue_1": FakeSim("blue_1", pos=(12000.0, 0.0, 7000.0), vel=(230.0, 0.0, 0.0)),
    }
    env._last_step_obs = {
        "red_0": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
        "red_1": {"enemy_track_source": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)},
        "red_2": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
    }
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env._death_reasons = {}
    env._evasion_step_records = []
    env._launch_quality_step_records = []
    env._reward_target_diagnostic_records = []
    env.brma_tam_scripted_composite_v1_config = _cfg()
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 14000.0
    env.mav_observation_range_m = 80000.0
    env._brma_tam_scripted_reset_episode_state()
    env._brma_tam_alive_before_step = {aid: True for aid in env.agent_ids}
    return env


def _base_components():
    rewards = {"red_0": 999.0, "red_1": 999.0, "red_2": 999.0}
    components = {
        rid: {
            "r_pitch": 1.0,
            "r_roll": 2.0,
            "r_vel": 3.0,
            "r_alt": 100.0,
            "r_bound": 200.0,
            "r_adv": 300.0,
            "r_end": 400.0,
            "r_death": -500.0,
            "total": 999.0,
        }
        for rid in rewards
    }
    return rewards, components


def test_composite_mode_configs_load_and_contract():
    from uav_env import make_env

    for cfg in [CFG_3V2, CFG_5V4]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "brma_tam_scripted_composite_v1"
            assert env.observation_mode == "mav_shared_geo"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.red_target_selection_mode == "closest"
            assert env.missile_evasion_mode == "brma_scripted"
            assert env.missile_evasion_teams in {"red_only", "both"}
        finally:
            env.close()


def test_config_contract_rejects_disabled_red_scripted_evasion():
    with pytest.raises(ValueError, match="red evasion is scripted"):
        HeteroUavCombatEnv(
            hetero_reward_mode="brma_tam_scripted_composite_v1",
            brma_tam_scripted_composite_v1=_cfg(),
            observation_mode="mav_shared_geo",
            missile_evasion={"mode": "none", "teams": "none"},
            max_num_red=3,
            max_num_blue=2,
            max_steps=5,
        )


def test_3d_angle_contract_cases():
    h = HeteroUavCombatEnv
    red = FakeSim("r", pos=(0.0, 0.0, 0.0), vel=(300.0, 0.0, 0.0))
    blue = FakeSim("b", pos=(1000.0, 0.0, 0.0), vel=(300.0, 0.0, 0.0))
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_ata_rad"] == pytest.approx(0.0, abs=1e-6)
    assert g["tam_aa_rad"] == pytest.approx(0.0, abs=1e-6)
    assert g["tam_angle_raw"] == pytest.approx(1.0, abs=1e-6)

    blue._vel = np.asarray([-300.0, 0.0, 0.0])
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_angle_raw"] == pytest.approx(0.0, abs=1e-6)

    blue._pos = np.asarray([-1000.0, 0.0, 0.0])
    blue._vel = np.asarray([300.0, 0.0, 0.0])
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_angle_raw"] == pytest.approx(-1.0, abs=1e-6)

    blue._pos = np.asarray([1000.0, 0.0, 1000.0])
    blue._vel = np.asarray([300.0, 0.0, 0.0])
    g3 = h._brma_tam_3d_geometry(red, blue)
    assert g3["target_distance_m"] == pytest.approx(math.sqrt(2) * 1000.0)
    assert g3["tam_ata_rad"] > 0.0

    red._vel = np.zeros(3)
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_geometry_valid"] == 0.0
    assert np.isfinite(g["tam_angle_raw"])


def test_speed_and_distance_formulas():
    h = HeteroUavCombatEnv
    assert h._brma_tam_speed_raw(300.0, 300.0)["tam_speed_raw"] == pytest.approx(0.0)
    assert h._brma_tam_speed_raw(300.0, 120.0)["tam_speed_raw"] == pytest.approx(1.0)
    assert h._brma_tam_speed_raw(300.0, 450.0)["tam_speed_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_speed_raw(300.0, 600.0)["tam_speed_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_speed_raw(0.0, 200.0)["speed_ratio_valid"] == pytest.approx(0.0)
    assert h._brma_tam_speed_raw(0.0, 200.0)["tam_speed_raw"] == pytest.approx(0.0)

    assert h._brma_tam_distance_raw(5000.0)["tam_distance_raw"] == pytest.approx(1.0)
    assert 0.0 < h._brma_tam_distance_raw(7500.0)["tam_distance_raw"] < 1.0
    assert h._brma_tam_distance_raw(10000.0)["tam_distance_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_distance_raw(float("nan"))["reward_distance_zone_code"] == pytest.approx(-1.0)


def test_uav_total_whitelists_terms_and_excludes_dodge():
    env = _bare_env()
    env._step_kill_count["red_1"] = 1
    rewards, components = _base_components()

    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    comp = components["red_1"]

    expected = (
        1.0 + 2.0 + 3.0
        + 10.0 * comp["tam_speed_raw"]
        + 15.0 * comp["tam_angle_raw"]
        + 10.0 * comp["tam_distance_raw"]
        + comp["uav_event_kill"]
        + comp["uav_event_loss"]
    )
    assert rewards["red_1"] == pytest.approx(expected)
    assert comp["brma_alt_log_only"] == pytest.approx(100.0)
    assert comp["brma_adv_log_only"] == pytest.approx(300.0)
    assert comp["tam_dodge_raw_log"] == pytest.approx(0.0)
    assert comp["uav_total"] == pytest.approx(expected)


def test_dead_before_step_gets_zero_but_same_step_death_gets_loss():
    env = _bare_env()
    env.red_planes["red_1"].is_alive = False
    env._brma_tam_alive_before_step["red_1"] = False
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert rewards["red_1"] == pytest.approx(0.0)

    env = _bare_env()
    env.red_planes["red_1"].is_alive = False
    env._brma_tam_alive_before_step["red_1"] = True
    env._death_reasons["red_1"] = "Missile_Kill"
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert components["red_1"]["uav_event_loss"] == pytest.approx(-200.0)


def test_mav_team_credit_not_awarded_after_previous_death():
    env = _bare_env()
    env.red_planes["red_0"].is_alive = False
    env._brma_tam_alive_before_step["red_0"] = False
    env._step_kill_count["red_1"] = 1
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert rewards["red_0"] == pytest.approx(0.0)
    assert components["red_0"]["mav_team_credit_delta"] == pytest.approx(0.0)


def test_mav_threat_uses_actual_incoming_missile():
    env = _bare_env()
    missile = type("M", (), {"is_alive": True, "uid": "m1"})()
    env.red_planes["red_0"].under_missiles = [missile]
    rewards, components = _base_components()
    _, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert components["red_0"]["mav_threat_raw"] == pytest.approx(-1.0)
    assert components["red_0"]["mav_actual_incoming_missile_count"] == pytest.approx(1.0)

