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

from scripts.experiment_logging_schema import (  # noqa: E402
    EPISODE_REWARD_COMPONENTS_COLUMNS,
    REWARD_COMPONENT_COLUMNS,
)
from scripts.rich_logging import RichExperimentLogger  # noqa: E402
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv  # noqa: E402


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_table1_v1.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_happo_table1_v1.yaml"
)


class FakeSim:
    def __init__(
        self,
        uid: str,
        *,
        pos=(0.0, 0.0, 6000.0),
        vel=(250.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        alive=True,
        warning=False,
    ):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.is_alive = bool(alive)
        self._warning = bool(warning)
        self.under_missiles = []

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return self._rpy

    def get_geodetic(self):
        return np.asarray([0.0, 0.0, self._pos[2]], dtype=np.float64)

    def check_missile_warning(self):
        return object() if self._warning else None


def _cfg():
    data = yaml.safe_load((ROOT / CFG_3V2).read_text(encoding="utf-8"))
    return data["tam_happo_table1_v1"]


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
        "red_0": FakeSim("red_0", pos=(0.0, 0.0, 6500.0), vel=(230.0, 0.0, 0.0)),
        "red_1": FakeSim("red_1", pos=(0.0, 0.0, 6000.0), vel=(300.0, 0.0, 0.0)),
        "red_2": FakeSim("red_2", pos=(0.0, 4000.0, 6000.0), vel=(300.0, 0.0, 0.0)),
    }
    env.blue_planes = {
        "blue_0": FakeSim("blue_0", pos=(4000.0, 0.0, 6000.0), vel=(230.0, 0.0, 0.0), rpy=(0.0, 0.0, math.pi)),
        "blue_1": FakeSim("blue_1", pos=(12000.0, 0.0, 6000.0), vel=(230.0, 0.0, 0.0), rpy=(0.0, 0.0, math.pi)),
    }
    env._last_step_obs = {
        "red_0": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
        "red_1": {"enemy_track_source": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)},
        "red_2": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
    }
    env._launch_quality_step_records = [
        {"shooter_id": "red_1", "launch_track_source": "mav_shared"},
    ]
    env._launch_quality_done_step_records = [
        {"shooter_id": "red_1", "launch_track_source": "mav_shared", "raw_termination_reason": "hit"},
    ]
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env.tam_happo_table1_v1_config = _cfg()
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MIN = 2500.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 14000.0
    env.MISSILE_LAUNCH_AO_THRESH = np.deg2rad(60.0)
    env.MISSILE_LAUNCH_TA_THRESH = np.deg2rad(90.0)
    env.mav_observation_range_m = 80000.0
    env.red_uav_track_policy = "direct_or_mav_shared"
    env._tam_happo_table1_v1_reset_episode_state()
    return env


def _base_components():
    rewards = {"red_0": 999.0, "red_1": 999.0, "red_2": 999.0}
    components = {
        "red_0": {"r_adv": 6.0, "r_end": 7.0, "total": 999.0},
        "red_1": {"r_adv": 60.0, "r_end": 70.0, "total": 999.0},
        "red_2": {"r_adv": 80.0, "r_end": 90.0, "total": 999.0},
    }
    return rewards, components


def test_table1_mode_registered_and_configs_load():
    from uav_env import make_env

    for cfg, max_red, max_blue in [(CFG_3V2, 3, 2), (CFG_5V4, 5, 4)]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "tam_happo_table1_v1"
            assert env.observation_mode == "mav_shared_geo"
            assert env.agent_roles["red_0"] == "mav"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.aircraft_type_params["attack_uav"]["num_missiles"] == 2
            assert env.red_target_selection_mode == "closest"
            assert env.max_num_red == max_red
            assert env.max_num_blue == max_blue
        finally:
            env.close()


def test_3v2_config_reset_and_step():
    from uav_env import make_env

    env = make_env(CFG_3V2, max_steps=5)
    try:
        obs, info = env.reset(seed=7)
        assert obs
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert set(env.red_ids) <= set(rewards)
        assert info["reward_mode"] == "tam_happo_table1_v1"
        assert "reward_components" in info
        assert any("tam_table1_" in key for key in info["reward_components"]["red_1"])
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
    finally:
        env.close()


def test_missing_table1_config_raises():
    with pytest.raises(ValueError, match="tam_happo_table1_v1"):
        HeteroUavCombatEnv(
            hetero_reward_mode="tam_happo_table1_v1",
            max_num_red=3,
            max_num_blue=2,
            max_steps=5,
        )


def test_uav_event_once_and_no_brma_active_reward():
    env = _bare_env()
    env._step_kill_count["red_1"] = 1
    rewards, components = _base_components()

    rewards, components = env._compute_tam_happo_table1_v1(rewards, components)
    comp = components["red_1"]

    assert comp["tam_table1_uav_kill"] == pytest.approx(200.0)
    assert comp["tam_table1_uav_brma_adv_log"] == pytest.approx(60.0)
    assert comp["tam_table1_uav_brma_end_log"] == pytest.approx(70.0)
    assert comp["tam_table1_uav_total"] == pytest.approx(rewards["red_1"])
    assert comp["tam_table1_uav_total"] != pytest.approx(60.0 + 70.0)

    env.red_planes["red_1"].is_alive = False
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_1"]["tam_table1_uav_death"] == pytest.approx(-200.0)
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_1"]["tam_table1_uav_death"] == pytest.approx(0.0)

    env.red_planes["red_2"]._pos = np.asarray([41000.0, 0.0, 6000.0])
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_2"]["tam_table1_uav_out_of_zone"] == pytest.approx(-100.0)
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_2"]["tam_table1_uav_out_of_zone"] == pytest.approx(0.0)


def test_mav_event_death_once_and_team_credit_cap():
    env = _bare_env()
    env._step_kill_count["red_1"] = 3
    rewards, components = _base_components()

    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_0"]["tam_table1_mav_team_credit_delta"] == pytest.approx(200.0)
    assert components["red_0"]["tam_table1_mav_team_credit_used"] == pytest.approx(200.0)

    env.red_planes["red_0"].is_alive = False
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_0"]["tam_table1_mav_death"] == pytest.approx(-200.0)
    rewards, components = _base_components()
    _, components = env._compute_tam_happo_table1_v1(rewards, components)
    assert components["red_0"]["tam_table1_mav_death"] == pytest.approx(0.0)


def test_mav_awareness_uses_continuous_ao_not_binary_observed():
    env = _bare_env()
    reward, observed = env._tam_table1_mav_awareness(env.red_planes["red_0"])

    assert observed == pytest.approx(2.0)
    assert reward > 0.3
    assert reward != pytest.approx(2.0)


def test_no_terminal_outcome_reward_is_added():
    env = _bare_env()
    rewards, components = _base_components()

    rewards, components = env._compute_tam_happo_table1_v1(rewards, components)

    for rid in env.red_ids:
        comp = components[rid]
        assert "tam_v7_terminal_per_agent" not in comp
        assert "brma_proxy_terminal_team" not in comp
        assert comp.get("r_end", 0.0) in (0.0, 7.0, 70.0, 90.0)
        assert rewards[rid] == pytest.approx(comp["tam_table1_total"])


def test_mav_role_blocked_launch_rule_unchanged():
    env = _bare_env()
    has_track, source = env._has_launch_track("red_0", "blue_0")
    assert has_track is False
    assert source == "role_blocked_mav"


def test_logging_schema_has_table1_fields_and_logger_writes(tmp_path):
    required = {
        "tam_table1_uav_total",
        "tam_table1_mav_total",
        "tam_table1_total",
        "tam_table1_uav_total_sum",
        "tam_table1_mav_total_sum",
        "tam_table1_total_sum",
    }
    assert {"tam_table1_uav_total", "tam_table1_mav_total", "tam_table1_total"} <= set(REWARD_COMPONENT_COLUMNS)
    assert {"tam_table1_uav_total_sum", "tam_table1_mav_total_sum", "tam_table1_total_sum"} <= set(
        EPISODE_REWARD_COMPONENTS_COLUMNS
    )

    logger = RichExperimentLogger(
        tmp_path,
        run_id="unit",
        method_name="unit",
        scenario_name="unit",
        device="cpu",
        num_envs=1,
        rollout_length_per_env=1,
        transitions_per_rollout=1,
        mode="full",
    )
    try:
        logger.write_reward_components(
            {
                "reward_components": {
                    "red_1": {
                        "tam_table1_uav_total": 123.0,
                        "tam_table1_total": 123.0,
                    }
                },
                "red_1": {"role": "attack_uav"},
            },
            scenario="unit",
            episode_id=0,
            step=1,
            sim_time=0.2,
        )
        logger.write_episode_reward_components(
            scenario="unit",
            episode_id=0,
            agent_id="red_1",
            role="attack_uav",
            team="red",
            episode_length=1,
            episode_return=123.0,
            component_sums={"tam_table1_uav_total_sum": 123.0, "tam_table1_total_sum": 123.0},
        )
    finally:
        logger.close()

    reward_text = (tmp_path / "reward_components.csv").read_text(encoding="utf-8")
    episode_text = (tmp_path / "episode_reward_components.csv").read_text(encoding="utf-8")
    assert "tam_table1_uav_total" in reward_text
    assert "123.0" in reward_text
    assert "tam_table1_uav_total_sum" in episode_text
    assert "123.0" in episode_text


def test_configs_do_not_override_forbidden_environment_contracts():
    forbidden = {
        "missile_launch_range_m",
        "missile_launch_ao_deg",
        "missile_launch_ta_deg",
        "missile_launch_min_range_m",
        "missile_attack_interval_sec",
    }
    for cfg in [CFG_3V2, CFG_5V4]:
        data = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        assert data["hetero_reward_mode"] == "tam_happo_table1_v1"
        assert data["red_target_selection_mode"] == "closest"
        assert "tam_paper_reward_v7_role_aligned" not in data
        assert not (forbidden & set(data))
        assert data["aircraft_type_params"]["mav"]["num_missiles"] == 0
        assert data["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2
        assert data["action_trim_by_role"]["mav"]["pitch"] == pytest.approx(0.0)
