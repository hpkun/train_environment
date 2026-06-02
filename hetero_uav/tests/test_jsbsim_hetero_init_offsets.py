"""Test that HeteroUavCombatEnv applies init altitude offset correctly.

Does not run MAPPO, does not add MAV GCAS, does not modify reward/missile/PID.
"""
from __future__ import annotations

import numpy as np
import pytest

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
from uav_env.JSBSim.env import UavCombatEnv


@pytest.fixture
def hetero_env():
    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=20,
        red_agent_types=["mav", "attack_uav"],
        blue_agent_types=["attack_uav", "attack_uav"],
        enable_gcas_for_blue=True,
        suppress_jsbsim_output=True,
        aircraft_type_params={
            "mav": {"init_altitude_offset_m": 2000.0, "init_speed_offset_mps": 0.0},
            "attack_uav": {"init_altitude_offset_m": 0.0, "init_speed_offset_mps": 0.0},
        },
    )
    yield env
    env.close()


def test_reset_succeeds(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    assert len(obs) == 4


def test_red0_is_mav_a4(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    models = info["agent_models"]
    assert models["red_0"] == "A-4"


def test_red0_offset_2000(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    offsets = info["agent_init_offsets"]
    assert abs(offsets["red_0"]["altitude_offset_m"] - 2000.0) < 0.01


def test_other_agents_offset_zero(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    offsets = info["agent_init_offsets"]
    assert offsets["red_1"]["altitude_offset_m"] == 0.0
    assert offsets["blue_0"]["altitude_offset_m"] == 0.0
    assert offsets["blue_1"]["altitude_offset_m"] == 0.0


def test_red0_higher_than_red1(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    red0_sim = hetero_env._get_sim("red_0")
    red1_sim = hetero_env._get_sim("red_1")
    alt0 = float(red0_sim.get_geodetic()[2])
    alt1 = float(red1_sim.get_geodetic()[2])
    diff = alt0 - alt1
    assert 1500.0 < diff < 2500.0, f"expected ~2000m diff, got {diff:.1f}m"


def test_zero_policy_20_steps_no_crash(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    for _ in range(20):
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in hetero_env.agent_ids}
        obs, _rew, terminated, truncated, info = hetero_env.step(actions)
        if all(terminated.values()):
            break


def test_bounded_random_20_steps_no_crash(hetero_env):
    rng = np.random.default_rng(0)
    obs, info = hetero_env.reset(seed=0)
    for _ in range(20):
        actions = {aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
                   for aid in hetero_env.agent_ids}
        obs, _rew, terminated, truncated, info = hetero_env.step(actions)
        if all(terminated.values()):
            break


def test_brma_env_no_init_offsets():
    """UavCombatEnv should not contain agent_init_offsets."""
    env = UavCombatEnv(
        max_num_blue=1, max_num_red=1, max_steps=10,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert "agent_init_offsets" not in info, \
            "UavCombatEnv should not have agent_init_offsets"
    finally:
        env.close()


def test_no_nan(hetero_env):
    obs, info = hetero_env.reset(seed=0)
    assert not np.isnan(list(obs.values())[0]["ego_state"]).any()
