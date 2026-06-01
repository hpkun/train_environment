from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _assert_no_backup_or_parent_imports():
    forbidden = [
        name
        for name in sys.modules
        if name == "my_uav_env"
        or name.startswith("my_uav_env.")
        or name == "uav_env.brma_env"
        or name.startswith("uav_env.brma_env.")
    ]
    assert forbidden == []


def test_jsbsim_brma_import_does_not_load_backup_or_parent():
    from uav_env.JSBSim.envs.uav_combat_env import UavCombatEnv

    assert UavCombatEnv is not None
    _assert_no_backup_or_parent_imports()


def test_jsbsim_brma_reset_and_random_step():
    from uav_env.JSBSim.envs.uav_combat_env import UavCombatEnv

    env = UavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        num_missiles_per_plane=1,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.agent_ids)
        assert isinstance(info, dict)
        actions = {
            aid: env.action_space.spaces[aid].sample().astype(np.float32)
            for aid in env.agent_ids
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert set(obs.keys()) == set(env.agent_ids)
        assert set(rewards.keys()) == set(env.agent_ids)
        assert set(terminated.keys()) == set(env.agent_ids)
        assert set(truncated.keys()) == set(env.agent_ids)
        assert isinstance(info, dict)
        _assert_no_backup_or_parent_imports()
    finally:
        env.close()


def test_jsbsim_aircraft_simulator_loads_f16():
    from uav_env.JSBSim.simulator import AircraftSimulator

    sim = AircraftSimulator(
        uid="test_f16",
        color="Red",
        model="f16",
        num_missiles=1,
        sim_freq=60,
        suppress_jsbsim_output=True,
    )
    try:
        assert sim.is_alive
        assert sim.model == "f16"
        _assert_no_backup_or_parent_imports()
    finally:
        sim.close()


def test_jsbsim_missile_simulator_can_be_created():
    from uav_env.JSBSim.simulator import AircraftSimulator, MissileSimulator

    shooter = AircraftSimulator(
        uid="red_0",
        color="Red",
        model="f16",
        init_state={"ic/east-ft": -10000.0, "ic/psi-true-deg": 90.0},
        num_missiles=1,
        sim_freq=60,
        suppress_jsbsim_output=True,
    )
    target = AircraftSimulator(
        uid="blue_0",
        color="Blue",
        model="f16",
        init_state={"ic/east-ft": 10000.0, "ic/psi-true-deg": 270.0},
        num_missiles=1,
        sim_freq=60,
        suppress_jsbsim_output=True,
    )
    try:
        missile = MissileSimulator.create(shooter, target, uid="missile_0")
        assert missile.parent_aircraft is shooter
        assert missile.target_aircraft is target
        assert missile.is_alive
        _assert_no_backup_or_parent_imports()
    finally:
        shooter.close()
        target.close()
