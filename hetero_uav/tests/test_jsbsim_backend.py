from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.core.aircraft import JSBSimAircraftPlatform
from uav_env.JSBSim.core.aircraft_types import AircraftType

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "uav_env" / "JSBSim" / "models"


def test_jsbsim_model_files_exist():
    assert (MODEL_ROOT / "aircraft" / "A-4" / "A-4.xml").exists()
    assert (MODEL_ROOT / "aircraft" / "F-16" / "F-16.xml").exists()
    assert (MODEL_ROOT / "engine" / "J52.xml").exists()
    assert (MODEL_ROOT / "engine" / "F100-PW-229.xml").exists()
    assert (MODEL_ROOT / "engine" / "direct.xml").exists()


def _aircraft_type(model: str) -> AircraftType:
    role = "mav" if model == "A-4" else "attack_uav"
    return AircraftType(
        name=role,
        aircraft_model=model,
        model_path=f"uav_env/JSBSim/models/aircraft/{model}/{model}.xml",
        role=role,
        radar_range=90000.0,
        missile_num=2,
        max_speed_scale=1.0,
        max_g=9.0,
        reward_role="leader_survival" if role == "mav" else "attack",
    )


@pytest.mark.parametrize("model", ["A-4", "F-16"])
def test_jsbsim_aircraft_platform_runs_one_step(model: str):
    pytest.importorskip("jsbsim")
    platform = JSBSimAircraftPlatform(
        "test_0",
        "red",
        _aircraft_type(model),
        np.array([0.0, 0.0, 6000.0], dtype=np.float32),
        np.array([250.0, 0.0, 0.0], dtype=np.float32),
        0.0,
        model_root=str(MODEL_ROOT),
        model_name=model,
        reference_lat=60.0,
        reference_lon=120.0,
        reference_alt=0.0,
        simulation_frequency=60,
    )
    platform.step(np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.2, (102.0, 408.0))
    assert platform.position.shape == (3,)
    assert np.isfinite(platform.position).all()
    assert np.isfinite(platform.velocity).all()
    platform.close()


def test_hetero_env_jsbsim_backend_reset_and_step():
    pytest.importorskip("jsbsim")
    env = make_env(
        "uav_env/configs/hetero_train_2v2_mav_attack.yaml",
        dynamics_backend="jsbsim",
        episode_limit=2,
    )
    obs, info = env.reset(seed=0)
    assert set(obs) == set(env.agent_ids)
    assert info["agent_types"]["red_0"] == "mav"
    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert set(obs) == set(env.agent_ids)
    assert set(rewards) == set(env.agent_ids)
    assert "blue_alive" in info
    env.close()
