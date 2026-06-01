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


def _load_model_and_run_ic(model: str) -> bool:
    jsbsim = pytest.importorskip("jsbsim")
    fdm = jsbsim.FGFDMExec(str(MODEL_ROOT))
    fdm.set_debug_level(0)
    if hasattr(fdm, "set_aircraft_path"):
        fdm.set_aircraft_path(str(MODEL_ROOT / "aircraft"))
    if hasattr(fdm, "set_engine_path"):
        fdm.set_engine_path(str(MODEL_ROOT / "engine"))
    assert fdm.load_model(model)
    props = {
        "ic/long-gc-deg": 120.0,
        "ic/lat-geod-deg": 60.0,
        "ic/h-sl-ft": 6000.0 / 0.3048,
        "ic/psi-true-deg": 0.0,
        "ic/theta-deg": 0.0,
        "ic/phi-deg": 0.0,
        "ic/u-fps": 250.0 / 0.3048,
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
        "ic/terrain-elevation-ft": 0.0,
    }
    for name, value in props.items():
        fdm.set_property_value(name, float(value))
    assert fdm.run_ic()
    return True


@pytest.mark.parametrize("model", ["A-4", "F-16"])
def test_jsbsim_load_model_and_run_ic(model: str):
    assert _load_model_and_run_ic(model)


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
    for _ in range(5):
        platform.step(np.array([0.0, 0.0, 0.5], dtype=np.float32), 0.2, (102.0, 408.0))
    assert platform.position.shape == (3,)
    assert np.isfinite(platform.position).all()
    assert np.isfinite(platform.velocity).all()
    platform.close()


def test_hetero_env_jsbsim_backend_reset_and_step():
    pytest.importorskip("jsbsim")
    env = make_env("uav_env/configs/hetero_2v2_jsbsim_debug.yaml")
    obs, info = env.reset(seed=0)
    assert set(obs) == set(env.agent_ids)
    assert info["agent_types"]["red_0"] == "mav"
    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    for _ in range(3):
        obs, rewards, terminated, truncated, info = env.step(actions)
    assert set(obs) == set(env.agent_ids)
    assert set(rewards) == set(env.agent_ids)
    assert "blue_alive" in info
    env.close()
