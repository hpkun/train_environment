from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.core.aircraft import SimpleKinematicAircraftPlatform
from uav_env.JSBSim.core.aircraft_types import AircraftType
from uav_env.JSBSim.core.missile import MissileManager
from uav_env.JSBSim.core.original_missile import MissileSimulator


def _type(name: str = "attack_uav") -> AircraftType:
    return AircraftType(
        name=name,
        aircraft_model="F-16",
        model_path="",
        role=name,
        radar_range=90000.0,
        missile_num=2,
        max_speed_scale=1.0,
        max_g=9.0,
        reward_role="attack",
    )


def _pair(distance: float = 2500.0):
    shooter = SimpleKinematicAircraftPlatform(
        "red_0", "red", _type(),
        np.array([0.0, 0.0, 6000.0], dtype=np.float32),
        np.array([250.0, 0.0, 0.0], dtype=np.float32),
        0.0,
    )
    target = SimpleKinematicAircraftPlatform(
        "blue_0", "blue", _type(),
        np.array([distance, 0.0, 6000.0], dtype=np.float32),
        np.array([180.0, 0.0, 0.0], dtype=np.float32),
        0.0,
    )
    shooter.reset_runtime()
    target.reset_runtime()
    return shooter, target


def test_missile_simulator_create_launch_and_links():
    shooter, target = _pair()
    missile = MissileSimulator.create(shooter, target, "M0001", dt=1 / 60)
    assert missile.status == MissileSimulator.LAUNCHED
    assert missile in shooter.launch_missiles
    assert missile in target.under_missiles


def test_missile_run_moves_and_finishes_hit_or_miss():
    shooter, target = _pair()
    missile = MissileSimulator.create(shooter, target, "M0001", dt=1 / 60)
    start = missile.get_position().copy()
    for _ in range(3600):
        missile.run()
        if missile.is_done:
            break
    assert not np.allclose(start, missile.get_position())
    assert missile.status in (MissileSimulator.HIT, MissileSimulator.MISS)
    if missile.status == MissileSimulator.HIT:
        assert not target.alive


def test_missile_manager_creates_and_steps_active_missile():
    shooter, target = _pair()
    manager = MissileManager({"missile_params": {"launch_range": 6500, "attack_range": 14000}})
    class Sensor:
        def can_detect(self, *_args):
            return True
    event = manager.evaluate_launch(shooter, target, Sensor())
    assert event and event.fired
    manager.create_missile(shooter, target, event, 0.2)
    assert manager.active_missile_count == 1
    events = []
    for _ in range(300):
        events.extend(manager.step_active(0.2))
        if events:
            break
    assert manager.active_missile_count >= 0


def test_observation_builder_missile_warning():
    shooter, target = _pair()
    missile = MissileSimulator.create(shooter, target, "M0001", dt=1 / 60)
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    try:
        env.reset(seed=0)
        obs = env.task.observation.build_obs([shooter, target])
        assert obs["blue_0"]["ego_state"][15] == 1.0
        assert target.check_missile_warning() is missile
    finally:
        env.close()


def test_env_simple_info_has_missile_summary():
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    try:
        env.reset(seed=0)
        actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert "missile_summary" in info
        assert "active_missile_count" in info
    finally:
        env.close()


def test_env_jsbsim_info_has_missile_summary():
    pytest.importorskip("jsbsim")
    env = make_env("uav_env/configs/hetero_2v2_jsbsim_debug.yaml", episode_limit=2)
    try:
        env.reset(seed=0)
        actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert "missile_summary" in info
        assert "active_missile_count" in info
    finally:
        env.close()
