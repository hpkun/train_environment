"""Verify TAM MissileSimulator aligns with hetero fixed-speed AAM."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


class _FakeAircraft:
    def __init__(self, uid, color, position, velocity, posture=(0,0,0), alive=True, dt=1/60):
        self.uid = uid; self.color = color; self.dt = dt
        self.lon0=120.0; self.lat0=60.0; self.alt0=0.0
        self.launch_missiles=[]; self.under_missiles=[]
        self._position=np.asarray(position, dtype=np.float64)
        self._velocity=np.asarray(velocity, dtype=np.float64)
        self._posture=np.asarray(posture, dtype=np.float64)
        self._alive=alive
    @property
    def is_alive(self): return self._alive
    def shotdown(self): self._alive=False
    def get_geodetic(self): return np.asarray([120,60,self._position[2]], dtype=np.float64)
    def get_position(self): return self._position
    def get_velocity(self): return self._velocity
    def get_rpy(self): return self._posture


def test_missile_speed_is_600():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair()
    assert m._missile_speed_mps == 600.0


def test_no_legacy_params():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair()
    for legacy in ['_v_min','_t_thrust','_Isp','_Length','_Diameter','_cD','_m0','_dm','_distance_pre','_distance_increment']:
        assert not hasattr(m, legacy), f"Legacy param {legacy} should not exist"


def test_launch_speed_near_600():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair(parent_velocity=(250,0,0))
    assert abs(np.linalg.norm(m.get_velocity()) - 600.0) < 1.0


def test_run_no_low_speed_termination():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair(parent_velocity=(30,0,0))
    for _ in range(120):
        m.run()
        if m.is_done:
            break
    assert m._termination_reason != "low_speed"
    assert "low_speed" not in str(m._termination_reason)


def test_run_no_overshoot_termination():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair(parent_velocity=(250,0,0), target_position=(-4000,0,6000))
    m._t_max = 0.2
    while not m.is_done:
        m.run()
    assert m._termination_reason == "timeout"
    assert m._termination_reason != "overshoot"


def test_roll_hit_probability_exists():
    from uav_env.JSBSim.simulator import MissileSimulator
    m, _, _ = _missile_pair()
    assert hasattr(m, '_roll_hit_probability')
    assert callable(m._roll_hit_probability)


def test_env_reset_step_ok():
    from uav_env import make_env
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"),
                   env_type="jsbsim_hetero", hetero_reward_mode="tam_paper_reward_v1")
    obs, info = env.reset(seed=42)
    for _ in range(5):
        actions = {rid: np.random.randint(0,40,4).astype(np.int64) for rid in env.red_ids}
        obs, rew, term, trunc, info = env.step(actions)
        for v in rew.values():
            assert np.isfinite(v)
    env.close()


def _missile_pair(parent_velocity=(250,0,0), target_position=(4000,0,6000)):
    from uav_env.JSBSim.simulator import MissileSimulator
    shooter = _FakeAircraft("red_1","Red",(0,0,6000),parent_velocity,posture=(0,0,0))
    target = _FakeAircraft("blue_0","Blue",target_position,(250,0,0))
    missile = MissileSimulator.create(shooter, target, "m0")
    return missile, shooter, target
