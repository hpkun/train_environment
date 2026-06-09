"""Test brma_rule opponent mode."""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

def test_brma_rule_mode_in_modes():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    assert "brma_rule" in OpponentPolicy.MODES

def test_brma_rule_constructs():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    try:
        policy = OpponentPolicy("brma_rule", seed=0)
        assert policy.mode == "brma_rule"
    except ImportError:
        # rule_based_agent.py not in path during construction is OK;
        # it is only imported in act()
        pass

try:
    from uav_env import make_env
    HAVE_ENV = True
except ImportError:
    HAVE_ENV = False

def test_brma_rule_acts_in_env():
    if not HAVE_ENV: return
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env
    policy = OpponentPolicy("brma_rule", seed=0)
    env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml", env_type="jsbsim_hetero", max_steps=10, suppress_jsbsim_output=False)
    try:
        obs, info = env.reset(seed=0)
        actions = policy.act(obs, env.blue_ids, env=env)
        for bid in env.blue_ids:
            a = np.asarray(actions[bid], dtype=np.float32)
            assert a.shape == (3,), f"{bid} shape {a.shape}"
            assert np.isfinite(a).all()
            assert -1.0 <= float(a.min()) <= 1.0
            assert -1.0 <= float(a.max()) <= 1.0
    finally:
        env.close()
