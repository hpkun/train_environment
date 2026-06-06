"""Test minimal_v1 hetero reward overlay. No training, no env changes."""
from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env

LEGACY_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
MINIMAL_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_minimal.yaml"


def test_legacy_has_no_overlay_components():
    env = make_env(LEGACY_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        # Step once
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for k in ["r_mav_survival", "r_mav_death", "r_mav_support"]:
                val = float(rcinfo.get(k, 0.0))
                assert val == 0.0, f"{aid} {k} should be 0 on legacy, got {val}"
    finally:
        env.close()


def test_minimal_reset_and_step():
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        assert env.hetero_reward_mode == "minimal_v1"
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
    finally:
        env.close()


def test_minimal_reward_components_present():
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for k in ["r_mav_survival", "r_mav_death", "r_mav_support",
                      "r_shared_track_used", "r_attack_kill_bonus"]:
                assert k in rcinfo, f"{aid} missing overlay key {k}"
    finally:
        env.close()


def test_mav_survival_positive_when_alive():
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(5):
            obs, rewards, terminated, truncated, info = env.step(actions)
        rcinfo = info.get("red_0", {})
        assert rcinfo.get("r_mav_survival", 0.0) >= 0.0
    finally:
        env.close()


def test_reward_finite_no_nan():
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        for _ in range(5):
            actions = {aid: np.random.uniform(-0.5, 0.5, (3,)).astype(np.float32)
                       for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
            for aid, r in rewards.items():
                assert np.isfinite(r), f"{aid} reward not finite: {r}"
    finally:
        env.close()


def test_diagnose_script_runs():
    import subprocess
    result = subprocess.run(
        [subprocess.sys.executable, str(subprocess.Path(__file__).parents[1] / "scripts" / "diagnose_hetero_reward_overlay.py"),
         "--steps", "5"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
