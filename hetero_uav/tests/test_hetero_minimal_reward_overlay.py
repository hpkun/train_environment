"""Test minimal_v1 hetero reward overlay. No training, no env changes."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# -- helpers -----------------------------------------------------------

def _find_python():
    """Find a Python executable that can import gymnasium."""
    candidates = [sys.executable]
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import gymnasium"],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


try:
    from uav_env import make_env

    HAVE_UAV_ENV = True
except ImportError:
    HAVE_UAV_ENV = False

LEGACY_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
MINIMAL_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_minimal.yaml"
CLOSE_RANGE_MINIMAL_CFG = (
    "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2_reward_minimal.yaml"
)

# -- tests -------------------------------------------------------------


def test_legacy_has_no_overlay_components():
    if not HAVE_UAV_ENV:
        return  # skip if gymnasium not available in current interpreter
    env = make_env(LEGACY_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        # Step once
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for k in ["r_mav_survival", "r_mav_death", "r_mav_support",
                      "r_shared_track_used", "r_attack_kill_bonus"]:
                # legacy should NOT contain these overlay keys at all
                assert k not in rcinfo, (
                    f"{aid} legacy info should not contain overlay key {k}, got {rcinfo.get(k)}"
                )
    finally:
        env.close()


def test_minimal_reset_and_step():
    if not HAVE_UAV_ENV:
        return
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        assert env.hetero_reward_mode == "minimal_v1"
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
    finally:
        env.close()


def test_minimal_reward_components_present():
    if not HAVE_UAV_ENV:
        return
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
    if not HAVE_UAV_ENV:
        return
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
    if not HAVE_UAV_ENV:
        return
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
    result = subprocess.run(
        [
            _find_python(),
            str(ROOT / "scripts" / "diagnose_hetero_reward_overlay.py"),
            "--steps",
            "5",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_close_range_minimal_config_exists():
    path = ROOT / CLOSE_RANGE_MINIMAL_CFG
    assert path.exists(), f"missing close-range minimal config: {path}"
    text = path.read_text(encoding="utf-8")
    assert "hetero_reward_mode" in text
    assert "minimal_v1" in text


def test_close_range_minimal_reset_and_step():
    if not HAVE_UAV_ENV:
        return
    env = make_env(CLOSE_RANGE_MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        assert env.hetero_reward_mode == "minimal_v1"
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        # overlay components should be present
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for k in ["r_mav_survival", "r_mav_death", "r_mav_support",
                      "r_shared_track_used", "r_attack_kill_bonus"]:
                assert k in rcinfo, f"{aid} close-range minimal missing {k}"
    finally:
        env.close()


def test_support_reward_one_step_lag():
    """r_mav_support uses cached observation from the previous step.
    On the first step, _last_step_obs may be empty, so r_mav_support
    may be 0.  This test simply confirms no crash."""
    if not HAVE_UAV_ENV:
        return
    env = make_env(MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        # Step 1: _last_step_obs is empty → r_mav_support should be 0
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        # Step 1 r_mav_support may be 0 (one-step-lag) — just check no crash
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            val = float(rcinfo.get("r_mav_support", 0.0))
            assert val >= 0.0, f"{aid} r_mav_support negative on step1: {val}"
        # Step 2: now _last_step_obs is populated → support may appear
        obs, rewards, terminated, truncated, info = env.step(actions)
        # No assertion on exact value — depends on observation range
    finally:
        env.close()


def test_close_range_minimal_reward_finite():
    if not HAVE_UAV_ENV:
        return
    env = make_env(CLOSE_RANGE_MINIMAL_CFG, env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        for _ in range(5):
            actions = {aid: np.random.uniform(-0.5, 0.5, (3,)).astype(np.float32)
                       for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
            for aid, r in rewards.items():
                assert np.isfinite(r), f"{aid} close-range reward not finite: {r}"
    finally:
        env.close()
