"""Test role_v1 reward. No training."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    candidates = [sys.executable]
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    for py in candidates:
        try:
            r = subprocess.run([py, "-c", "import gymnasium"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


PYTHON = _find_python()


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


try:
    from uav_env import make_env
    import numpy as np
    HAVE_ENV = True
except ImportError:
    HAVE_ENV = False


def test_role_v1_config_loads():
    if not HAVE_ENV:
        return
    env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_role_v1.yaml",
                   env_type="jsbsim_hetero", max_steps=10)
    try:
        assert env.hetero_reward_mode == "role_v1"
        obs, info = env.reset(seed=0)
        acts = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(acts)
    finally:
        env.close()


def test_legacy_no_role_keys():
    if not HAVE_ENV:
        return
    env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
                   env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        acts = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(acts)
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for k in ["r_role_mav_survival", "r_role_uav_kill_bonus"]:
                assert k not in rcinfo, f"{aid} legacy should not have {k}"
    finally:
        env.close()


def test_role_v1_has_role_keys():
    if not HAVE_ENV:
        return
    env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_role_v1.yaml",
                   env_type="jsbsim_hetero", max_steps=10)
    try:
        obs, info = env.reset(seed=0)
        acts = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(acts)
        mav_keys = ["r_role_mav_survival", "r_role_mav_death", "r_role_mav_support",
                    "r_role_mav_team_contribution"]
        uav_keys = ["r_role_uav_attack_window", "r_role_uav_kill_bonus",
                    "r_role_uav_death_penalty", "r_role_uav_missile_warning"]
        r0 = info.get("red_0", {})
        r1 = info.get("red_1", {})
        for k in mav_keys:
            assert k in r0, f"red_0 missing {k}"
        for k in uav_keys:
            assert k in r1, f"red_1 missing {k}"
    finally:
        env.close()


def test_diagnose_script_runs():
    result = subprocess.run(
        [PYTHON, "scripts/diagnose_role_v1_reward.py", "--steps", "5",
         "--output-json", "outputs/test_environment_audit/role_v1_diag.json",
         "--output-md", "outputs/test_environment_audit/role_v1_diag.md"],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (ROOT / "outputs/test_environment_audit/role_v1_diag.json").exists()
    assert (ROOT / "outputs/test_environment_audit/role_v1_diag.md").exists()
    data = json.loads((ROOT / "outputs/test_environment_audit/role_v1_diag.json").read_text(encoding="utf-8"))
    assert data["nan"] is False
    assert data["legacy_has_role_keys"] is False
    assert data["role_has_role_keys"] is True
