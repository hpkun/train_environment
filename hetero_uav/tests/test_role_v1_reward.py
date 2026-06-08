"""Tests for role_v1 heterogeneous reward overlay."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from uav_env import make_env

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
ROLE_V1_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_role_v1.yaml"


ROLE_KEYS = [
    "r_role_mav_survival",
    "r_role_mav_support",
    "r_role_mav_first_death",
    "r_role_mav_team_kill",
    "r_role_uav_angle",
    "r_role_uav_distance",
    "r_role_uav_kill",
    "r_role_uav_death",
]


def test_brma_legacy_has_no_role_v1_components():
    env = make_env(LEGACY_CFG, env_type="jsbsim_hetero", max_steps=5)
    try:
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert env.hetero_reward_mode == "brma_legacy"
        for aid in env.red_ids:
            for key in ROLE_KEYS:
                assert key not in info.get(aid, {})
    finally:
        env.close()


def test_role_v1_config_reset_step_components_present_and_finite():
    env = make_env(ROLE_V1_CFG, env_type="jsbsim_hetero", max_steps=5)
    try:
        assert env.hetero_reward_mode == "role_v1"
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid, reward in rewards.items():
            assert np.isfinite(reward), f"{aid} reward not finite: {reward}"
        for aid in env.red_ids:
            rcinfo = info.get(aid, {})
            for key in ROLE_KEYS:
                assert key in rcinfo, f"{aid} missing {key}"
        assert info["red_0"]["r_role_mav_survival"] > 0.0
    finally:
        env.close()


def test_role_v1_mode_does_not_replace_minimal_v1():
    minimal_cfg = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_minimal.yaml"
    env = make_env(minimal_cfg, env_type="jsbsim_hetero", max_steps=5)
    try:
        assert env.hetero_reward_mode == "minimal_v1"
    finally:
        env.close()


def test_diagnose_role_v1_reward_runs(tmp_path):
    out_json = tmp_path / "role_v1.json"
    out_md = tmp_path / "role_v1.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_role_v1_reward.py",
            "--steps",
            "3",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["config"].endswith("hetero_mav_shared_geo_3v2_reward_role_v1.yaml")
    assert data["hetero_reward_mode"] == "role_v1"
    assert data["nan_detected"] is False
    assert data["role_v1_component_keys"]
    text = out_md.read_text(encoding="utf-8")
    assert "role_v1" in text
    assert "MAV" in text
    assert "UAV" in text
