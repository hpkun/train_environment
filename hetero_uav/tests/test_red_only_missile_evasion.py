"""Tests for red-only scripted missile evasion policy."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from uav_env import make_env

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "uav_env" / "JSBSim" / "env.py"


def test_env_static_red_only_missile_evasion_guard():
    text = ENV_PATH.read_text(encoding="utf-8", errors="replace")
    start = text.index("Missile Evasion Script")
    missile_block = text[start : text.index("Layer 2", start)]
    assert "BOTH teams" not in missile_block
    assert "RED team only" in missile_block
    assert "rule-based opponent" in missile_block
    assert "if not is_blue" in missile_block
    assert "sim.check_missile_warning()" in missile_block


def test_red_only_missile_evasion_audit_script(tmp_path):
    out_json = tmp_path / "red_only_evasion.json"
    out_md = tmp_path / "red_only_evasion.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_red_only_missile_evasion.py",
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
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["red_scripted_evasion_enabled"] is True
    assert data["blue_scripted_evasion_enabled"] is False
    assert data["blue_gcas_still_enabled"] is True
    assert data["action_dim"] == 3
    assert data["blocking_issues"] == []


def test_red_only_missile_evasion_short_env_smoke():
    env = make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
        env_type="jsbsim_hetero",
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs
        for _ in range(3):
            actions = {
                aid: np.zeros(3, dtype=np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert not _obs_has_nan(obs)

        obs, info = env.reset(seed=1)
        rng = np.random.default_rng(1)
        for _ in range(3):
            actions = {
                aid: rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert not _obs_has_nan(obs)
    finally:
        env.close()


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False
