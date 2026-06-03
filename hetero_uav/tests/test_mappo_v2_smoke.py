"""V2 MAPPO smoke tests: 96-dim actor, 480-dim critic, no HAPPO/attention."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_baseline.py"


def test_train_accepts_v2():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=10)
    assert "--obs-adapter-version" in result.stdout


def test_v2_env_observation_mode():
    from uav_env import make_env
    env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
                   env_type="jsbsim_hetero", max_steps=5)
    try:
        assert env.observation_mode == "mav_shared_geo"
    finally:
        env.close()


def test_v2_adapter_dims():
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    a = HeteroObsAdapterV2()
    assert a.flat_actor_obs_dim == 96
    assert a.critic_state_dim == 480


def test_v2_train_smoke():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
         "--obs-adapter-version", "v2",
         "--iterations", "1", "--rollout-length", "8",
         "--max-steps", "16",
         "--opponent-policy", "rule_nearest",
         "--output-dir", "outputs/test_v2_smoke",
         "--log-csv", "outputs/test_v2_smoke/train_log.csv",
         "--device", "cpu", "--debug"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "Saved" in result.stdout


def test_v2_model_saved():
    model_path = "outputs/test_v2_smoke/latest/model.pt"
    assert Path(model_path).exists()
    meta_path = "outputs/test_v2_smoke/latest/meta.json"
    assert Path(meta_path).exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["obs_adapter_version"] == "v2"


def test_v2_eval_smoke():
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--model", "outputs/test_v2_smoke/latest/model.pt",
         "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
         "--obs-adapter-version", "v2",
         "--episodes", "1", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_v2_diagnose_trainability():
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "diagnose_mappo_v2_trainability.py"),
         "--iterations", "2", "--rollout-length", "8", "--max-steps", "16",
         "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "episodes_completed" in result.stdout or "no completed episode" in result.stdout


def test_v2_zero_shot_smoke():
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "diagnose_mappo_v2_zero_shot_smoke.py"),
         "--model", "outputs/mappo_v2_trainability/latest/model.pt",
         "--episodes", "2", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "episodes: 2" in result.stdout


def test_v2_zero_shot_eval_episodes_two():
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
         "--model", "outputs/mappo_v2_trainability/latest/model.pt",
         "--obs-adapter-version", "v2",
         "--episodes", "2", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=180)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "episodes: 2" in result.stdout


def test_no_nan_v2():
    log_csv = "outputs/mappo_v2_trainability/train_log.csv"
    if Path(log_csv).exists():
        import csv
        with open(log_csv) as f:
            for row in csv.DictReader(f):
                assert int(row["nan_detected"]) == 0


def test_v1_still_works():
    """V1 path must still pass."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--config", "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
         "--obs-adapter-version", "v1",
         "--iterations", "1", "--rollout-length", "8",
         "--opponent-policy", "rule_nearest",
         "--output-dir", "outputs/test_v1_still_works",
         "--log-csv", "outputs/test_v1_still_works/train_log.csv",
         "--device", "cpu", "--debug"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "Saved" in result.stdout
