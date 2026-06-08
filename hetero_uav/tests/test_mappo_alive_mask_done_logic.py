"""Tests for MAPPO alive mask and team-level done handling."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from algorithms.mappo.trainer import _team_dones_from_repeated_agent_dones
from scripts.train_mappo_baseline import _build_red_alive_mask, _team_episode_done

ROOT = Path(__file__).resolve().parents[1]


def test_build_red_alive_mask_prefers_info_alive():
    info = {
        "red_0": {"alive": True},
        "red_1": {"alive": False},
        "red_2": {"alive": True},
    }
    mask = _build_red_alive_mask(info, env=None, red_ids=["red_0", "red_1", "red_2"])
    np.testing.assert_array_equal(mask, np.array([1.0, 0.0, 1.0], dtype=np.float32))


def test_team_episode_done_only_when_all_done():
    terminated = {"red_0": False, "red_1": True, "red_2": False}
    truncated = {"red_0": False, "red_1": False, "red_2": False}
    assert _team_episode_done(terminated, truncated) is False

    assert _team_episode_done(
        {"red_0": True, "red_1": True, "red_2": True},
        truncated,
    ) is True
    assert _team_episode_done(
        {"red_0": False, "red_1": False, "red_2": False},
        {"red_0": True, "red_1": True, "red_2": True},
    ) is True
    assert _team_episode_done(
        {"red_0": False, "red_1": False, "red_2": False},
        {"red_0": True, "red_1": False, "red_2": True},
    ) is False


def test_trainer_team_done_uses_repeated_team_done_column():
    dones = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    team_dones = _team_dones_from_repeated_agent_dones(dones)
    torch.testing.assert_close(team_dones, torch.tensor([0.0, 1.0]))


def test_main_runner_smoke_after_alive_mask_fix():
    out_dir = ROOT / "outputs" / "test_mappo_alive_mask_fix"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_main_mappo_experiment.py",
            "--total-env-steps",
            "64",
            "--rollout-length",
            "16",
            "--max-steps",
            "64",
            "--eval-episodes",
            "1",
            "--device",
            "cpu",
            "--opponent-policy",
            "rule_nearest",
            "--eval-during-training",
            "--eval-interval-steps",
            "32",
            "--train-eval-episodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    train_log = out_dir / "train_log.csv"
    eval_log = out_dir / "eval_log.csv"
    best_meta = out_dir / "best" / "meta.json"
    assert train_log.exists()
    assert eval_log.exists()
    assert best_meta.exists()

    rows = list(csv.DictReader(open(train_log, encoding="utf-8")))
    assert rows
    assert all(int(row["nan_detected"]) == 0 for row in rows)

    meta = json.loads((out_dir / "latest" / "meta.json").read_text(encoding="utf-8"))
    assert meta["actor_obs_dim"] == 96
    assert meta["critic_state_dim"] == 480
