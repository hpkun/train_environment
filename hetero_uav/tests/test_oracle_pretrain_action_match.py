from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_check_oracle_pretrain_action_match_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/check_oracle_pretrain_action_match.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dataset" in result.stdout
    assert "--checkpoint" in result.stdout


def test_check_oracle_pretrain_action_match_fake_inputs(tmp_path):
    from algorithms.happo import HAPPOReferencePolicy

    dataset = tmp_path / "fake_oracle.npz"
    checkpoint = tmp_path / "model.pt"
    output_json = tmp_path / "action_match.json"
    output_md = tmp_path / "action_match.md"

    rng = np.random.default_rng(123)
    actor_obs = rng.normal(size=(12, 96)).astype(np.float32)
    oracle_action = np.clip(rng.normal(size=(12, 3)), -1.0, 1.0).astype(np.float32)
    np.savez_compressed(dataset, actor_obs=actor_obs, oracle_action=oracle_action)
    torch.save(HAPPOReferencePolicy(96, 480).state_dict(), checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_oracle_pretrain_action_match.py",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--device",
            "cpu",
            "--max-samples",
            "12",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    for key in [
        "mse_mean_action_vs_oracle",
        "wrapped_mse_mean_action_vs_oracle",
        "mae_mean_action_vs_oracle",
        "wrapped_mae_mean_action_vs_oracle",
        "cosine_similarity",
        "policy_log_std",
        "checkpoint_has_uav_actor",
        "uav_actor_parameter_delta_after_load",
        "actor_obs_dim",
        "action_dim",
    ]:
        assert key in payload
    assert payload["actor_obs_dim"] == 96
    assert payload["action_dim"] == 3
