from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_pretrain_uav_actor_from_oracle_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/pretrain_uav_actor_from_oracle.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dataset" in result.stdout
    assert "--epochs" in result.stdout


def test_pretrain_uav_actor_from_oracle_fake_dataset_one_epoch():
    from algorithms.happo import HAPPOReferencePolicy

    out_dir = ROOT / "outputs/test_oracle_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = out_dir / "fake_oracle.npz"
    init_ckpt = out_dir / "init.pt"
    model_out = out_dir / "model.pt"
    meta_out = out_dir / "meta.json"

    rng = np.random.default_rng(0)
    np.savez_compressed(
        dataset,
        actor_obs=rng.normal(size=(16, 96)).astype(np.float32),
        oracle_action=np.clip(rng.normal(size=(16, 3)), -1, 1).astype(np.float32),
        role_id=np.ones(16, dtype=np.int64),
        agent_id=np.array(["red_1"] * 16),
        episode_id=np.zeros(16, dtype=np.int64),
        step=np.arange(16, dtype=np.int64),
        alive_mask=np.ones(16, dtype=np.float32),
        nearest_enemy_distance=np.ones(16, dtype=np.float32),
        launch_range_flag=np.zeros(16, dtype=np.float32),
        launch_angle_flag=np.zeros(16, dtype=np.float32),
        launch_envelope_flag=np.zeros(16, dtype=np.float32),
        missile_fired_this_step=np.zeros(16, dtype=np.float32),
        missile_hit_this_step=np.zeros(16, dtype=np.float32),
    )
    policy = HAPPOReferencePolicy(96, 480)
    torch.save(policy.state_dict(), init_ckpt)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/pretrain_uav_actor_from_oracle.py",
            "--dataset",
            str(dataset.relative_to(ROOT)),
            "--init-checkpoint",
            str(init_ckpt.relative_to(ROOT)),
            "--output-checkpoint",
            str(model_out.relative_to(ROOT)),
            "--output-meta",
            str(meta_out.relative_to(ROOT)),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert model_out.exists()
    assert meta_out.exists()
    meta = json.loads(meta_out.read_text(encoding="utf-8"))
    assert meta["pretrained_from_oracle"] is True
    assert meta["frozen_mav_actor"] is True
    assert meta["frozen_critic"] is True
    assert meta["num_samples"] == 16

