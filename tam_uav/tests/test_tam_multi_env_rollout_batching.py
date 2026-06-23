from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from scripts.train_tam_happo_direct import _assert_env_specs_match
from uav_env import make_env


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_two_env_specs_match_for_rollout_batching():
    envs = [
        make_env(
            CONFIG,
            env_type="jsbsim_hetero",
            hetero_reward_mode="tam_paper_reward_v1",
            max_steps=200,
        )
        for _ in range(2)
    ]
    try:
        _assert_env_specs_match(envs[0], envs)
        assert repr(envs[0].action_space) == repr(envs[1].action_space)
        assert repr(envs[0].observation_space) == repr(envs[1].observation_space)
    finally:
        for env in envs:
            env.close()


def test_rollout_buffer_splits_recurrent_sequences_by_env_id():
    buffer = HAPPORolloutBuffer(
        4, 3, 8, 12, 4, [0, 1, 1],
        rnn_hidden_size=5, action_dtype=np.int64, num_envs=2,
    )
    buffer.set_rnn_hidden_initial(0, np.full((3, 5), 0.25, dtype=np.float32))
    buffer.set_rnn_hidden_initial(1, np.full((3, 5), 0.75, dtype=np.float32))
    for step in range(2):
        for env_id in range(2):
            buffer.store(
                np.full((3, 8), env_id, dtype=np.float32),
                np.full(12, step, dtype=np.float32),
                np.zeros((3, 4), dtype=np.int64),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0.0,
                np.ones(3, dtype=np.float32),
                next_value=0.0,
                env_id=env_id,
                env_step_index=step,
                rnn_hidden=np.zeros((3, 5), dtype=np.float32),
                episode_start_masks=np.zeros(3, dtype=np.float32),
            )

    sequences = buffer.get_sequences(torch.device("cpu"))
    assert len(sequences) == 2
    assert [int(seq["env_ids"][0].item()) for seq in sequences] == [0, 1]
    assert [seq["actor_obs"].shape[0] for seq in sequences] == [2, 2]
    torch.testing.assert_close(sequences[0]["env_step_index"], torch.tensor([0, 1]))
    torch.testing.assert_close(sequences[1]["env_step_index"], torch.tensor([0, 1]))
    torch.testing.assert_close(
        sequences[0]["rnn_hidden_initial"],
        torch.full((3, 5), 0.25),
    )
    torch.testing.assert_close(
        sequences[1]["rnn_hidden_initial"],
        torch.full((3, 5), 0.75),
    )


def test_train_script_accepts_num_envs_two_and_writes_batching_meta(tmp_path):
    output_dir = tmp_path / "numenv2_train"
    cmd = [
        sys.executable,
        "-u",
        "scripts/train_tam_happo_direct.py",
        "--config", CONFIG,
        "--output-dir", str(output_dir),
        "--total-env-steps", "512",
        "--rollout-length", "64",
        "--num-envs", "2",
        "--max-steps", "200",
        "--device", "cpu",
        "--policy-arch", "tam_categorical_recurrent",
        "--opponent-policy", "tam_direct_fsm",
        "--reward-mode", "tam_paper_reward_v1",
        "--tam-paper-mode",
        "--happo-update-granularity", "agent",
        "--advantage-mode", "per_agent_reward",
        "--actor-lr", "0.0005",
        "--critic-lr", "0.0005",
        "--entropy-coef", "0.01",
        "--clip-param", "0.2",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--max-grad-norm", "10.0",
        "--ppo-epochs", "2",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    meta = json.loads((output_dir / "latest" / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_envs"] == 2
    assert meta["rollout_length"] == 64
    assert meta["transitions_per_rollout"] == 128
    assert meta["multi_env_rollout_mode"] == "serial_env_batching"
