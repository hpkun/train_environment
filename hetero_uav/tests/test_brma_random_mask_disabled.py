from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def _run_script(script: str, output_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / script),
            "--config",
            CONFIG,
            "--output-dir",
            output_dir,
            "--total-env-steps",
            "0",
            "--rollout-length",
            "1",
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--policy-arch",
            "brma_recurrent_masked",
            "--brma-random-scale-mask",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )


def _run_script_with_init_checkpoint(script: str, output_dir: str, checkpoint: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / script),
            "--config",
            CONFIG,
            "--output-dir",
            output_dir,
            "--total-env-steps",
            "0",
            "--rollout-length",
            "1",
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--policy-arch",
            "brma_recurrent_masked",
            "--init-checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_single_runner_rejects_brma_random_scale_mask():
    result = _run_script(
        "train_happo_reference.py",
        "outputs/_test_reject_random_mask_single",
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "random_scale_mask" in combined
    assert "old/new log_prob" in combined
    assert "mask replay" in combined


def test_parallel_runner_rejects_brma_random_scale_mask():
    result = _run_script(
        "train_happo_reference_parallel.py",
        "outputs/_test_reject_random_mask_parallel",
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "random_scale_mask" in combined
    assert "old/new log_prob" in combined
    assert "mask replay" in combined


def test_single_runner_rejects_unsafe_random_scale_mask_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "unsafe"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_bytes(b"not used before meta validation")
    (ckpt_dir / "meta.json").write_text(
        '{"policy_arch":"brma_recurrent_masked","random_scale_mask":true}',
        encoding="utf-8",
    )

    result = _run_script_with_init_checkpoint(
        "train_happo_reference.py",
        "outputs/_test_reject_random_mask_checkpoint_single",
        checkpoint,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "unsafe random_scale_mask checkpoint" in combined
    assert "diagnostic eval" in combined
    assert "mask replay" in combined


def test_parallel_runner_rejects_unsafe_random_scale_mask_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "unsafe"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_bytes(b"not used before meta validation")
    (ckpt_dir / "meta.json").write_text(
        '{"policy_arch":"brma_recurrent_masked","random_scale_mask":true}',
        encoding="utf-8",
    )

    result = _run_script_with_init_checkpoint(
        "train_happo_reference_parallel.py",
        "outputs/_test_reject_random_mask_checkpoint_parallel",
        checkpoint,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "unsafe random_scale_mask checkpoint" in combined
    assert "diagnostic eval" in combined
    assert "mask replay" in combined


def test_brma_recurrent_masked_without_random_mask_has_stable_log_prob_replay():
    from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy

    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=False,
        biased_mask=False,
    )
    policy.train()
    actor_obs = torch.zeros((4, 3, 96), dtype=torch.float32)
    roles = torch.tensor([[0, 1, 1]] * 4)
    critic = torch.zeros((4, 480), dtype=torch.float32)
    actions = torch.zeros((4, 3, 3), dtype=torch.float32)
    hidden = torch.zeros((4, 3, policy.rnn_hidden_size), dtype=torch.float32)

    first = policy.evaluate_actions(actor_obs, roles, critic, actions, rnn_hidden=hidden)[0]
    second = policy.evaluate_actions(actor_obs, roles, critic, actions, rnn_hidden=hidden)[0]

    assert torch.allclose(first, second)
