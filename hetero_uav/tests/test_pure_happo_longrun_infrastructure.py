from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from scripts.train_hetero_3v2_pure_happo_v1 import (
    _args,
    _last_record_step,
    _planned_rollout_count,
    _prepare_output_dir,
    _restore_buffer,
    _save_train_state,
    _training_hyperparameters,
    _validate_resume_state,
)


def _meta(output, args):
    return {
        "formal_contract": "hetero_3v2_pure_happo_v1",
        "reward_contract": {"version": "formal_role_reward_v2"},
        "algorithm_contract": "pure_happo_sequential_v2",
        "actor_obs_dim": 4,
        "critic_state_dim": 6,
        "action_dim": 3,
        "num_agents": 3,
        "seed": args.seed,
        "training_hyperparameters": _training_hyperparameters(args),
        "run_identity": str(output.resolve()),
    }


def test_output_directory_protection(tmp_path):
    fresh = tmp_path / "fresh"
    _prepare_output_dir(fresh, None)
    assert fresh.is_dir()
    (fresh / "train_log.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        _prepare_output_dir(fresh, None)
    state = fresh / "latest_resume" / "train_state.pt"
    state.parent.mkdir()
    state.write_bytes(b"state")
    _prepare_output_dir(fresh, state)
    with pytest.raises(ValueError, match="does not exist"):
        _prepare_output_dir(tmp_path / "different", state)


def test_full_rollout_plan_and_parameter_parsing():
    args = _args([
        "--total-env-steps", "200000", "--rollout-length", "256",
        "--actor-lr", "0.00025", "--critic-lr", "0.0005",
        "--ppo-epochs", "5", "--critic-epochs", "3",
        "--gamma", "0.98", "--gae-lambda", "0.94",
        "--entropy-coef", "0.02", "--clip-param", "0.15",
        "--max-grad-norm", "7.0",
    ])
    assert _planned_rollout_count(args.total_env_steps, args.rollout_length) == 782
    assert 782 * 256 == 200_192
    assert _training_hyperparameters(args) == {
        "actor_lr": 0.00025, "critic_lr": 0.0005,
        "ppo_epochs": 5, "critic_epochs": 3,
        "gamma": 0.98, "gae_lambda": 0.94,
        "entropy_coef": 0.02, "clip_param": 0.15,
        "max_grad_norm": 7.0, "rollout_length": 256,
    }


def test_train_state_roundtrip_and_contract_validation(tmp_path):
    args = _args(["--seed", "11", "--rollout-length", "8"])
    output = tmp_path / "run"
    output.mkdir()
    policy = PureHAPPOPolicy(4, 6, 3, 3)
    trainer = PureHAPPOTrainer(policy, seed=11)
    buffer = HAPPORolloutBuffer(8, 3, 4, 6, 3, [0, 1, 1])
    buffer.store(
        np.zeros((3, 4), np.float32), np.zeros(6, np.float32),
        np.zeros((3, 3), np.float32), np.zeros(3, np.float32),
        np.ones(3, np.float32), np.zeros(3, np.float32), 0.0,
        np.ones(3, np.float32), next_value=0.5,
        raw_actions=np.full((3, 3), 0.25, np.float32),
        terminated=0.0, episode_done=0.0)
    meta = _meta(output, args)
    path = output / "latest_resume" / "train_state.pt"
    _save_train_state(
        path, policy, trainer, meta=meta, total_steps=17, iteration=2,
        completed_episode_count=3, requested_checkpoint_step=16,
        buffer=buffer, rollout_stats={"reward": [1.0]},
        rollout_counts={"red_launches": 1}, rollout_completed=[{"outcome": "draw"}],
        episode_reset_seed=28, saved_eval_steps={16}, saved_resume_steps={16},
        resumed=False)
    state = torch.load(path, map_location="cpu", weights_only=False)
    _validate_resume_state(state, meta, output)
    restored = _restore_buffer(state["partial_rollout_buffer"])
    assert len(restored) == 1
    assert np.all(restored.raw_actions[0] == np.float32(0.25))
    assert len(state["actor_optimizer_state_dicts"]) == 3
    assert "critic_optimizer_state_dict" in state
    assert "trainer_rng_state" in state and "numpy_rng_state" in state
    assert state["total_env_steps"] == 17
    assert state["iteration"] == 2
    assert state["completed_episode_count"] == 3
    assert state["resume_semantics"] == "episode_boundary_exact_training_state"
    bad = dict(meta, seed=12)
    with pytest.raises(ValueError, match="seed"):
        _validate_resume_state(state, bad, output)
    with pytest.raises(ValueError, match="run_identity"):
        _validate_resume_state(state, meta, tmp_path / "other")


def test_resume_log_step_validation_helpers(tmp_path):
    train_log = tmp_path / "train_log.csv"
    with train_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["total_steps", "finite"])
        writer.writeheader()
        writer.writerow({"total_steps": 256, "finite": 1})
        writer.writerow({"total_steps": 513, "finite": 1})
    detail = tmp_path / "update_metrics.jsonl"
    detail.write_text(
        json.dumps({"total_steps": 256}) + "\n" + json.dumps({"total_steps": 513}) + "\n",
        encoding="utf-8")
    assert _last_record_step(train_log, csv_file=True) == 513
    assert _last_record_step(detail, csv_file=False) == 513
