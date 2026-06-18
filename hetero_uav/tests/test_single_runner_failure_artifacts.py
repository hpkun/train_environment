from __future__ import annotations

import json

import pytest
import torch


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save(self, path):
        torch.save(self.state_dict(), path)


def test_failure_artifacts_save_finite_policy_and_structured_status(tmp_path):
    from scripts.train_happo_reference import _write_failure_artifacts

    state = {
        "total_steps": 430583,
        "iteration": 1682,
        "episode_id": 672,
        "output_dir": tmp_path,
        "meta": {"policy_arch": "brma_recurrent_masked"},
    }
    exc = ValueError("Non-finite actor_obs for active agent")

    _write_failure_artifacts(TinyPolicy(), state, exc)

    assert (tmp_path / "latest_failure/model.pt").exists()
    failure_meta = json.loads((tmp_path / "latest_failure/meta.json").read_text())
    assert failure_meta["status"] == "failed"
    assert failure_meta["total_env_steps_actual"] == 430583

    status = json.loads((tmp_path / "runner_status.json").read_text())
    assert status == {
        "status": "failed",
        "runner_completed_normally": False,
        "total_env_steps_actual": 430583,
        "iteration": 1682,
        "exception_type": "ValueError",
        "exception_message": "Non-finite actor_obs for active agent",
        "failed_step": 430583,
        "failed_episode_id": 672,
        "output_dir": str(tmp_path),
        "nan_detected": True,
        "nonfinite_detected": True,
        "failure_checkpoint_saved": True,
    }


def test_normal_runner_status_contains_completion_fields(tmp_path):
    from scripts.train_happo_reference import _write_runner_status

    _write_runner_status(
        tmp_path,
        status="normal",
        total_steps=500000,
        iteration=1954,
    )

    status = json.loads((tmp_path / "runner_status.json").read_text())
    assert status["status"] == "normal"
    assert status["runner_completed_normally"] is True
    assert status["total_env_steps_actual"] == 500000
    assert status["iteration"] == 1954
    assert status["exception_type"] == ""
    assert status["nonfinite_detected"] is False


def test_main_wrapper_preserves_failure_artifacts(monkeypatch, tmp_path):
    import scripts.train_happo_reference as runner

    runner._SINGLE_RUNNER_STATE.update({
        "policy": TinyPolicy(),
        "output_dir": tmp_path,
        "total_steps": 41,
        "iteration": 3,
        "episode_id": 2,
        "meta": {"policy_arch": "brma_recurrent_masked"},
    })
    monkeypatch.setattr(
        runner,
        "_run_training_main",
        lambda: (_ for _ in ()).throw(ValueError("Non-finite actor_obs")),
    )

    with pytest.raises(ValueError, match="Non-finite actor_obs"):
        runner.main()

    status = json.loads((tmp_path / "runner_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failed_step"] == 41
    assert status["failed_episode_id"] == 2
    assert (tmp_path / "latest_failure/model.pt").exists()
