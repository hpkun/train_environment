import torch

from algorithms.happo import TAMCategoricalHAPPOTrainer
from test_tam_categorical_happo_trainer import _buffer, _policy


def _shared_snapshot(policy):
    return [parameter.detach().clone() for parameter in policy.actor_shared_parameters()]


def _changed(before, after):
    return any(not torch.equal(left, right) for left, right in zip(before, after))


def test_default_full_mode_updates_shared_from_mav_role():
    policy = _policy()
    buffer = _buffer(policy, steps=6)
    buffer.role_ids[:] = 0
    before = _shared_snapshot(policy)
    trainer = TAMCategoricalHAPPOTrainer(policy, ppo_epochs=1)
    metrics = trainer.update(buffer)
    assert trainer.mav_shared_update_mode == "full"
    assert _changed(before, _shared_snapshot(policy))
    assert metrics["mav_shared_step_enabled"] == 1


def test_head_only_mav_does_not_step_shared_and_clears_shared_gradients():
    policy = _policy()
    buffer = _buffer(policy, steps=6)
    buffer.role_ids[:] = 0
    before = _shared_snapshot(policy)
    trainer = TAMCategoricalHAPPOTrainer(
        policy, ppo_epochs=1, mav_shared_update_mode="head_only"
    )
    metrics = trainer.update(buffer)
    assert not _changed(before, _shared_snapshot(policy))
    assert metrics["mav_shared_step_enabled"] == 0
    assert metrics["mav_shared_grad_norm_before_clear"] >= 0.0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in policy.actor_shared_parameters()
    )


def test_head_only_still_allows_uav_role_to_step_shared():
    policy = _policy()
    buffer = _buffer(policy, steps=6)
    buffer.role_ids[:] = 1
    before = _shared_snapshot(policy)
    trainer = TAMCategoricalHAPPOTrainer(
        policy, ppo_epochs=1, mav_shared_update_mode="head_only"
    )
    metrics = trainer.update(buffer)
    assert _changed(before, _shared_snapshot(policy))
    assert metrics["uav_shared_step_enabled"] == 1
    assert metrics["grad_norm_shared_from_uav"] > 0.0
