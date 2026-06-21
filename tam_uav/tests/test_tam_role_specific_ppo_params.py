from types import SimpleNamespace
from numbers import Real

import numpy as np
import torch

from algorithms.happo import TAMCategoricalHAPPOTrainer
from scripts.train_tam_happo_direct import resolve_tam_update_params
from test_tam_categorical_happo_trainer import _buffer, _policy


def _args(preset="default", **overrides):
    values = dict(
        tam_update_preset=preset,
        actor_lr=2e-4,
        entropy_coef=0.02,
        clip_param=0.2,
        mav_actor_lr_scale=None,
        uav_actor_lr_scale=None,
        mav_entropy_coef=None,
        uav_entropy_coef=None,
        mav_clip_param=None,
        uav_clip_param=None,
        mav_target_kl=None,
        uav_target_kl=None,
        role_kl_early_stop=None,
        mav_shared_update_mode="full",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_preset_preserves_legacy_values():
    args = resolve_tam_update_params(_args())
    assert args.mav_actor_lr_scale == args.uav_actor_lr_scale == 1.0
    assert args.mav_entropy_coef == args.uav_entropy_coef == args.entropy_coef
    assert args.mav_clip_param == args.uav_clip_param == args.clip_param
    assert args.mav_target_kl == args.uav_target_kl == 0.0
    assert args.role_kl_early_stop is False
    assert args.mav_shared_update_mode == "full"


def test_mav_conservative_preset_and_explicit_override_priority():
    args = resolve_tam_update_params(_args(
        "mav_conservative", mav_entropy_coef=0.004, mav_clip_param=0.08
    ))
    assert args.mav_actor_lr_scale == 0.25
    assert args.uav_actor_lr_scale == 1.0
    assert args.mav_entropy_coef == 0.004
    assert args.uav_entropy_coef == args.entropy_coef
    assert args.mav_clip_param == 0.08
    assert args.uav_clip_param == args.clip_param
    assert args.mav_target_kl == 0.015
    assert args.uav_target_kl == 0.04
    assert args.role_kl_early_stop is True


def test_role_lr_scales_only_change_role_heads_and_not_shared_optimizer():
    trainer = TAMCategoricalHAPPOTrainer(
        _policy(), actor_lr=2e-4,
        mav_actor_lr_scale=0.25, uav_actor_lr_scale=0.5,
    )
    assert trainer.shared_actor_opt.param_groups[0]["lr"] == 2e-4
    assert trainer.mav_opt.param_groups[0]["lr"] == 5e-5
    assert trainer.uav_opt.param_groups[0]["lr"] == 1e-4


def test_role_specific_entropy_clip_and_kl_early_stop_are_finite():
    policy = _policy()
    trainer = TAMCategoricalHAPPOTrainer(
        policy, ppo_epochs=4,
        mav_entropy_coef=0.003, uav_entropy_coef=0.02,
        mav_clip_param=0.1, uav_clip_param=0.2,
        mav_target_kl=1e-12, uav_target_kl=0.0,
        role_kl_early_stop=True,
    )
    metrics = trainer.update(_buffer(policy, steps=8))
    assert trainer.role_entropy_coef == {0: 0.003, 1: 0.02}
    assert trainer.role_clip_param == {0: 0.1, 1: 0.2}
    assert metrics["mav_kl_early_stop_count"] >= 1
    assert metrics["mav_update_skipped_by_kl"] >= 1
    assert metrics["uav_update_skipped_by_kl"] == 0
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, Real))
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
