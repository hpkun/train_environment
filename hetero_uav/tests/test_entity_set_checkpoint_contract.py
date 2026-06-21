from __future__ import annotations

import json

import pytest
import torch

from scripts.eval_happo_reference import _build_policy_from_meta
from scripts.train_happo_reference import _build_policy, _entity_policy_meta


REQUIRED_FIELDS = {
    "feature_schema_version", "adapter_mode", "actor_obs_format",
    "critic_obs_format", "entity_dim", "role_dim", "role_vocab",
    "action_dim", "rnn_hidden_size", "policy_arch", "actor_arch",
    "critic_arch", "scale_support_mode", "padding_mode", "hidden_dim",
    "num_attention_heads",
    "policy_class", "critic_class", "observation_adapter",
}


def _meta():
    return {
        "feature_schema_version": "hetero_entity_set_v1",
        "adapter_mode": "hetero_entity_set",
        "actor_obs_format": "entity_tokens_keep_mask",
        "critic_obs_format": "global_entity_tokens_keep_mask",
        "entity_dim": 19,
        "role_dim": 4,
        "role_vocab": ["mav", "attack_uav", "scout_uav", "interceptor_uav"],
        "action_dim": 3,
        "rnn_hidden_size": 64,
        "hidden_dim": 96,
        "num_attention_heads": 3,
        "policy_arch": "hetero_entity_recurrent",
        "actor_arch": "entity_attention_grucell_role_heads",
        "critic_arch": "global_entity_attention_value",
        "scale_support_mode": "variable_token_count",
        "padding_mode": "keep_mask",
        "policy_class": "HeteroEntityRecurrentPolicy",
        "critic_class": "_GlobalEntityCritic",
        "observation_adapter": "HeteroEntitySetAdapter",
    }


def test_entity_checkpoint_meta_rebuilds_policy():
    meta = _meta()
    assert REQUIRED_FIELDS <= set(meta)
    policy = _build_policy_from_meta(meta, torch.device("cpu"))
    assert policy.action_dim == 3
    assert policy.rnn_hidden_size == 64
    assert policy.hidden_dim == 96
    assert policy.num_attention_heads == 3


def test_entity_checkpoint_rejects_action_or_schema_mismatch():
    bad_action = _meta()
    bad_action["action_dim"] = 4
    with pytest.raises(ValueError, match="action_dim"):
        _build_policy_from_meta(bad_action, torch.device("cpu"))

    bad_schema = _meta()
    bad_schema["feature_schema_version"] = "unknown"
    with pytest.raises(ValueError, match="feature_schema_version"):
        _build_policy_from_meta(bad_schema, torch.device("cpu"))


def test_training_policy_factory_builds_entity_recurrent():
    policy = _build_policy("hetero_entity_recurrent", 96, 480, torch.device("cpu"))
    assert policy.action_dim == 3
    assert policy.critic.__class__.__name__ != "Sequential"
    assert REQUIRED_FIELDS <= set(_entity_policy_meta(policy))


def test_training_factory_restores_entity_architecture_from_meta(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(_meta()), encoding="utf-8")
    policy = _build_policy(
        "hetero_entity_recurrent", 96, 480, torch.device("cpu"),
        init_checkpoint_meta=meta_path,
    )
    assert policy.hidden_dim == 96
    assert policy.num_attention_heads == 3


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("role_vocab", ["mav", "uav"]),
        ("actor_arch", "wrong_actor"),
        ("critic_arch", "wrong_critic"),
        ("actor_obs_format", "flat"),
        ("critic_obs_format", "flat"),
        ("role_dim", 3),
        ("policy_class", "WrongPolicy"),
        ("critic_class", "WrongCritic"),
    ],
)
def test_entity_checkpoint_schema_mismatch_is_rejected_by_train_and_eval(
    tmp_path, field, bad_value,
):
    meta = _meta()
    meta[field] = bad_value
    meta_path = tmp_path / f"{field}.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        _build_policy(
            "hetero_entity_recurrent", 96, 480, torch.device("cpu"),
            init_checkpoint_meta=meta_path,
        )
    with pytest.raises(ValueError, match=field):
        _build_policy_from_meta(meta, torch.device("cpu"))
