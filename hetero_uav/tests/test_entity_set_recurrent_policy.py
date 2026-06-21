from __future__ import annotations

import torch
import numpy as np

from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.happo.happo_trainer import HAPPOReferenceTrainer


def _inputs(red_count: int, blue_count: int, entity_dim: int = 19):
    actor_tokens = torch.randn(red_count, red_count + blue_count, entity_dim)
    actor_mask = torch.ones(red_count, red_count + blue_count, dtype=torch.bool)
    critic_tokens = torch.randn(red_count + blue_count, entity_dim)
    critic_mask = torch.ones(red_count + blue_count, dtype=torch.bool)
    roles = torch.tensor([0] + [1] * (red_count - 1))
    return actor_tokens, actor_mask, critic_tokens, critic_mask, roles


def test_entity_recurrent_policy_supports_variable_token_counts():
    policy = HeteroEntityRecurrentPolicy(entity_dim=19, action_dim=3, rnn_hidden_size=64)
    parameter_shapes = {name: tuple(value.shape) for name, value in policy.state_dict().items()}

    for red_count, blue_count in ((3, 2), (5, 4), (7, 6)):
        actor_tokens, actor_mask, critic_tokens, critic_mask, roles = _inputs(red_count, blue_count)
        out = policy.act(
            actor_tokens,
            actor_mask,
            roles,
            critic_tokens,
            critic_mask,
            deterministic=True,
        )
        assert out["action"].shape == (red_count, 3)
        assert out["log_prob"].shape == (red_count,)
        assert out["rnn_hidden"].shape == (red_count, 64)
        assert out["value"].shape == (1,)
        assert torch.isfinite(out["log_prob"]).all()
        assert torch.isfinite(out["entropy"]).all()
        assert torch.isfinite(out["value"]).all()

    assert parameter_shapes == {name: tuple(value.shape) for name, value in policy.state_dict().items()}


def test_padding_tokens_do_not_change_deterministic_output():
    torch.manual_seed(7)
    policy = HeteroEntityRecurrentPolicy(entity_dim=19, action_dim=3, rnn_hidden_size=32).eval()
    actor_tokens, actor_mask, critic_tokens, critic_mask, roles = _inputs(3, 2)
    base = policy.act(actor_tokens, actor_mask, roles, critic_tokens, critic_mask, deterministic=True)

    padded_actor = torch.cat([actor_tokens, torch.randn(3, 4, 19) * 100], dim=1)
    padded_actor_mask = torch.cat([actor_mask, torch.zeros(3, 4, dtype=torch.bool)], dim=1)
    padded_critic = torch.cat([critic_tokens, torch.randn(4, 19) * 100], dim=0)
    padded_critic_mask = torch.cat([critic_mask, torch.zeros(4, dtype=torch.bool)], dim=0)
    padded = policy.act(
        padded_actor,
        padded_actor_mask,
        roles,
        padded_critic,
        padded_critic_mask,
        deterministic=True,
    )

    assert torch.allclose(base["action"], padded["action"], atol=1e-6)
    assert torch.allclose(base["value"], padded["value"], atol=1e-6)


def test_evaluate_actions_replays_pre_action_hidden_and_entity_inputs():
    policy = HeteroEntityRecurrentPolicy(entity_dim=19, action_dim=3, rnn_hidden_size=32)
    actor_tokens, actor_mask, critic_tokens, critic_mask, roles = _inputs(3, 2)
    pre_hidden = policy.init_hidden(3)
    sampled = policy.act(
        actor_tokens, actor_mask, roles, critic_tokens, critic_mask,
        rnn_hidden=pre_hidden, deterministic=True,
    )
    log_prob, entropy, value, mean, _roles, next_hidden = policy.evaluate_actions(
        actor_tokens.unsqueeze(0),
        actor_mask.unsqueeze(0),
        roles.unsqueeze(0),
        critic_tokens.unsqueeze(0),
        critic_mask.unsqueeze(0),
        sampled["action"].unsqueeze(0),
        rnn_hidden=pre_hidden.unsqueeze(0),
    )
    assert torch.allclose(log_prob[0], sampled["log_prob"], atol=1e-6)
    assert entropy.shape == (1, 3)
    assert value.shape == (1,)
    assert mean.shape == (1, 3, 3)
    assert next_hidden.shape == (1, 3, 32)


def test_entity_rollout_buffer_and_trainer_update_use_entity_critic():
    policy = HeteroEntityRecurrentPolicy(entity_dim=19, action_dim=3, rnn_hidden_size=32)
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)
    buffer = HAPPORolloutBuffer(
        2, 3, 0, 0, 3, [0, 1, 1], rnn_hidden_size=32,
        actor_token_count=5, critic_token_count=5, entity_dim=19,
    )
    for step in range(2):
        actor_tokens, actor_mask, critic_tokens, critic_mask, roles = _inputs(3, 2)
        pre_hidden = torch.zeros(3, 32)
        with torch.no_grad():
            out = policy.act(
                actor_tokens, actor_mask, roles, critic_tokens, critic_mask,
                rnn_hidden=pre_hidden,
            )
        buffer.store(
            None, None, out["action"].numpy(), out["log_prob"].numpy(),
            np.ones(3, np.float32), np.zeros(3, np.float32),
            float(out["value"].item()), np.ones(3, np.float32),
            next_value=0.0, rnn_hidden=pre_hidden.numpy(),
            actor_entity_tokens=actor_tokens.numpy(),
            actor_keep_mask=actor_mask.numpy(),
            critic_entity_tokens=critic_tokens.numpy(),
            critic_keep_mask=critic_mask.numpy(),
        )
    stats = trainer.update(buffer)
    assert torch.isfinite(torch.tensor(stats["actor_loss_mav"]))
    assert torch.isfinite(torch.tensor(stats["actor_loss_uav"]))
    assert torch.isfinite(torch.tensor(stats["critic_loss"]))
