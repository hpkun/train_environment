from __future__ import annotations

import torch

import algorithms.happo as happo


def _policy():
    torch.manual_seed(7)
    return happo.TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19,
        actor_obs_dim=96,
        critic_state_dim=480,
        action_dim=4,
        action_levels=40,
        rnn_hidden_size=32,
        hidden_dim=32,
        num_attention_heads=4,
    )


def test_stochastic_act_returns_valid_long_indices_and_logits():
    policy = _policy()
    actor_obs = torch.randn(3, 96)
    critic_state = torch.randn(480)
    out = policy.act(actor_obs, roles=[0, 1, 1], critic_state=critic_state)

    assert out["action"].dtype == torch.long
    assert out["action"].shape == (3, 4)
    assert torch.all((0 <= out["action"]) & (out["action"] < 40))
    assert out["action_logits"].shape == (3, 4, 40)
    assert out["action_probs"].shape == (3, 4, 40)
    assert out["rnn_hidden"].shape == (3, 32)
    assert torch.isfinite(out["entropy"]).all()


def test_deterministic_action_is_per_axis_argmax():
    policy = _policy()
    out = policy.act(torch.randn(2, 96), roles=[0, 1], deterministic=True)
    assert torch.equal(out["action"], out["action_logits"].argmax(dim=-1))


def test_act_and_evaluate_actions_use_identical_categorical_log_prob():
    policy = _policy()
    obs = torch.randn(3, 96)
    critic = torch.randn(480)
    hidden = policy.init_hidden(3)
    out = policy.act(obs, roles=[0, 1, 1], critic_state=critic, rnn_hidden=hidden)
    log_prob, entropy, values, expected, roles = policy.evaluate_actions(
        obs, [0, 1, 1], critic, out["action"], rnn_hidden=hidden
    )
    torch.testing.assert_close(log_prob, out["log_prob"])
    torch.testing.assert_close(entropy, out["entropy"])
    assert values.shape == (1,)
    assert expected.shape == (3, 4)
    assert roles.shape == (3,)


def test_policy_has_attention_critic_and_no_gaussian_parameters():
    policy = _policy()
    assert any(isinstance(module, torch.nn.MultiheadAttention) for module in policy.critic.modules())
    names = dict(policy.named_parameters())
    assert not any("action_log_std" in name for name in names)
    assert not hasattr(policy, "action_log_std_mav")
    assert not hasattr(policy, "action_log_std_uav")
    values = policy.value(torch.randn(5, 480))
    assert values.shape == (5,)
    assert torch.isfinite(values).all()


def test_policy_checkpoint_round_trip(tmp_path):
    policy = _policy()
    path = tmp_path / "model.pt"
    policy.save(path)
    restored = _policy()
    restored.load(path, map_location="cpu")
    for expected, actual in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
