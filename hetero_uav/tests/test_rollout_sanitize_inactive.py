"""Test shared rollout safety helpers for inactive-agent sanitization."""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# sanitize_policy_inputs tests
# ---------------------------------------------------------------------------
def test_inactive_nan_obs_zeroed_by_sanitize():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs
    from algorithms.happo.happo_policy import HAPPOReferencePolicy

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    actor_obs[0, :] = np.nan
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    san = sanitize_policy_inputs(actor_obs, active, context={"env_idx": 0})
    assert san["diagnostics"]["inactive_count"] == 1
    assert not np.isnan(san["actor_obs"]).any()
    assert np.all(san["actor_obs"][0] == 0.0)
    assert np.any(san["actor_obs"][1] != 0.0)
    assert np.any(san["actor_obs"][2] != 0.0)

    policy = HAPPOReferencePolicy(96, 480)
    policy.eval()
    with torch.no_grad():
        out = policy.act(torch.as_tensor(san["actor_obs"]), roles=[0, 1, 1], deterministic=True)
    assert torch.isfinite(out["action"]).all()


def test_active_nan_obs_raises_with_specific_row_and_col():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    actor_obs[1, 5] = np.nan   # row=1, col=5
    active = np.ones(3, dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        sanitize_policy_inputs(actor_obs, active)
    msg = str(exc_info.value)
    assert "Non-finite actor_obs" in msg
    assert "row=1" in msg
    assert "cols=[5" in msg


def test_active_nan_obs_two_rows_reports_both():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    actor_obs[0, 3] = np.nan
    actor_obs[2, 7] = np.inf
    active = np.ones(3, dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        sanitize_policy_inputs(actor_obs, active)
    msg = str(exc_info.value)
    assert "row=0" in msg and "cols=[3" in msg
    assert "row=2" in msg and "cols=[7" in msg


def test_active_critic_nan_raises():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    active = np.ones(3, dtype=np.float32)
    critic = np.random.randn(480).astype(np.float32)
    critic[10] = np.nan

    with pytest.raises(ValueError, match="Non-finite critic_state"):
        sanitize_policy_inputs(actor_obs, active, critic_state=critic)


def test_inactive_actor_and_critic_chunk_nan_are_zeroed():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    actor_obs[0, :] = np.nan
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    critic = np.random.randn(480).astype(np.float32)
    critic[:96] = np.nan

    san = sanitize_policy_inputs(actor_obs, active, critic_state=critic)

    assert np.all(san["actor_obs"][0] == 0.0)
    assert np.all(san["critic_state"][:96] == 0.0)
    assert np.isfinite(san["critic_state"]).all()


def test_active_critic_chunk_nan_still_raises():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    critic = np.random.randn(480).astype(np.float32)
    critic[96 + 7] = np.nan

    with pytest.raises(ValueError, match="Non-finite critic_state for active agent"):
        sanitize_policy_inputs(actor_obs, active, critic_state=critic)


def test_active_rnn_hidden_nan_fail_fast():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    active = np.ones(3, dtype=np.float32)
    hidden = np.random.randn(3, 128).astype(np.float32)
    hidden[1, 42] = np.nan

    with pytest.raises(ValueError, match="Non-finite rnn_hidden for active agent"):
        sanitize_policy_inputs(actor_obs, active, rnn_hidden=hidden)


def test_inactive_rnn_hidden_nan_zeroed_not_raised():
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs = np.random.randn(3, 96).astype(np.float32)
    actor_obs[0, :] = np.nan  # dead MAV
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    hidden = np.random.randn(3, 128).astype(np.float32)
    hidden[0, :] = np.nan  # dead MAV has NaN hidden

    san = sanitize_policy_inputs(actor_obs, active, rnn_hidden=hidden)
    assert np.all(san["rnn_hidden"][0] == 0.0)
    assert np.any(san["rnn_hidden"][1] != 0.0)


# ---------------------------------------------------------------------------
# zero_inactive_actions / zero_inactive_hidden tests
# ---------------------------------------------------------------------------
def test_zero_inactive_actions_preserves_active():
    from algorithms.happo.rollout_safety import zero_inactive_actions

    actions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    result = zero_inactive_actions(actions, active)
    assert np.all(result[0] == 0.0)
    assert np.all(result[1] == actions[1])
    assert np.all(result[2] == actions[2])


def test_zero_inactive_hidden_zeros_inactive():
    from algorithms.happo.rollout_safety import zero_inactive_hidden

    hidden = np.random.randn(3, 128).astype(np.float32)
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    result = zero_inactive_hidden(hidden, active)
    assert np.all(result[0] == 0.0)
    assert np.all(result[1] == hidden[1])
    assert np.all(result[2] == hidden[2])


# ---------------------------------------------------------------------------
# End-to-end recurrent
# ---------------------------------------------------------------------------
def test_recurrent_policy_survives_dead_agent_loop():
    from algorithms.happo.rollout_safety import (
        sanitize_policy_inputs, zero_inactive_actions, zero_inactive_hidden)
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(
        entity_dim=19, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    policy.eval()
    rnn_hidden = np.zeros((3, 128), dtype=np.float32)

    for step in range(5):
        actor_obs = np.random.randn(3, 96).astype(np.float32)
        actor_obs[0, :] = np.nan
        active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        san = sanitize_policy_inputs(actor_obs, active, rnn_hidden=rnn_hidden)
        rnn_hidden = san["rnn_hidden"]
        with torch.no_grad():
            out = policy.act(
                torch.as_tensor(san["actor_obs"]), roles=[0, 1, 1],
                deterministic=True,
                rnn_hidden=torch.as_tensor(rnn_hidden),
            )
        assert torch.isfinite(out["action"]).all()
        actions = zero_inactive_actions(out["action"].cpu().numpy(), active)
        rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)
        assert np.all(actions[0] == 0.0)
        assert np.all(rnn_hidden[0] == 0.0)
