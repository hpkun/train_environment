"""Test shared rollout safety helpers for inactive-agent sanitization."""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------
def _make_obs(num_agents: int = 3, dead_agent: int = 0) -> tuple:
    actor_obs = np.zeros((num_agents, 96), dtype=np.float32)
    for i in range(num_agents):
        actor_obs[i] = np.random.randn(96).astype(np.float32)
    active = np.ones(num_agents, dtype=np.float32)
    if dead_agent is not None and 0 <= dead_agent < num_agents:
        actor_obs[dead_agent] = np.nan
        active[dead_agent] = 0.0
    return actor_obs, active


# ---------------------------------------------------------------------------
# sanitize_policy_inputs tests
# ---------------------------------------------------------------------------
def test_inactive_nan_obs_zeroed_by_sanitize():
    """Inactive MAV with NaN obs → sanitized, policy.act does not crash."""
    from algorithms.happo.rollout_safety import sanitize_policy_inputs
    from algorithms.happo.happo_policy import HAPPOReferencePolicy

    actor_obs, active = _make_obs(3, dead_agent=0)
    san = sanitize_policy_inputs(actor_obs, active, context={"env_idx": 0})

    assert san["diagnostics"]["inactive_count"] == 1
    assert not np.isnan(san["actor_obs"]).any()
    assert np.all(san["actor_obs"][0] == 0.0)
    assert np.any(san["actor_obs"][1] != 0.0)

    policy = HAPPOReferencePolicy(96, 480)
    policy.eval()
    with torch.no_grad():
        out = policy.act(torch.as_tensor(san["actor_obs"]), roles=[0, 1, 1], deterministic=True)
    assert torch.isfinite(out["action"]).all()


def test_inactive_rnn_hidden_zeroed_by_sanitize():
    """Inactive hidden rows (even with random values) must be zeroed."""
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs, active = _make_obs(3, dead_agent=0)
    actor_obs[0] = 0.0  # assume already zeroed by prior sanitization
    h = np.random.randn(3, 128).astype(np.float32)
    h[0] = np.nan  # simulate stale hidden for dead MAV

    san = sanitize_policy_inputs(actor_obs, active, rnn_hidden=h)
    assert np.all(san["rnn_hidden"][0] == 0.0)


def test_active_nan_obs_raises_value_error():
    """Active agent obs with NaN must raise ValueError with row info."""
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs, active = _make_obs(3, dead_agent=None)  # all alive
    actor_obs[1, 5] = np.nan  # UAV 1 has NaN

    with pytest.raises(ValueError, match="Non-finite actor_obs for active agent"):
        sanitize_policy_inputs(actor_obs, active, context={"env_idx": 0})


def test_active_critic_nan_raises_value_error():
    """Active critic_state with NaN must raise ValueError."""
    from algorithms.happo.rollout_safety import sanitize_policy_inputs

    actor_obs, active = _make_obs(3, dead_agent=None)
    critic = np.random.randn(480).astype(np.float32)
    critic[10] = np.nan

    with pytest.raises(ValueError, match="Non-finite critic_state"):
        sanitize_policy_inputs(actor_obs, active, critic_state=critic)


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
# End-to-end: recurrent policy survives sanitized dead-agent forward
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
        actor_obs, active = _make_obs(3, dead_agent=0)
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
