"""Test RoleConditionedMAPPOActorCritic. No env needed."""
from __future__ import annotations
import torch
import numpy as np
from algorithms.mappo.policy import RoleConditionedMAPPOActorCritic, MAPPOActorCritic


def _make_obs(num_agents, role_val=0):
    """Build a fake 96-dim actor_obs with deterministic features, role changed."""
    torch.manual_seed(42)  # fixed seed for reproducibility
    base = torch.randn(num_agents, 96, dtype=torch.float32)
    base[:, 7:11] = 0.0
    if role_val == 0:
        base[:, 7] = 1.0
    elif role_val == 1:
        base[:, 8] = 1.0
    return base


def test_forward_shape():
    model = RoleConditionedMAPPOActorCritic()
    obs = _make_obs(2)
    critic_state = torch.randn(1, 480)
    dist, value, action, log_prob, entropy = model(obs, critic_state)
    assert action.shape == (2, 3)
    assert value.shape == (1,)
    assert log_prob.shape == (2,)


def test_evaluate_actions_shape():
    model = RoleConditionedMAPPOActorCritic()
    obs = _make_obs(2)
    critic_state = torch.randn(1, 480)
    actions = torch.randn(2, 3)
    lp, ent, val = model.evaluate_actions(obs, critic_state, actions)
    assert lp.shape == (2,)
    assert ent.shape == (2,)
    assert val.shape == (1,)


def test_roles_get_different_means_same_obs():
    """Same base observation, only role one-hot differs."""
    model = RoleConditionedMAPPOActorCritic()
    mav_obs = _make_obs(1, role_val=0)   # [1,0,0,0]
    uav_obs = _make_obs(1, role_val=1)   # [0,1,0,0]
    # Verify only role channels differ
    assert torch.allclose(mav_obs[:, :7], uav_obs[:, :7])
    assert torch.allclose(mav_obs[:, 11:], uav_obs[:, 11:])
    assert not torch.allclose(mav_obs[:, 7:11], uav_obs[:, 7:11])
    with torch.no_grad():
        mean_mav = model._role_conditioned_mean(mav_obs)
        mean_uav = model._role_conditioned_mean(uav_obs)
    assert not torch.allclose(mean_mav, mean_uav), "MAV and UAV heads should differ"


def test_role_layout_params_match_v2():
    model = RoleConditionedMAPPOActorCritic(
        role_start=7, role_dim=4, mav_role_index=0, obs_layout="v2")
    obs = _make_obs(1, role_val=1)
    with torch.no_grad():
        mean = model._role_conditioned_mean(obs)
    assert mean.shape == (1, 3)


def test_invalid_role_layout_rejected():
    try:
        RoleConditionedMAPPOActorCritic(obs_layout="v3")
        assert False, "should have raised"
    except ValueError:
        pass


def test_invalid_obs_dim_rejected():
    try:
        RoleConditionedMAPPOActorCritic(actor_obs_dim=10, role_start=7, role_dim=4)
        assert False, "should have raised"
    except ValueError:
        pass


def test_mav_batch_updates_mav_head_not_uav_head():
    model = RoleConditionedMAPPOActorCritic()
    mav_obs = _make_obs(3, role_val=0)
    mav_obs.requires_grad_(True)
    mean = model._role_conditioned_mean(mav_obs)
    loss = mean.sum()
    loss.backward()
    # mav_head should have grads
    for p in model.mav_head.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "mav_head grad is zero"
    # uav_head should have no grads (was not used)
    for p in model.uav_head.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, "uav_head should have no grad"
    # shared_encoder should have grads
    for p in model.shared_encoder.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "shared_encoder grad is zero"


def test_uav_batch_updates_uav_head_not_mav_head():
    model = RoleConditionedMAPPOActorCritic()
    uav_obs = _make_obs(3, role_val=1)
    uav_obs.requires_grad_(True)
    mean = model._role_conditioned_mean(uav_obs)
    loss = mean.sum()
    loss.backward()
    for p in model.uav_head.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "uav_head grad is zero"
    for p in model.mav_head.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, "mav_head should have no grad"
    for p in model.shared_encoder.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "shared_encoder grad is zero"


def test_dims_match_baseline():
    rc = RoleConditionedMAPPOActorCritic(96, 480, 3)
    bl = MAPPOActorCritic(96, 480, 3)
    assert rc.actor_obs_dim == bl.actor_obs_dim
    assert rc.critic_state_dim == bl.critic_state_dim
    assert rc.action_dim == bl.action_dim


def test_baseline_unchanged():
    model = MAPPOActorCritic(96, 480, 3)
    obs = torch.randn(2, 96)
    cs = torch.randn(1, 480)
    _, val, act, lp, ent = model(obs, cs)
    assert act.shape == (2, 3)
    assert val.shape == (1,)
