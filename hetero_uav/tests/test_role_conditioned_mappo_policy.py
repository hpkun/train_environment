"""Test RoleConditionedMAPPOActorCritic. No env needed."""
from __future__ import annotations
import torch
import numpy as np

from algorithms.mappo.policy import RoleConditionedMAPPOActorCritic, MAPPOActorCritic


def _make_obs(num_agents, roles):
    """Build a fake 96-dim actor_obs. role: 0=mav 1=uav."""
    obs = torch.randn(num_agents, 96, dtype=torch.float32)
    obs[:, 7:11] = 0.0
    for i, r in enumerate(roles):
        obs[i, 7 + r] = 1.0  # ego_role one-hot
    return obs


def test_forward_shape():
    model = RoleConditionedMAPPOActorCritic()
    actor_obs = _make_obs(2, [0, 1])
    critic_state = torch.randn(1, 480)
    dist, value, action, log_prob, entropy = model(actor_obs, critic_state)
    assert action.shape == (2, 3)
    assert value.shape == (1,)
    assert log_prob.shape == (2,)


def test_evaluate_actions_shape():
    model = RoleConditionedMAPPOActorCritic()
    actor_obs = _make_obs(2, [0, 1])
    critic_state = torch.randn(1, 480)
    actions = torch.randn(2, 3)
    lp, ent, val = model.evaluate_actions(actor_obs, critic_state, actions)
    assert lp.shape == (2,)
    assert ent.shape == (2,)
    assert val.shape == (1,)


def test_roles_get_different_means():
    model = RoleConditionedMAPPOActorCritic()
    mav_obs = _make_obs(1, [0])
    uav_obs = _make_obs(1, [1])
    with torch.no_grad():
        mean_mav = model._role_conditioned_mean(mav_obs)
        mean_uav = model._role_conditioned_mean(uav_obs)
    assert not torch.allclose(mean_mav, mean_uav), "MAV and UAV heads should differ"


def test_dims_match_baseline():
    rc = RoleConditionedMAPPOActorCritic(96, 480, 3)
    bl = MAPPOActorCritic(96, 480, 3)
    assert rc.actor_obs_dim == bl.actor_obs_dim
    assert rc.critic_state_dim == bl.critic_state_dim
    assert rc.action_dim == bl.action_dim


def test_baseline_unchanged():
    """Verify MAPPOActorCritic still works as before."""
    model = MAPPOActorCritic(96, 480, 3)
    obs = torch.randn(2, 96)
    cs = torch.randn(1, 480)
    _, val, act, lp, ent = model(obs, cs)
    assert act.shape == (2, 3)
    assert val.shape == (1,)
