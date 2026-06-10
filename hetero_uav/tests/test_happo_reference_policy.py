import torch

from algorithms.happo import HAPPOReferencePolicy


def test_happo_reference_policy_forward_shapes():
    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    obs = torch.zeros(3, 96)
    obs[0, 7] = 1.0
    obs[1:, 8] = 1.0
    state = torch.zeros(480)
    out = policy.act(obs, roles=["mav", "uav", "uav"], critic_state=state)
    assert out["action"].shape == (3, 3)
    assert out["log_prob"].shape == (3,)
    assert out["entropy"].shape == (3,)
    assert out["value"].shape == (1,)
    assert out["mean"].shape == (3, 3)
    assert out["role_mask"].tolist() == [0, 1, 1]
    assert torch.all(out["action"] <= 1.0)
    assert torch.all(out["action"] >= -1.0)


def test_happo_reference_policy_separate_actor_outputs():
    torch.manual_seed(0)
    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    obs = torch.ones(2, 96) * 0.1
    out = policy.act(obs, roles=["mav", "uav"], deterministic=True)
    assert not torch.allclose(out["mean"][0], out["mean"][1])


def test_happo_reference_policy_supports_5v4_role_mapping():
    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    obs = torch.zeros(5, 96)
    roles = ["mav", "uav", "uav", "uav", "uav"]
    out = policy.act(obs, roles=roles, deterministic=True)
    assert out["action"].shape == (5, 3)
    assert out["role_mask"].tolist() == [0, 1, 1, 1, 1]


def test_happo_reference_policy_evaluate_actions_shapes():
    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    obs = torch.zeros(4, 3, 96)
    state = torch.zeros(4, 480)
    actions = torch.zeros(4, 3, 3)
    roles = torch.tensor([0, 1, 1]).view(1, 3).expand(4, 3)
    log_prob, entropy, value, mean, role_mask = policy.evaluate_actions(
        obs, roles, state, actions)
    assert log_prob.shape == (4, 3)
    assert entropy.shape == (4, 3)
    assert value.shape == (4,)
    assert mean.shape == (4, 3, 3)
    assert role_mask.shape == (4, 3)
