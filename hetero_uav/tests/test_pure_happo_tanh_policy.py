"""Tests for corrected tanh-squashed Pure-HAPPO action accounting."""
import numpy as np
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTanhPolicy, PureHAPPOTrainer
from scripts.train_happo_reference import _build_policy, _pure_happo_meta


def test_tanh_policy_action_bounds_and_finite_log_prob():
    policy = PureHAPPOTanhPolicy(num_agents=3)
    out = policy.act(torch.randn(3, 96), critic_state=torch.randn(480))
    assert out["action"].shape == (3, 3)
    assert out["raw_action"].shape == (3, 3)
    assert torch.all(out["action"] <= 1.0)
    assert torch.all(out["action"] >= -1.0)
    assert torch.isfinite(out["log_prob"]).all()


def test_tanh_policy_deterministic_action_is_tanh_mean():
    policy = PureHAPPOTanhPolicy(num_agents=3)
    out = policy.act(torch.randn(3, 96), critic_state=torch.randn(480), deterministic=True)
    assert torch.allclose(out["action"], torch.tanh(out["mean"]), atol=1e-6)


def test_tanh_policy_evaluate_actions_replays_act_log_prob():
    torch.manual_seed(7)
    policy = PureHAPPOTanhPolicy(num_agents=3)
    obs = torch.randn(3, 96)
    critic = torch.randn(480)
    out = policy.act(obs, critic_state=critic)
    lp, entropy, values, means = policy.evaluate_actions(
        obs.unsqueeze(0), critic.unsqueeze(0), out["action"].unsqueeze(0)
    )
    assert torch.max(torch.abs(lp.squeeze(0) - out["log_prob"])) < 1e-5
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()
    assert means.shape == (1, 3, 3)


def test_tanh_policy_evaluate_agent_actions_replays_single_agent():
    torch.manual_seed(9)
    policy = PureHAPPOTanhPolicy(num_agents=3)
    obs = torch.randn(3, 96)
    out = policy.act(obs, critic_state=torch.randn(480))
    for idx in range(3):
        lp, entropy, mean = policy.evaluate_agent_actions(idx, obs[idx:idx+1], out["action"][idx:idx+1])
        assert torch.max(torch.abs(lp.squeeze(0) - out["log_prob"][idx])) < 1e-5
        assert torch.isfinite(entropy).all()
        assert mean.shape == (1, 3)


def test_tanh_policy_large_std_has_consistent_replay():
    torch.manual_seed(11)
    policy = PureHAPPOTanhPolicy(num_agents=3, init_log_std=2.0)
    obs = torch.randn(3, 96)
    critic = torch.randn(480)
    out = policy.act(obs, critic_state=critic)
    lp, _, _, _ = policy.evaluate_actions(obs.unsqueeze(0), critic.unsqueeze(0), out["action"].unsqueeze(0))
    assert torch.isfinite(lp).all()
    assert torch.max(torch.abs(lp.squeeze(0) - out["log_prob"])) < 1e-5


def test_tanh_policy_trainer_update_no_nan():
    torch.manual_seed(13)
    np.random.seed(13)
    policy = PureHAPPOTanhPolicy(num_agents=3)
    buf = HAPPORolloutBuffer(12, 3, 96, 480, 3, [0, 1, 1])
    for t in range(12):
        obs = torch.randn(3, 96)
        critic = torch.randn(480)
        with torch.no_grad():
            out = policy.act(obs, critic_state=critic)
        buf.store(
            obs.numpy(), critic.numpy(), out["action"].numpy(), out["log_prob"].numpy(),
            np.random.randn(3).astype(np.float32) * 0.05,
            np.zeros(3, dtype=np.float32), out["value"].item(),
            np.ones(3, dtype=np.float32), env_id=t % 2,
        )
    metrics = PureHAPPOTrainer(policy, ppo_epochs=1, seed=13).update(buf)
    assert np.isfinite(metrics["actor_loss_mean"])
    assert np.isfinite(metrics["critic_loss"])


def test_pure_happo_is_tanh_squashed():
    """PureHAPPOPolicy now IS the tanh-squashed version. Legacy clamp is LegacyClampPureHAPPOPolicy."""
    policy = _build_policy("pure_happo", 96, 480, torch.device("cpu"), num_agents=3)
    assert policy.__class__.__name__ == "PureHAPPOPolicy"
    meta = _pure_happo_meta(policy)
    assert meta["policy_arch"] == "pure_happo"
    assert meta.get("bounded_action_distribution", "") == "tanh_squashed_gaussian"
    assert meta.get("logprob_correction", "") == "tanh_jacobian"


def test_legacy_clamp_policy_still_available():
    from algorithms.pure_happo.policy import LegacyClampPureHAPPOPolicy
    old = LegacyClampPureHAPPOPolicy(num_agents=3, actor_obs_dim=96, critic_state_dim=480)
    assert old.__class__.__name__ == "LegacyClampPureHAPPOPolicy"
