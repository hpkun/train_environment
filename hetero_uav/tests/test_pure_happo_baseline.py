"""Tests for paper-aligned pure HAPPO baseline."""
import numpy as np
import pytest
import torch

from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.trainer import _compute_grouped_gae


class TestPureHAPPOPolicy:
    def test_independent_actors(self):
        policy = PureHAPPOPolicy(num_agents=3)
        assert len(policy.actors) == 3
        assert len(policy.action_log_stds) == 3
        assert policy.actors[0] is not policy.actors[1]
        assert policy.action_log_stds[0] is not policy.action_log_stds[1]

    def test_act_shapes(self):
        policy = PureHAPPOPolicy(num_agents=3)
        obs = torch.randn(3, 96)
        out = policy.act(obs, critic_state=torch.randn(480))
        assert out["action"].shape == (3, 3)

    def test_agent_count_guard(self):
        policy = PureHAPPOPolicy(num_agents=3)
        with pytest.raises(ValueError, match="got 5"):
            policy.act(torch.randn(5, 96))


class TestGroupedGAE:
    def test_interleaved_env_ids(self):
        rewards     = torch.tensor([1.0, 2.0, 1.0, 2.0])
        values      = torch.tensor([0.0, 0.0, 0.0, 0.0])
        next_values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        dones       = torch.tensor([0.0, 0.0, 1.0, 1.0])
        env_ids     = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        adv, ret = _compute_grouped_gae(rewards, values, next_values, dones, env_ids, 1.0, 1.0)
        expected = torch.tensor([2.0, 4.0, 1.0, 2.0])
        assert torch.allclose(adv, expected), f"got {adv}"
        assert torch.allclose(ret, expected)

    def test_bootstrap_next_value(self):
        rewards     = torch.tensor([0.0])
        values      = torch.tensor([0.0])
        next_values = torch.tensor([5.0])
        dones       = torch.tensor([0.0])
        env_ids     = torch.tensor([0], dtype=torch.long)
        adv, ret = _compute_grouped_gae(rewards, values, next_values, dones, env_ids, 1.0, 1.0)
        assert torch.allclose(adv, torch.tensor([5.0]))
        assert torch.allclose(ret, torch.tensor([5.0]))


class TestPureHAPPOTrainer:
    def _fake_buffer(self, T=16, N=3):
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        policy = PureHAPPOPolicy(num_agents=N, actor_obs_dim=96, critic_state_dim=480)
        buf = HAPPORolloutBuffer(T, N, 96, 480, 3, [0, 1, 1])
        for t in range(T):
            obs = torch.randn(N, 96); crit = torch.randn(480)
            with torch.no_grad():
                out = policy.act(obs, critic_state=crit)
            buf.store(obs.numpy(), crit.numpy(), out["action"].numpy(),
                      out["log_prob"].numpy(),
                      np.random.randn(N).astype(np.float32) * 0.1,
                      np.zeros(N, dtype=np.float32), out["value"].item(),
                      np.ones(N, dtype=np.float32), env_id=t % 2)
        return buf

    def test_update_no_nan(self):
        buf = self._fake_buffer()
        trainer = PureHAPPOTrainer(PureHAPPOPolicy(num_agents=3), ppo_epochs=2, seed=42)
        m = trainer.update(buf)
        assert np.isfinite(m["actor_loss_mean"])
        assert np.isfinite(m["critic_loss"])

    def test_random_order(self):
        buf = self._fake_buffer(T=8)
        trainer = PureHAPPOTrainer(PureHAPPOPolicy(num_agents=3), ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        assert sorted(m["last_update_order"]) == [0, 1, 2]

    def test_correction_factor_changes(self):
        buf = self._fake_buffer(T=16)
        trainer = PureHAPPOTrainer(PureHAPPOPolicy(num_agents=3),
                                    actor_lr=3e-3, ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        assert len(m["ratio_after_mean_per_agent"]) == 3
        assert len(m["m_abs_mean_after_each_agent"]) >= 3
        active_ratios = [v for i, v in enumerate(m["ratio_after_mean_per_agent"])
                         if m["valid_sample_count_per_agent"][i] > 0]
        assert len(active_ratios) > 0
        assert any(abs(v - 1.0) > 1e-7 for v in active_ratios), \
            f"correction factor should change, got {active_ratios}"
        assert m["valid_sample_count_per_agent"][0] > 0

    def test_inactive_agent_no_contamination(self):
        policy = PureHAPPOPolicy(num_agents=3)
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        buf = HAPPORolloutBuffer(8, 3, 96, 480, 3, [0, 1, 1])
        for t in range(8):
            obs = torch.randn(3, 96); crit = torch.randn(480)
            with torch.no_grad():
                out = policy.act(obs, critic_state=crit)
            active = np.ones(3, dtype=np.float32); active[1] = 0.0
            buf.store(obs.numpy(), crit.numpy(), out["action"].numpy(),
                      out["log_prob"].numpy(),
                      np.random.randn(3).astype(np.float32) * 0.1,
                      np.zeros(3, dtype=np.float32), out["value"].item(),
                      active, env_id=0)
        trainer = PureHAPPOTrainer(policy, ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        assert m["valid_sample_count_per_agent"][1] == 0
        assert m["ratio_after_mean_per_agent"][1] == 0.0
        assert not np.isnan(m["actor_loss_mean"])
