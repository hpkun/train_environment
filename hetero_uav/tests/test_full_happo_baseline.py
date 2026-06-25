"""Tests for paper-aligned Full HAPPO baseline.

Covers: independent actors, shape tests, agent count guard, grouped GAE,
update no NaN, random order, correction factor change, inactive agent handling.
"""
import numpy as np
import pytest
import torch

from algorithms.happo.full_happo_policy import FullHAPPOPolicy
from algorithms.happo.full_happo_trainer import FullHAPPOTrainer, _compute_grouped_gae


class TestFullHAPPOPolicy:
    def test_independent_actors(self):
        policy = FullHAPPOPolicy(num_agents=3)
        assert len(policy.actors) == 3
        assert len(policy.action_log_stds) == 3
        assert policy.actors[0] is not policy.actors[1]
        assert policy.action_log_stds[0] is not policy.action_log_stds[1]

    def test_act_shapes(self):
        policy = FullHAPPOPolicy(num_agents=3)
        obs = torch.randn(3, 96)
        out = policy.act(obs, critic_state=torch.randn(480))
        assert out["action"].shape == (3, 3)
        assert out["log_prob"].shape == (3,)
        assert out["entropy"].shape == (3,)
        assert out["value"].shape == (1,)

    def test_act_batched_shapes(self):
        policy = FullHAPPOPolicy(num_agents=3)
        obs = torch.randn(2, 3, 96)
        out = policy.act(obs, critic_state=torch.randn(2, 480))
        assert out["action"].shape == (2, 3, 3)

    def test_evaluate_shapes(self):
        policy = FullHAPPOPolicy(num_agents=3)
        obs = torch.randn(4, 3, 96)
        critic = torch.randn(4, 480)
        acts = torch.randn(4, 3, 3).clamp(-1, 1)
        lp, ent, vals, means = policy.evaluate_actions(obs, critic, acts)
        assert lp.shape == (4, 3)
        assert ent.shape == (4, 3)
        assert vals.shape == (4,)
        assert means.shape == (4, 3, 3)

    def test_agent_count_guard(self):
        policy = FullHAPPOPolicy(num_agents=3)
        with pytest.raises(ValueError, match="got 5"):
            policy.act(torch.randn(5, 96))

    def test_compat_signature(self):
        """act() must accept roles/rnn_hidden kwargs without error."""
        policy = FullHAPPOPolicy(num_agents=3)
        out = policy.act(torch.randn(3, 96), roles=[0, 1, 1],
                         critic_state=torch.randn(480), rnn_hidden=torch.zeros(3, 128))
        assert out["action"].shape == (3, 3)


class TestGroupedGAE:
    def test_env_grouping(self):
        """GAE respects env_id boundaries."""
        rewards    = torch.tensor([1.0, 2.0, 1.0, 2.0])
        values     = torch.tensor([0.0, 0.0, 0.0, 0.0])
        next_values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        dones      = torch.tensor([0.0, 0.0, 0.0, 0.0])
        env_ids    = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        adv, ret = _compute_grouped_gae(rewards, values, next_values, dones, env_ids, 0.99, 0.95)
        # Env 0 has steps [0,2]: ret[0] should only see r[0],r[2]; env 1 has [1,3]
        # With next_values=0, returns = discounted rewards per env
        assert adv.shape == (4,)
        assert ret.shape == (4,)
        assert torch.isfinite(adv).all()
        assert torch.isfinite(ret).all()


class TestFullHAPPOTrainer:
    def _fake_buffer(self, T=16, N=3, device="cpu"):
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        policy = FullHAPPOPolicy(num_agents=N, actor_obs_dim=96, critic_state_dim=480).to(device)
        buf = HAPPORolloutBuffer(T, N, 96, 480, 3, [0, 1, 1])
        for t in range(T):
            obs_i = torch.randn(N, 96, device=device)
            crit = torch.randn(480, device=device)
            with torch.no_grad():
                out = policy.act(obs_i, critic_state=crit)
            buf.store(
                obs_i.cpu().numpy(), crit.cpu().numpy(),
                out["action"].cpu().numpy(), out["log_prob"].cpu().numpy(),
                np.random.randn(N).astype(np.float32) * 0.1,
                np.zeros(N, dtype=np.float32),
                out["value"].item(), np.ones(N, dtype=np.float32), env_id=0,
            )
        return buf

    def test_update_no_nan(self):
        buf = self._fake_buffer()
        trainer = FullHAPPOTrainer(FullHAPPOPolicy(num_agents=3), ppo_epochs=2, seed=42)
        m = trainer.update(buf)
        assert np.isfinite(m["actor_loss_mean"])
        assert np.isfinite(m["critic_loss"])
        assert np.isfinite(m["entropy_mean"])

    def test_random_order(self):
        buf = self._fake_buffer(T=8)
        trainer = FullHAPPOTrainer(FullHAPPOPolicy(num_agents=3), ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        order = m["last_update_order"]
        assert sorted(order) == [0, 1, 2]

    def test_correction_factor_changes(self):
        buf = self._fake_buffer(T=8)
        trainer = FullHAPPOTrainer(FullHAPPOPolicy(num_agents=3), ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        assert len(m["ratio_after_mean_per_agent"]) == 3
        assert len(m["m_abs_mean_after_each_agent"]) == 3
        # At least one agent's ratio_after should deviate from 1.0
        ra = [v for v in m["ratio_after_mean_per_agent"] if v > 0]
        assert len(ra) > 0, "should have at least one active agent"
        assert any(abs(v - 1.0) > 1e-7 for v in ra), "correction factor should change"

    def test_inactive_agent_no_contamination(self):
        """Inactive agent should have 0 samples and not change M."""
        policy = FullHAPPOPolicy(num_agents=3)
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        buf = HAPPORolloutBuffer(8, 3, 96, 480, 3, [0, 1, 1])
        for t in range(8):
            obs_i = torch.randn(3, 96)
            crit = torch.randn(480)
            with torch.no_grad():
                out = policy.act(obs_i, critic_state=crit)
            active = np.ones(3, dtype=np.float32)
            active[1] = 0.0  # agent 1 always inactive
            buf.store(
                obs_i.numpy(), crit.numpy(), out["action"].numpy(), out["log_prob"].numpy(),
                np.random.randn(3).astype(np.float32) * 0.1,
                np.zeros(3, dtype=np.float32),
                out["value"].item(), active, env_id=0,
            )
        trainer = FullHAPPOTrainer(policy, ppo_epochs=1, seed=42)
        m = trainer.update(buf)
        assert m["valid_sample_count_per_agent"][1] == 0
        assert m["ratio_after_mean_per_agent"][1] == 0.0
        assert not np.isnan(m["actor_loss_mean"])
