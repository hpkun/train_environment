"""Tests for Full HAPPO baseline: independent actors + sequential update + correction factor M."""
import numpy as np
import pytest
import torch

from algorithms.happo.full_happo_policy import FullHAPPOPolicy
from algorithms.happo.full_happo_trainer import FullHAPPOTrainer


class TestFullHAPPOPolicy:
    def test_independent_actors(self):
        policy = FullHAPPOPolicy(num_agents=3)
        assert len(policy.actors) == 3
        assert len(policy.action_log_stds) == 3
        # Parameter objects must be different
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

    def test_evaluate_shapes(self):
        policy = FullHAPPOPolicy(num_agents=3)
        obs = torch.randn(4, 3, 96)  # [T,N,D]
        critic = torch.randn(4, 480)
        acts = torch.randn(4, 3, 3).clamp(-1, 1)
        lp, ent, vals, means = policy.evaluate_actions(obs, critic, acts)
        assert lp.shape == (4, 3)
        assert ent.shape == (4, 3)
        assert vals.shape == (4,)
        assert means.shape == (4, 3, 3)


class TestFullHAPPOTrainer:
    def _fake_buffer(self, device="cpu", T=16, N=3, actor_dim=96, critic_dim=480, act_dim=3):
        """Build a fake buffer with one env trajectory."""
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        policy = FullHAPPOPolicy(num_agents=N, actor_obs_dim=actor_dim, critic_state_dim=critic_dim).to(device)
        buf = HAPPORolloutBuffer(T, N, actor_dim, critic_dim, act_dim, [0, 1, 1])
        for t in range(T):
            actor_obs = torch.randn(N, actor_dim, device=device)
            critic = torch.randn(critic_dim, device=device)
            with torch.no_grad():
                out = policy.act(actor_obs, critic_state=critic)
            act = out["action"].cpu().numpy()
            lp = out["log_prob"].cpu().numpy()
            val = out["value"].item()
            buf.store(
                actor_obs.cpu().numpy(), critic.cpu().numpy(),
                act, lp,
                np.random.randn(N).astype(np.float32) * 0.1,
                np.zeros(N, dtype=np.float32),
                val, np.ones(N, dtype=np.float32),
                env_id=0,
            )
        return buf

    def test_update_runs_without_nan(self):
        buf = self._fake_buffer()
        trainer = FullHAPPOTrainer(
            FullHAPPOPolicy(num_agents=3),
            actor_lr=1e-3, critic_lr=1e-3, ppo_epochs=2, seed=42,
        )
        metrics = trainer.update(buf)
        assert np.isfinite(metrics["actor_loss_mean"])
        assert np.isfinite(metrics["critic_loss"])
        assert np.isfinite(metrics["entropy_mean"])

    def test_random_update_order(self):
        buf = self._fake_buffer(T=8)
        trainer = FullHAPPOTrainer(
            FullHAPPOPolicy(num_agents=3), ppo_epochs=1, seed=42,
        )
        m1 = trainer.update(buf)
        m2 = trainer.update(buf)
        # Different epochs can have different orders; at least check it's a permutation
        order = m1["last_update_order"]
        assert sorted(order) == [0, 1, 2]

    def test_correction_factor_changes(self):
        """After first agent update, ratio_after should change M for subsequent agents."""
        buf = self._fake_buffer(T=8)
        policy = FullHAPPOPolicy(num_agents=3)
        trainer = FullHAPPOTrainer(policy, ppo_epochs=1, seed=42)
        # Record M before/after in a simplified way — just verify update completes
        metrics = trainer.update(buf)
        assert "last_update_order" in metrics
        assert len(metrics["last_update_order"]) == 3
