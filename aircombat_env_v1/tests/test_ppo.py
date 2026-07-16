import numpy as np
import torch

from aircombat_env_v1.ppo import (
    Actor, Critic, RolloutBuffer, SquashedNormal, compute_gae)
from aircombat_env_v1.training import curriculum_stage


def test_squashed_normal_samples_in_bounds_and_has_finite_log_prob():
    distribution = SquashedNormal(torch.zeros(100, 3), torch.ones(100, 3))
    samples = distribution.rsample()
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)
    assert torch.isfinite(distribution.log_prob(samples)).all()


def test_actor_and_critic_output_shapes_and_std_bounds():
    observations = torch.zeros(4, 20)
    actor, critic = Actor(), Critic()
    mean, std = actor(observations)
    assert mean.shape == (4, 3)
    assert std.shape == (4, 3)
    assert torch.all(std >= 0.05)
    assert torch.all(std <= 0.6)
    assert critic(observations).shape == (4,)


def test_compute_gae_matches_hand_calculation():
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0]], dtype=np.float32)
    next_values = np.array([[1.0], [3.0]], dtype=np.float32)
    terminated = np.array([[False], [False]])
    truncated = np.array([[False], [True]])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated,
        gamma=0.9, gae_lambda=0.8)
    second = 2.0 + 0.9 * 3.0 - 1.0
    first = 1.0 + 0.9 * 1.0 - 0.5 + 0.9 * 0.8 * second
    np.testing.assert_allclose(advantages[:, 0], [first, second])
    np.testing.assert_allclose(returns, advantages + values)


def test_terminated_does_not_bootstrap_but_truncated_does():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    next_values = np.full((2, 1), 10.0, dtype=np.float32)
    terminated = np.array([[True], [False]])
    truncated = np.array([[False], [True]])
    advantages, _ = compute_gae(
        rewards, values, next_values, terminated, truncated,
        gamma=0.5, gae_lambda=1.0)
    assert advantages[0, 0] == 1.0
    assert advantages[1, 0] == 6.0


def test_rollout_buffer_shapes():
    buffer = RolloutBuffer(2, 3)
    for _ in range(2):
        buffer.add(
            np.zeros((3, 20)), np.zeros((3, 3)), np.zeros(3),
            np.zeros(3), np.zeros(3), np.zeros(3),
            np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))
    flat = buffer.flatten()
    assert flat["observations"].shape == (6, 20)
    assert flat["actions"].shape == (6, 3)
    assert flat["returns"].shape == (6,)


def test_curriculum_boundaries():
    assert curriculum_stage(0, 100) == 1
    assert curriculum_stage(19, 100) == 1
    assert curriculum_stage(20, 100) == 2
    assert curriculum_stage(69, 100) == 2
    assert curriculum_stage(70, 100) == 3
