import random

import numpy as np
import torch

from aircombat_env_v1.ppo import Actor, Critic, deterministic_action
from aircombat_env_v1.training import (
    TrainingConfig, load_training_state, save_training_state)


def test_checkpoint_reload_preserves_action_and_training_state(tmp_path):
    actor, critic = Actor(), Critic()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    observation = torch.randn(1, 20)
    loss = actor(observation)[0].sum() + critic(observation).sum()
    loss.backward()
    optimizer.step()
    expected_action = deterministic_action(actor, observation)
    path = tmp_path / "training_state.pt"
    save_training_state(
        path, actor, critic, optimizer, 123, 7, 0.5, TrainingConfig())

    restored_actor, restored_critic = Actor(), Critic()
    restored_optimizer = torch.optim.Adam(
        list(restored_actor.parameters()) + list(restored_critic.parameters()),
        lr=3e-4)
    state = load_training_state(
        path, restored_actor, restored_critic, restored_optimizer, "cpu")
    actual_action = deterministic_action(restored_actor, observation)
    torch.testing.assert_close(actual_action, expected_action)
    assert state["global_step"] == 123
    assert state["update_index"] == 7
    assert restored_optimizer.state_dict()["state"]
    assert isinstance(random.getstate(), tuple)
    assert isinstance(np.random.get_state(), tuple)
