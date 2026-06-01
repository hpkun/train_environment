from __future__ import annotations

import numpy as np
import torch

from algorithms.mappo import MAPPOTrainer, RolloutStorage
from uav_env import make_env
from uav_env.wrappers import MAPPOEnvWrapper


def test_mappo_update_smoke():
    device = torch.device("cpu")
    env = MAPPOEnvWrapper(make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml"))
    obs, state, _info = env.reset(seed=42)
    trainer = MAPPOTrainer(
        obs_dim=env.obs_shape,
        state_dim=env.state_shape,
        action_dim=env.action_shape,
        hidden_dim=32,
        lr=1e-3,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=10.0,
        device=device,
    )
    storage = RolloutStorage(
        rollout_steps=4,
        num_agents=env.num_agents,
        obs_dim=env.obs_shape,
        state_dim=env.state_shape,
        action_dim=env.action_shape,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )
    for _ in range(4):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        state_batch = state_t.unsqueeze(0).expand(env.num_agents, -1)
        batch = trainer.model.act(obs_t, state_batch)
        actions = batch.actions.cpu().numpy()
        next_obs, next_state, rewards, dones, _info = env.step(actions)
        storage.insert(
            obs=obs,
            state=state,
            actions=actions,
            log_probs=batch.log_probs.cpu().numpy(),
            values=batch.values.cpu().numpy(),
            rewards=rewards.astype(np.float32),
            dones=dones,
        )
        obs, state = next_obs, next_state

    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        next_value = trainer.model.value(state_t.unsqueeze(0).expand(env.num_agents, -1)).cpu().numpy()
    stats = trainer.update(storage.compute_batch(next_value), epochs=1, minibatch_size=8)
    assert set(stats) == {"policy_loss", "value_loss", "entropy"}
    env.close()
