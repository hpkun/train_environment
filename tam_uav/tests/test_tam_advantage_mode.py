"""Test advantage_mode: team_average and per_agent_reward."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def test_team_average_keeps_existing_logic():
    """Default team_average mode maintains original team_reward mean."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
        entropy_coef=0.01, ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="agent", agent_ids=["red_0", "red_1", "red_2"],
        advantage_mode="team_average",
    )
    assert trainer.advantage_mode == "team_average"

    T, N = 4, 3
    buf = HAPPORolloutBuffer(
        max_len=T, num_red=N, actor_dim=96, critic_dim=480, action_dim=4,
        role_ids=[0, 1, 1], rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )
    obs = np.random.randn(T, N, 96).astype(np.float32) * 0.1
    critic = np.random.randn(T, 480).astype(np.float32) * 0.1
    actions = np.random.randint(0, 40, (T, N, 4)).astype(np.int64)
    roles = [0, 1, 1]

    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            torch.as_tensor(obs, device=device), roles,
            torch.as_tensor(critic, device=device),
            torch.as_tensor(actions, device=device),
            initial_hidden=torch.zeros(N, 128, device=device),
            episode_start_masks=torch.ones(T, N, device=device),
            active_masks=torch.ones(T, N, device=device),
        )
    for t in range(T):
        active = np.ones(N, dtype=np.float32)
        buf.store(obs[t], critic[t], actions[t], out["log_prob"][t].cpu().numpy(),
                  np.array([0.5, 0.2, 0.1], dtype=np.float32),
                  np.array([0.0] * N, dtype=np.float32),
                  0.0, active, next_value=0.0, env_id=0,
                  rnn_hidden=np.zeros((N, 128), dtype=np.float32),
                  episode_start_masks=np.ones(N, dtype=np.float32))

    data = buf.get(device)
    adv, ret = trainer._advantages_and_returns(data)
    assert adv.ndim == 1, f"team_average advantage should be [T], got {adv.shape}"
    assert ret.ndim == 1, f"team_average returns should be [T], got {ret.shape}"


def test_per_agent_reward_returns_per_agent_advantage():
    """per_agent_reward mode returns [T,N] advantages."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
        entropy_coef=0.01, ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="agent", agent_ids=["red_0", "red_1", "red_2"],
        advantage_mode="per_agent_reward",
    )
    assert trainer.advantage_mode == "per_agent_reward"

    T, N = 4, 3
    buf = HAPPORolloutBuffer(
        max_len=T, num_red=N, actor_dim=96, critic_dim=480, action_dim=4,
        role_ids=[0, 1, 1], rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )
    obs = np.random.randn(T, N, 96).astype(np.float32) * 0.1
    critic = np.random.randn(T, 480).astype(np.float32) * 0.1
    actions = np.random.randint(0, 40, (T, N, 4)).astype(np.int64)
    roles = [0, 1, 1]

    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            torch.as_tensor(obs, device=device), roles,
            torch.as_tensor(critic, device=device),
            torch.as_tensor(actions, device=device),
            initial_hidden=torch.zeros(N, 128, device=device),
            episode_start_masks=torch.ones(T, N, device=device),
            active_masks=torch.ones(T, N, device=device),
        )
    for t in range(T):
        active = np.ones(N, dtype=np.float32)
        buf.store(obs[t], critic[t], actions[t], out["log_prob"][t].cpu().numpy(),
                  np.array([0.5, 0.2, 0.1], dtype=np.float32),
                  np.array([0.0] * N, dtype=np.float32),
                  0.0, active, next_value=0.0, env_id=0,
                  rnn_hidden=np.zeros((N, 128), dtype=np.float32),
                  episode_start_masks=np.ones(N, dtype=np.float32))

    data = buf.get(device)
    adv, ret = trainer._advantages_and_returns(data)
    assert adv.ndim == 2, f"per_agent_reward advantage should be [T,N], got {adv.shape}"
    assert adv.shape[1] == N, f"should have {N} agent columns, got {adv.shape[1]}"


def test_per_agent_not_diluted():
    """Per-agent reward with MAV death -4.0 should NOT be diluted to -1.33."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
        entropy_coef=0.01, ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="agent", agent_ids=["red_0", "red_1", "red_2"],
        advantage_mode="per_agent_reward",
    )

    T, N = 3, 3
    buf = HAPPORolloutBuffer(
        max_len=T, num_red=N, actor_dim=96, critic_dim=480, action_dim=4,
        role_ids=[0, 1, 1], rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )
    obs = np.zeros((T, N, 96), dtype=np.float32)
    critic = np.zeros((T, 480), dtype=np.float32)
    actions = np.zeros((T, N, 4), dtype=np.int64)
    roles = [0, 1, 1]

    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            torch.as_tensor(obs, device=device), roles,
            torch.as_tensor(critic, device=device),
            torch.as_tensor(actions, device=device),
            initial_hidden=torch.zeros(N, 128, device=device),
            episode_start_masks=torch.ones(T, N, device=device),
            active_masks=torch.ones(T, N, device=device),
        )
    # t=0: MAV gets -4 death penalty, UAVs get 0
    buf.store(obs[0], critic[0], actions[0], out["log_prob"][0].cpu().numpy(),
              np.array([-4.0, 0.0, 0.0], dtype=np.float32),
              np.array([0.0] * N, dtype=np.float32),
              0.0, np.ones(N, dtype=np.float32), next_value=0.0, env_id=0,
              rnn_hidden=np.zeros((N, 128), dtype=np.float32),
              episode_start_masks=np.ones(N, dtype=np.float32))
    # t=1, t=2: no death, small rewards
    for t in range(1, T):
        buf.store(obs[t], critic[t], actions[t], out["log_prob"][t].cpu().numpy(),
                  np.array([0.01, 0.01, 0.01], dtype=np.float32),
                  np.array([0.0] * N, dtype=np.float32),
                  0.0, np.ones(N, dtype=np.float32), next_value=0.0, env_id=0,
                  rnn_hidden=np.zeros((N, 128), dtype=np.float32),
                  episode_start_masks=np.ones(N, dtype=np.float32))

    data = buf.get(device)
    adv, _ret = trainer._advantages_and_returns(data)
    # red_0 (MAV) at t=0 should have much more negative advantage than red_1/red_2
    assert adv.shape == (T, N)
    mav_adv = abs(float(adv[0, 0]))
    uav1_adv = abs(float(adv[0, 1]))
    uav2_adv = abs(float(adv[0, 2]))
    assert mav_adv > 3 * uav1_adv, (
        f"MAV adv ({mav_adv:.4f}) should be >> UAV1 adv ({uav1_adv:.4f}) "
        f"when MAV gets death penalty and UAVs don't")


def test_per_agent_does_not_change_reward_buffer():
    """per_agent_reward mode must not modify data['rewards']."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
        entropy_coef=0.01, ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="agent", agent_ids=["red_0", "red_1", "red_2"],
        advantage_mode="per_agent_reward",
    )

    T, N = 2, 3
    buf = HAPPORolloutBuffer(
        max_len=T, num_red=N, actor_dim=96, critic_dim=480, action_dim=4,
        role_ids=[0, 1, 1], rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )
    obs = np.zeros((T, N, 96), dtype=np.float32)
    critic = np.zeros((T, 480), dtype=np.float32)
    actions = np.zeros((T, N, 4), dtype=np.int64)
    original_rewards = np.array([[-4.0, 0.0, 0.0], [0.01, 0.01, 0.01]], dtype=np.float32)

    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            torch.as_tensor(obs, device=device), [0, 1, 1],
            torch.as_tensor(critic, device=device),
            torch.as_tensor(actions, device=device),
            initial_hidden=torch.zeros(N, 128, device=device),
            episode_start_masks=torch.ones(T, N, device=device),
            active_masks=torch.ones(T, N, device=device),
        )
    for t in range(T):
        buf.store(obs[t], critic[t], actions[t], out["log_prob"][t].cpu().numpy(),
                  original_rewards[t],
                  np.array([0.0] * N, dtype=np.float32),
                  0.0, np.ones(N, dtype=np.float32), next_value=0.0, env_id=0,
                  rnn_hidden=np.zeros((N, 128), dtype=np.float32),
                  episode_start_masks=np.ones(N, dtype=np.float32))

    data = buf.get(device)
    rewards_before = data["rewards"].clone()
    _adv, _ret = trainer._advantages_and_returns(data)
    assert torch.equal(data["rewards"], rewards_before), "per_agent_reward must not mutate rewards"


def test_team_average_default_constructor():
    """Default trainer constructor uses team_average advantage_mode."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)
    trainer = TAMCategoricalHAPPOTrainer(policy)
    assert trainer.advantage_mode == "team_average", "Default should be team_average"
