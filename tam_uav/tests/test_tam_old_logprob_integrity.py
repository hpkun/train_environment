"""Verify old_log_probs are not mutated during PPO update."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def test_old_log_probs_immutable_during_update():
    """Buffer old_log_probs must remain unchanged after trainer.update()."""
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    device = torch.device("cpu")
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    ).to(device)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=2e-4, critic_lr=5e-4,
        clip_param=0.2, entropy_coef=0.02, max_grad_norm=10.0,
        ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="role",
        agent_ids=["red_0", "red_1", "red_2"],
    )

    num_red = 3
    roles = [0, 1, 1]
    T = 4
    buf = HAPPORolloutBuffer(
        max_len=T, num_red=num_red, actor_dim=96, critic_dim=480,
        action_dim=4, role_ids=roles, rnn_hidden_size=128,
        action_dtype=np.int64, num_envs=1,
    )

    obs_full = np.random.randn(T, num_red, 96).astype(np.float32) * 0.1
    critic_full = np.random.randn(T, 480).astype(np.float32) * 0.1
    actions_full = np.random.randint(0, 40, (T, num_red, 4)).astype(np.int64)

    with torch.no_grad():
        seq_out = policy.evaluate_action_sequence(
            torch.as_tensor(obs_full, device=device),
            roles,
            torch.as_tensor(critic_full, device=device),
            torch.as_tensor(actions_full, device=device),
            initial_hidden=torch.zeros(num_red, 128, device=device),
            episode_start_masks=torch.ones(T, num_red, device=device),
            active_masks=torch.ones(T, num_red, device=device),
        )
    log_probs_full = seq_out["log_prob"].cpu().numpy()

    for t in range(T):
        active = np.ones(num_red, dtype=np.float32)
        buf.store(
            obs_full[t], critic_full[t], actions_full[t], log_probs_full[t],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            0.0, active, next_value=0.0, env_id=0,
            rnn_hidden=np.zeros((num_red, 128), dtype=np.float32),
            episode_start_masks=np.ones(num_red, dtype=np.float32),
        )

    old_log_probs_before = buf.log_probs[:T].copy()
    _ = trainer.update(buf)
    old_log_probs_after = buf.log_probs[:T]

    assert np.array_equal(old_log_probs_before, old_log_probs_after), (
        "old_log_probs were mutated during trainer.update()"
    )
