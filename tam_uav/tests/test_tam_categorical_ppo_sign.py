"""Test categorical PPO sign: negative advantage decreases, positive advantage increases."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def _build_policy():
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    return TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
        action_dim=4, action_levels=40, rnn_hidden_size=128,
    )


def _build_mini_buffer(policy, device, mav_action):
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer
    num_red, actor_dim, critic_dim, action_dim = 3, policy.actor_obs_dim, 480, 4
    roles = [0, 1, 1]
    buf = HAPPORolloutBuffer(
        max_len=1, num_red=num_red, actor_dim=actor_dim, critic_dim=critic_dim,
        action_dim=action_dim, role_ids=roles,
        rnn_hidden_size=policy.rnn_hidden_size, action_dtype=np.int64, num_envs=1,
    )
    obs = np.zeros((num_red, actor_dim), dtype=np.float32)
    obs[0, :7] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    obs[0, 7:11] = [1.0, 0.0, 0.0, 0.0]
    obs[1, :7] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    obs[1, 7:11] = [0.0, 1.0, 0.0, 0.0]
    obs[2, :7] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    obs[2, 7:11] = [0.0, 1.0, 0.0, 0.0]
    critic = np.zeros(critic_dim, dtype=np.float32)
    actions_np = np.tile(np.asarray(mav_action, dtype=np.int64).reshape(1, -1), (num_red, 1))
    # Compute log_prob using evaluate_action_sequence with proper [T, N, ...] shapes
    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            torch.as_tensor(obs[np.newaxis], device=device),  # [1, N, D]
            roles,
            torch.as_tensor(critic[np.newaxis], device=device),  # [1, D]
            torch.as_tensor(actions_np[np.newaxis], device=device),  # [1, N, A]
            initial_hidden=torch.zeros(num_red, policy.rnn_hidden_size, device=device),  # [N, H]
            episode_start_masks=torch.ones(1, num_red, device=device),  # [T, N]
            active_masks=torch.ones(1, num_red, device=device),
        )
    log_probs = out["log_prob"].squeeze(0).cpu().numpy()  # [N]
    active = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    buf.store(
        obs, critic, actions_np, log_probs,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        0.0, active, next_value=0.0, env_id=0, env_step_index=0,
        rnn_hidden=np.zeros((num_red, policy.rnn_hidden_size), dtype=np.float32),
        episode_start_masks=np.ones(num_red, dtype=np.float32),
    )
    return buf


def _run_update(policy, buf, device, advantage_val):
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=2e-4, critic_lr=5e-4,
        clip_param=0.2, entropy_coef=0.02, max_grad_norm=10.0,
        ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="role",
        agent_ids=["red_0", "red_1", "red_2"],
    )
    data = buf.get(device)
    data["advantages"] = torch.full((data["actor_obs"].shape[0],), float(advantage_val), device=device)
    sequences = buf.get_sequences(device)

    policy.train()
    shared_params = list(trainer.shared_actor_params)
    mav_params_list = list(trainer.mav_params)

    if trainer.shared_actor_opt:
        trainer.shared_actor_opt.zero_grad()
    trainer.mav_opt.zero_grad()

    loss_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    valid_sum = torch.zeros((), device=device)

    for seq in sequences:
        out_seq = trainer._evaluate_sequence(seq)
        role_mask = (seq["role_ids"].view(1, -1) == 0).float()  # MAV only
        valid = seq["agent_alive_masks"] * role_mask
        old = seq["old_log_probs"]
        ratio = torch.exp(out_seq["log_prob"] - old)
        adv = data["advantages"][seq["buffer_indices"]].unsqueeze(-1)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        loss_sum = loss_sum - (torch.minimum(surr1, surr2) * valid).sum()
        entropy_sum = entropy_sum + (out_seq["entropy"] * valid).sum()
        valid_sum = valid_sum + valid.sum()

    if valid_sum.item() <= 0:
        return None

    policy_loss = loss_sum / valid_sum
    (policy_loss - 0.02 * (entropy_sum / valid_sum)).backward()

    all_params = shared_params + mav_params_list
    torch.nn.utils.clip_grad_norm_(all_params, 10.0)
    trainer.mav_opt.step()
    if trainer.shared_actor_opt:
        trainer.shared_actor_opt.step()

    return {"policy_loss": float(policy_loss.item()),
            "entropy": float((entropy_sum / valid_sum).item())}


def _get_mav_selected_probs(policy, buf, device):
    """Get MAV selected action probabilities via trainer's _evaluate_sequence."""
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=2e-4, critic_lr=5e-4,
        clip_param=0.2, entropy_coef=0.02, max_grad_norm=10.0,
        ppo_epochs=1, gamma=0.99, gae_lambda=0.95,
        happo_update_granularity="role",
        agent_ids=["red_0", "red_1", "red_2"],
    )
    sequences = buf.get_sequences(device)
    policy.eval()
    with torch.no_grad():
        out_seq = trainer._evaluate_sequence(sequences[0])
    probs = out_seq["action_probs"]  # [T, N, 4, 40]
    actions_idx = sequences[0]["actions"].long()
    selected = probs[0, 0, torch.arange(4), actions_idx[0]]
    return float(torch.prod(selected).item()), selected.cpu().numpy()


def test_negative_advantage_decreases_probability():
    """adv < 0 → selected action probability must decrease."""
    device = torch.device("cpu")
    policy = _build_policy().to(device)
    mav_action = [38, 20, 22, 21]

    buf = _build_mini_buffer(policy, device, mav_action)
    joint_before, per_axis_before = _get_mav_selected_probs(policy, buf, device)
    _run_update(policy, buf, device, advantage_val=-1.0)
    joint_after, per_axis_after = _get_mav_selected_probs(policy, buf, device)

    assert joint_after < joint_before, (
        f"Negative advantage should decrease probability: "
        f"{joint_before:.6f} → {joint_after:.6f} (delta={joint_after-joint_before:+.6f})"
    )


def test_positive_advantage_increases_probability():
    """adv > 0 → selected action probability must increase."""
    device = torch.device("cpu")
    policy = _build_policy().to(device)
    mav_action = [38, 20, 22, 21]

    buf = _build_mini_buffer(policy, device, mav_action)
    joint_before, per_axis_before = _get_mav_selected_probs(policy, buf, device)
    _run_update(policy, buf, device, advantage_val=+1.0)
    joint_after, per_axis_after = _get_mav_selected_probs(policy, buf, device)

    assert joint_after > joint_before, (
        f"Positive advantage should increase probability: "
        f"{joint_before:.6f} → {joint_after:.6f} (delta={joint_after-joint_before:+.6f})"
    )


def test_log_prob_is_sum_across_axes():
    """Joint log_prob must be sum of 4 action axis log_probs."""
    device = torch.device("cpu")
    policy = _build_policy().to(device)
    policy.eval()

    obs = torch.zeros(1, 3, 96, device=device)
    actions = torch.tensor([[[38, 20, 22, 21]]], device=device).expand(1, 3, -1)

    with torch.no_grad():
        out = policy.evaluate_action_sequence(
            obs, [0, 1, 1],
            torch.zeros(1, 480, device=device),
            actions,
            initial_hidden=torch.zeros(3, 128, device=device),
            episode_start_masks=torch.ones(1, 3, device=device),
            active_masks=torch.ones(1, 3, device=device),
        )
    log_prob = out["log_prob"]  # [1, 3]
    probs = out["action_probs"]  # [1, 3, 4, 40]

    selected_probs = probs[0, 0, torch.arange(4), actions[0, 0]]
    manual_log_prob = torch.log(selected_probs).sum()

    assert torch.allclose(log_prob[0, 0], manual_log_prob, atol=1e-5), (
        f"log_prob={log_prob[0,0].item():.6f} ≠ manual sum={manual_log_prob.item():.6f}"
    )
