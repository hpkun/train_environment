from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_brma_recurrent_policy_forward_shapes():
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    actor_obs = torch.zeros((3, policy.flat_actor_obs_dim), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)

    assert out["action"].shape == (3, 3)
    assert out["log_prob"].shape == (3,)
    assert out["entropy"].shape == (3,)
    assert out["mean"].shape == (3, 3)
    assert "rnn_hidden" in out
    assert out["rnn_hidden"].shape == (3, 128)
    assert torch.isfinite(out["action"]).all()
    assert torch.isfinite(out["rnn_hidden"]).all()


def test_brma_recurrent_policy_hidden_state_shapes():
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    actor_obs = torch.zeros((2, policy.flat_actor_obs_dim), dtype=torch.float32)

    # Zero hidden init
    h0 = policy.init_hidden(2)
    assert h0.shape == (2, 128)
    assert (h0 == 0).all()

    out1 = policy.act(actor_obs, roles=[0, 1], deterministic=True, rnn_hidden=h0)
    h1 = out1["rnn_hidden"]
    assert h1.shape == (2, 128)

    # Chain with returned hidden state
    out2 = policy.act(actor_obs, roles=[0, 1], deterministic=True, rnn_hidden=h1)
    assert out2["rnn_hidden"].shape == (2, 128)


def test_brma_recurrent_done_reset_hidden_state():
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    actor_obs = torch.randn((2, policy.flat_actor_obs_dim), dtype=torch.float32)

    h0 = policy.init_hidden(2)
    out1 = policy.act(actor_obs, roles=[0, 1], deterministic=True, rnn_hidden=h0)
    h1 = out1["rnn_hidden"].detach()

    # Done reset for agent 0: zero its hidden state
    h1_reset = h1.clone()
    h1_reset[0, :] = 0.0

    # Fresh input to make the difference visible
    actor_obs2 = torch.randn((2, policy.flat_actor_obs_dim), dtype=torch.float32)
    out2_reset = policy.act(actor_obs2, roles=[0, 1], deterministic=True, rnn_hidden=h1_reset)
    out2_noreset = policy.act(actor_obs2, roles=[0, 1], deterministic=True, rnn_hidden=h1)

    # Agent 0's action should differ because hidden state was reset
    assert not torch.allclose(out2_reset["action"][0], out2_noreset["action"][0])


def test_brma_recurrent_evaluate_actions_batch():
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    actor_obs = torch.zeros((4, 3, policy.flat_actor_obs_dim), dtype=torch.float32)
    roles = torch.tensor([[0, 1, 1]] * 4)
    critic = torch.zeros((4, 480), dtype=torch.float32)
    actions = torch.zeros((4, 3, 3), dtype=torch.float32)
    h0 = policy.init_hidden(12).reshape(4, 3, 128)

    log_prob, entropy, value, mean, role_ids = policy.evaluate_actions(
        actor_obs, roles, critic, actions, rnn_hidden=h0)

    assert log_prob.shape == (4, 3)
    assert entropy.shape == (4, 3)
    assert value.shape == (4,)
    assert mean.shape == (4, 3, 3)
    assert role_ids.shape == (4, 3)
    assert torch.isfinite(log_prob).all()


def test_brma_recurrent_policy_save_load_roundtrip(tmp_path):
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    path = tmp_path / "model.pt"
    policy.save(path)
    loaded = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    loaded.load(path, map_location="cpu")
    h0 = loaded.init_hidden(3)
    out = loaded.act(torch.zeros((3, policy.flat_actor_obs_dim)), roles=[0, 1, 1], deterministic=True, rnn_hidden=h0)
    assert out["action"].shape == (3, 3)
    assert "rnn_hidden" in out


def test_rollout_buffer_preserves_recurrent_hidden_state():
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    buffer = HAPPORolloutBuffer(
        max_len=2,
        num_red=3,
        actor_dim=140,
        critic_dim=480,
        action_dim=3,
        role_ids=[0, 1, 1],
        rnn_hidden_size=128,
    )
    hidden = np.ones((3, 128), dtype=np.float32)
    buffer.store(
        actor_obs=np.zeros((3, 140), dtype=np.float32),
        critic_state=np.zeros(480, dtype=np.float32),
        actions=np.zeros((3, 3), dtype=np.float32),
        log_probs=np.zeros(3, dtype=np.float32),
        rewards=np.zeros(3, dtype=np.float32),
        dones=np.zeros(3, dtype=np.float32),
        value=0.0,
        active_masks=np.ones(3, dtype=np.float32),
        next_value=0.0,
        env_id=0,
        rnn_hidden=hidden,
    )

    data = buffer.get(torch.device("cpu"))
    assert data["rnn_hidden"].shape == (1, 3, 128)
    assert torch.allclose(data["rnn_hidden"], torch.ones((1, 3, 128)))


def test_train_and_eval_help_include_brma_recurrent():
    result = subprocess.run(
        [sys.executable, "scripts/train_happo_reference.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "brma_recurrent" in result.stdout


def test_run_brma_recurrent_smoke_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_brma_recurrent_smoke.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch" in result.stdout
    assert "brma_recurrent" in result.stdout
    assert "debug_brma_recurrent_smoke" in result.stdout


def test_brma_entity_tests_still_pass():
    """Verify brma_entity policy is unaffected."""
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3)
    actor_obs = torch.zeros((3, policy.flat_actor_obs_dim), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)
    assert out["action"].shape == (3, 3)
    assert "rnn_hidden" not in out  # non-recurrent should not have rnn_hidden


def test_recurrent_replay_hidden_is_pre_action():
    """evaluate_actions with pre-action hidden must reproduce rollout log_prob."""
    from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy

    torch.manual_seed(42)
    policy = BRMARecurrentHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
    policy.eval()

    actor_obs = torch.randn((2, policy.flat_actor_obs_dim), dtype=torch.float32)
    roles = [0, 1]
    critic = torch.randn((480,), dtype=torch.float32)
    actions = torch.randn((2, 3), dtype=torch.float32)

    # Simulate rollout: act with h0 鈫?h1
    h0 = policy.init_hidden(2)
    out = policy.act(actor_obs, roles=roles, critic_state=critic, deterministic=False, rnn_hidden=h0)
    rollout_log_prob = out["log_prob"].detach().clone()
    rollout_action = out["action"].detach().clone()

    # Simulate PPO replay with pre-action hidden h0
    log_prob_ppo, _, _, _, _ = policy.evaluate_actions(
        actor_obs.unsqueeze(0),
        torch.tensor(roles).unsqueeze(0),
        critic.unsqueeze(0),
        rollout_action.unsqueeze(0),
        rnn_hidden=h0.unsqueeze(0),
    )
    assert torch.allclose(log_prob_ppo.squeeze(0), rollout_log_prob, atol=1e-5), (
        "PPO replay with pre-action hidden must reproduce rollout log_prob"
    )

    # Simulate PPO replay with post-action hidden h1 (this was the bug)
    h1 = out["rnn_hidden"].detach()
    log_prob_bug, _, _, _, _ = policy.evaluate_actions(
        actor_obs.unsqueeze(0),
        torch.tensor(roles).unsqueeze(0),
        critic.unsqueeze(0),
        rollout_action.unsqueeze(0),
        rnn_hidden=h1.unsqueeze(0),
    )
    assert not torch.allclose(log_prob_bug.squeeze(0), rollout_log_prob, atol=1e-5), (
        "PPO replay with post-action hidden must NOT match rollout log_prob "
        "(this would mean the buffer stored the wrong hidden state)"
    )


def test_flat_policy_tests_still_pass():
    """Verify flat policy is unaffected."""
    from algorithms.happo.happo_policy import HAPPOReferencePolicy
    policy = HAPPOReferencePolicy(96, 480)
    actor_obs = torch.zeros((3, 96), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)
    assert out["action"].shape == (3, 3)
    assert "rnn_hidden" not in out


def test_brma_entity_dim_19_old_checkpoint_compat():
    """entity_dim=19 still works for old checkpoint loading (full-geometry truncated)."""
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    assert policy.entity_dim == 19
    actor_obs = torch.zeros((3, policy.flat_actor_obs_dim), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)
    assert out["action"].shape == (3, 3)


def test_brma_entity_dim_30_default_full_geometry():
    """Default entity_dim=30 must accommodate full-geometry from enemy_flat_dim=18."""
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
    policy = BRMAEntityHAPPOReferencePolicy(critic_state_dim=480, action_dim=3)
    assert policy.entity_dim == 30
    assert policy.enemy_flat_dim == 18
    # entity_dim=30 >= 19 + (18-7) = 30, so all full-geometry fits
    extra_slots = policy.entity_dim - 19
    assert extra_slots >= (policy.enemy_flat_dim - 7), (
        f"entity_dim={policy.entity_dim} cannot hold all enemy_flat_dim={policy.enemy_flat_dim} dims"
    )
