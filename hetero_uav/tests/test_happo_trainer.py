import torch

from algorithms.happo import HAPPOReferencePolicy, HAPPORolloutBuffer, HAPPOReferenceTrainer


def _fill_buffer(policy):
    buffer = HAPPORolloutBuffer(4, 3, 96, 480, 3, role_ids=[0, 1, 1])
    for t in range(4):
        obs = torch.randn(3, 96) * 0.1
        obs[0, 7] = 1.0
        obs[1:, 8] = 1.0
        state = torch.randn(480) * 0.1
        with torch.no_grad():
            out = policy.act(obs, roles=[0, 1, 1], critic_state=state)
        rewards = torch.tensor([0.1, 0.2, 0.2])
        dones = torch.tensor([0.0, 0.0, 0.0])
        if t == 3:
            dones[:] = 1.0
        buffer.store(
            obs.numpy(),
            state.numpy(),
            out["action"].numpy(),
            out["log_prob"].numpy(),
            rewards.numpy(),
            dones.numpy(),
            float(out["value"].item()),
            torch.tensor([1.0, 1.0, 1.0]).numpy(),
        )
    return buffer


def _param_snapshot(params):
    return [p.detach().clone() for p in params]


def _changed(before, params):
    return any(not torch.allclose(old, new.detach()) for old, new in zip(before, params))


def test_happo_reference_trainer_update_runs_and_updates_all_modules():
    torch.manual_seed(0)
    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=2)
    buffer = _fill_buffer(policy)
    mav_before = _param_snapshot(policy.mav_actor.parameters())
    uav_before = _param_snapshot(policy.uav_actor.parameters())
    critic_before = _param_snapshot(policy.critic.parameters())
    stats = trainer.update(buffer)
    for key in ["actor_loss_mav", "actor_loss_uav", "critic_loss"]:
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))
    assert _changed(mav_before, policy.mav_actor.parameters())
    assert _changed(uav_before, policy.uav_actor.parameters())
    assert _changed(critic_before, policy.critic.parameters())
