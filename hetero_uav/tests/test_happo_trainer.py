import torch

from algorithms.happo import HAPPOReferencePolicy, HAPPORolloutBuffer, HAPPOReferenceTrainer
from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy


def _fill_buffer(policy):
    hidden_size = int(getattr(policy, "rnn_hidden_size", 0))
    buffer = HAPPORolloutBuffer(4, 3, 96, 480, 3, role_ids=[0, 1, 1], rnn_hidden_size=hidden_size)
    hidden = torch.zeros((3, hidden_size), dtype=torch.float32) if hidden_size > 0 else None
    for t in range(4):
        obs = torch.randn(3, 96) * 0.1
        obs[0, 7] = 1.0
        obs[1:, 8] = 1.0
        state = torch.randn(480) * 0.1
        with torch.no_grad():
            if hidden is None:
                out = policy.act(obs, roles=[0, 1, 1], critic_state=state)
            else:
                out = policy.act(obs, roles=[0, 1, 1], critic_state=state, rnn_hidden=hidden)
        rewards = torch.tensor([0.1, 0.2, 0.2])
        dones = torch.tensor([0.0, 0.0, 0.0])
        if t == 3:
            dones[:] = 1.0
        hidden_to_store = hidden.numpy() if hidden is not None else None
        buffer.store(
            obs.numpy(),
            state.numpy(),
            out["action"].numpy(),
            out["log_prob"].numpy(),
            rewards.numpy(),
            dones.numpy(),
            float(out["value"].item()),
            torch.tensor([1.0, 1.0, 1.0]).numpy(),
            next_value=float(out["value"].item()),
            rnn_hidden=hidden_to_store,
        )
        if hidden is not None and "next_rnn_hidden" in out:
            hidden = out["next_rnn_hidden"].detach()
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


def _optimizer_param_ids(optimizer):
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def test_brma_recurrent_masked_optimizer_parameter_ownership_is_disjoint():
    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=False,
        biased_mask=False,
    )
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)

    mav_ids = _optimizer_param_ids(trainer.mav_opt)
    uav_ids = _optimizer_param_ids(trainer.uav_opt)
    shared_ids = _optimizer_param_ids(trainer.shared_actor_opt)

    assert mav_ids.isdisjoint(uav_ids)
    assert shared_ids.isdisjoint(mav_ids)
    assert shared_ids.isdisjoint(uav_ids)

    encoder_gru_ids = {
        id(param)
        for module in [policy.encoder, policy.rnn]
        for param in module.parameters()
    }
    assert encoder_gru_ids
    assert encoder_gru_ids <= shared_ids

    mav_role_ids = {id(p) for p in policy.mav_actor.parameters()} | {id(policy.action_log_std_mav)}
    uav_role_ids = {id(p) for p in policy.uav_actor.parameters()} | {id(policy.action_log_std_uav)}
    assert mav_role_ids <= mav_ids
    assert uav_role_ids <= uav_ids
    assert mav_role_ids.isdisjoint(shared_ids)
    assert uav_role_ids.isdisjoint(shared_ids)


def test_brma_recurrent_masked_trainer_update_runs_with_disjoint_optimizers():
    torch.manual_seed(1)
    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=False,
        biased_mask=False,
    )
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)
    stats = trainer.update(_fill_buffer(policy))
    for key in ["actor_loss_mav", "actor_loss_uav", "approx_kl_mav", "approx_kl_uav"]:
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))
