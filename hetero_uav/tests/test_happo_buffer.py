import torch

from algorithms.happo import HAPPORolloutBuffer


def test_happo_rollout_buffer_store_and_get():
    buffer = HAPPORolloutBuffer(4, 3, 96, 480, 3, role_ids=[0, 1, 1])
    for _ in range(2):
        buffer.store(
            actor_obs=torch.zeros(3, 96).numpy(),
            critic_state=torch.zeros(480).numpy(),
            actions=torch.zeros(3, 3).numpy(),
            log_probs=torch.zeros(3).numpy(),
            rewards=torch.ones(3).numpy(),
            dones=torch.zeros(3).numpy(),
            value=0.5,
            active_masks=torch.tensor([1.0, 0.0, 1.0]).numpy(),
        )
    data = buffer.get(torch.device("cpu"))
    assert len(buffer) == 2
    assert data["actor_obs"].shape == (2, 3, 96)
    assert data["critic_state"].shape == (2, 480)
    assert data["active_masks"].shape == (2, 3)
    assert data["role_ids"].tolist() == [0, 1, 1]
