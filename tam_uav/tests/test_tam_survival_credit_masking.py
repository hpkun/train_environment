import numpy as np
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer


def test_death_transition_reward_and_pre_step_active_mask_are_recorded():
    buffer = HAPPORolloutBuffer(1, 3, 2, 3, 4, [0, 1, 1])
    buffer.store(
        np.zeros((3, 2)), np.zeros(3), np.zeros((3, 4)), np.zeros(3),
        np.array([-20.0, 0.0, 0.0]), np.ones(3), 0.0,
        np.ones(3), next_value=0.0,
        death_transition_masks=np.array([1.0, 0.0, 0.0]),
    )
    data = buffer.get(torch.device("cpu"))
    assert data["active_masks"][0, 0].item() == 1.0
    assert data["death_transition_masks"][0, 0].item() == 1.0
    assert data["rewards"][0, 0].item() == -20.0
