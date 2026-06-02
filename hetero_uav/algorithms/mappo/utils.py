"""GAE and utility functions for MAPPO baseline."""
from __future__ import annotations

import torch


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Compute GAE advantages and returns."""
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns
