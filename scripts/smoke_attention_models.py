"""Pure PyTorch smoke test for attention MAPPO model components."""
from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from attention_models import (
    AttentionActor,
    AttentionCritic,
    EntityObservationEncoder,
)


def main():
    batch_size = 2
    n_entities = 5
    entity_dim = 11

    entities = torch.randn(batch_size, n_entities, entity_dim)
    entity_mask = torch.zeros(batch_size, n_entities, dtype=torch.long)
    entity_mask[0, 3] = 1
    entity_mask[1, 4] = 1
    rnn = torch.zeros(batch_size, 128)

    encoder = EntityObservationEncoder(
        entity_dim=entity_dim, hidden_size=128, num_heads=4)
    encoded, attn = encoder(entities, entity_mask)
    assert encoded.shape == (batch_size, 128)
    assert attn.shape == (batch_size, 4, n_entities, n_entities)
    assert torch.isfinite(encoded).all()

    actor = AttentionActor(
        entity_dim=entity_dim, action_dim=3,
        hidden_size=128, rnn_hidden=128, num_heads=4)
    dist, new_rnn, attn = actor(entities, entity_mask, rnn)
    action = dist.mean
    assert action.shape == (batch_size, 3)
    assert new_rnn.shape == (batch_size, 128)
    assert torch.isfinite(action).all()
    assert attn.shape == (batch_size, 4, n_entities, n_entities)

    critic = AttentionCritic(
        entity_dim=entity_dim, hidden_size=128, num_heads=4)
    value = critic(entities, entity_mask)
    assert value.shape == (batch_size, 1)
    assert torch.isfinite(value).all()

    print("attention smoke test passed")


if __name__ == "__main__":
    main()
