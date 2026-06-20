"""Paper-aligned TAM recurrent categorical actor and attention critic."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Categorical

from .brma_entity_policy import BRMAEntityObservationEncoder
from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


def _head(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, out_dim)
    )


class TAMCentralizedAttentionCritic(nn.Module):
    def __init__(self, critic_state_dim: int = 480, actor_obs_dim: int = 96,
                 max_red: int = 5, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        if critic_state_dim != actor_obs_dim * max_red:
            raise ValueError("critic_state_dim must equal actor_obs_dim * max_red")
        self.actor_obs_dim = int(actor_obs_dim)
        self.max_red = int(max_red)
        self.slot_embed = nn.Sequential(
            nn.Linear(actor_obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, critic_state) -> torch.Tensor:
        state = torch.as_tensor(
            critic_state, dtype=torch.float32, device=next(self.parameters()).device
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        slots = state.reshape(-1, self.max_red, self.actor_obs_dim)
        valid = slots.abs().sum(dim=-1) > 0
        all_empty = ~valid.any(dim=-1)
        valid = valid.clone()
        valid[all_empty, 0] = True
        embedded = self.slot_embed(slots)
        attended, _weights = self.attention(
            embedded, embedded, embedded, key_padding_mask=~valid, need_weights=False
        )
        mask = valid.unsqueeze(-1).to(attended.dtype)
        pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.value_head(pooled).squeeze(-1)


class TAMCategoricalRecurrentHAPPOPolicy(nn.Module):
    def __init__(
        self,
        entity_dim: int = 19,
        actor_obs_dim: int = 96,
        critic_state_dim: int = 480,
        action_dim: int = 4,
        action_levels: int = 40,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        rnn_hidden_size: int = 128,
        max_allies: int = 4,
        max_enemies: int = 4,
        max_red: int = 5,
    ):
        super().__init__()
        if action_dim != 4 or action_levels <= 1:
            raise ValueError("TAM categorical policy requires four actions and >1 levels")
        self.entity_dim = int(entity_dim)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.action_levels = int(action_levels)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.max_allies = int(max_allies)
        self.max_enemies = int(max_enemies)
        self.flat_actor_obs_dim = self.actor_obs_dim
        self.action_distribution = "multidiscrete_categorical"

        self.encoder = BRMAEntityObservationEncoder(
            entity_dim=entity_dim, hidden_size=hidden_dim,
            num_heads=num_attention_heads,
        )
        self.rnn = nn.GRUCell(self.encoder.output_dim, rnn_hidden_size)
        logits_dim = action_dim * action_levels
        self.mav_actor = _head(rnn_hidden_size, logits_dim)
        self.uav_actor = _head(rnn_hidden_size, logits_dim)
        self.critic = TAMCentralizedAttentionCritic(
            critic_state_dim, actor_obs_dim, max_red, hidden_dim,
            num_attention_heads,
        )

    @staticmethod
    def infer_role_ids(roles: Iterable[str | int] | torch.Tensor | None,
                       batch: int, device) -> torch.Tensor:
        if roles is None:
            ids = torch.full((batch,), UAV_ROLE_ID, dtype=torch.long, device=device)
            if batch:
                ids[0] = MAV_ROLE_ID
            return ids
        if isinstance(roles, torch.Tensor):
            return roles.to(device=device, dtype=torch.long).reshape(-1)
        values = [MAV_ROLE_ID if role == "mav" else UAV_ROLE_ID if isinstance(role, str) else int(role)
                  for role in roles]
        return torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)

    def actor_shared_parameters(self):
        return list(self.encoder.parameters()) + list(self.rnn.parameters())

    def init_hidden(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(
            batch, self.rnn_hidden_size,
            device=device or next(self.parameters()).device,
        )

    def _flat_to_entities(self, flat_obs: torch.Tensor):
        flat = flat_obs.reshape(-1, flat_obs.shape[-1])
        batch = flat.shape[0]
        entities = torch.zeros(
            batch, 1 + self.max_allies + self.max_enemies, self.entity_dim,
            dtype=flat.dtype, device=flat.device,
        )
        keep = torch.zeros(entities.shape[:2], dtype=torch.bool, device=flat.device)
        ego = flat[:, :12]
        allies_start = 12
        enemies_start = allies_start + self.max_allies * 9
        masks_start = enemies_start + self.max_enemies * 7
        allies = flat[:, allies_start:enemies_start].reshape(batch, self.max_allies, 9)
        enemies = flat[:, enemies_start:masks_start].reshape(batch, self.max_enemies, 7)
        masks = flat[:, masks_start:masks_start + 20]
        ally_valid = masks[:, :self.max_allies]
        ally_alive = masks[:, self.max_allies:self.max_allies * 2]
        enemy_valid = masks[:, self.max_allies * 2:self.max_allies * 2 + self.max_enemies]
        enemy_alive = masks[:, self.max_allies * 2 + self.max_enemies:self.max_allies * 2 + self.max_enemies * 2]
        enemy_observed = masks[:, self.max_allies * 2 + self.max_enemies * 2:self.max_allies * 2 + self.max_enemies * 3]
        entities[:, 0, 0] = 1.0
        entities[:, 0, 3:7] = ego[:, 7:11]
        entities[:, 0, 7:14] = ego[:, :7]
        entities[:, 0, 14] = 1.0
        entities[:, 0, 16] = ego[:, 11]
        keep[:, 0] = True
        for i in range(self.max_allies):
            idx = 1 + i
            entities[:, idx, 1] = 1.0
            entities[:, idx, 3:7] = allies[:, i, 5:9]
            entities[:, idx, 7:12] = allies[:, i, :5]
            entities[:, idx, 14] = 1.0
            keep[:, idx] = (ally_valid[:, i] > .5) & (ally_alive[:, i] > .5)
        for i in range(self.max_enemies):
            idx = 1 + self.max_allies + i
            entities[:, idx, 2] = 1.0
            entities[:, idx, 7:12] = enemies[:, i, :5]
            entities[:, idx, 15] = 1.0
            entities[:, idx, 17:19] = enemies[:, i, 5:7]
            keep[:, idx] = ((enemy_valid[:, i] > .5) & (enemy_alive[:, i] > .5)
                           & (enemy_observed[:, i] > .5))
        return entities, keep

    def encode(self, actor_obs):
        raw = torch.as_tensor(
            actor_obs, dtype=torch.float32, device=next(self.parameters()).device
        )
        leading = tuple(raw.shape[:-1])
        if raw.shape[-1] != self.actor_obs_dim:
            raise ValueError(f"expected actor obs dim {self.actor_obs_dim}")
        entities, keep = self._flat_to_entities(raw)
        pooled, _weights = self.encoder(entities, keep)
        return pooled, leading

    def _logits(self, hidden: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        logits = torch.empty(
            hidden.shape[0], self.action_dim, self.action_levels,
            dtype=hidden.dtype, device=hidden.device,
        )
        mav = roles == MAV_ROLE_ID
        if mav.any():
            logits[mav] = self.mav_actor(hidden[mav]).reshape(-1, self.action_dim, self.action_levels)
        if (~mav).any():
            logits[~mav] = self.uav_actor(hidden[~mav]).reshape(-1, self.action_dim, self.action_levels)
        return logits

    def _actor_forward(self, actor_obs, roles, rnn_hidden=None):
        pooled, leading = self.encode(actor_obs)
        batch = pooled.shape[0]
        hidden = self.init_hidden(batch, pooled.device) if rnn_hidden is None else torch.as_tensor(
            rnn_hidden, dtype=torch.float32, device=pooled.device
        ).reshape(-1, self.rnn_hidden_size)
        hidden_new = self.rnn(pooled, hidden)
        role_ids = self.infer_role_ids(roles, batch, pooled.device)
        return self._logits(hidden_new, role_ids), hidden_new, role_ids, leading

    def act(self, actor_obs, roles=None, critic_state=None, deterministic=False,
            rnn_hidden=None):
        logits, hidden, role_ids, leading = self._actor_forward(
            actor_obs, roles, rnn_hidden
        )
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        probs = dist.probs
        bins = torch.linspace(-1.0, 1.0, self.action_levels, device=logits.device)
        expected = (probs * bins).sum(dim=-1)
        value = self.critic(critic_state) if critic_state is not None else None
        return {
            "action": action.view(*leading, self.action_dim),
            "log_prob": dist.log_prob(action).sum(-1).view(*leading),
            "entropy": dist.entropy().sum(-1).view(*leading),
            "value": value,
            "mean": expected.view(*leading, self.action_dim),
            "role_mask": role_ids.view(*leading),
            "rnn_hidden": hidden.view(*leading, self.rnn_hidden_size),
            "action_logits": logits.view(*leading, self.action_dim, self.action_levels),
            "action_probs": probs.view(*leading, self.action_dim, self.action_levels),
        }

    def evaluate_actions(self, actor_obs, roles, critic_state, actions,
                         rnn_hidden=None):
        logits, _hidden, role_ids, leading = self._actor_forward(
            actor_obs, roles, rnn_hidden
        )
        action_indices = torch.as_tensor(
            actions, dtype=torch.long, device=logits.device
        ).reshape(-1, self.action_dim)
        dist = Categorical(logits=logits)
        probs = dist.probs
        bins = torch.linspace(-1.0, 1.0, self.action_levels, device=logits.device)
        expected = (probs * bins).sum(dim=-1)
        return (
            dist.log_prob(action_indices).sum(-1).view(*leading),
            dist.entropy().sum(-1).view(*leading),
            self.critic(critic_state),
            expected.view(*leading, self.action_dim),
            role_ids.view(*leading),
        )

    def value(self, critic_state):
        return self.critic(critic_state)

    def action_probabilities(self, actor_obs, roles=None, rnn_hidden=None):
        logits, _hidden, role_ids, leading = self._actor_forward(actor_obs, roles, rnn_hidden)
        return torch.softmax(logits, -1).view(*leading, self.action_dim, self.action_levels), role_ids

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
