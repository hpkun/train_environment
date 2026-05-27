"""BRMA rollout buffer schema and storage.

Provides ``BRMARolloutStorage`` for per-timestep mask-generator outputs,
dual log-prob placeholders, and next-observation placeholders.

This module is **not wired** into training.  The default ``enabled=False``
means no arrays are allocated and ``store_step`` raises ``RuntimeError``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BRMARolloutSchemaConfig:
    """Configuration for BRMA rollout storage dimensions."""

    num_steps: int
    num_envs: int
    num_agents: int
    n_entities: int = 4
    entity_dim: int = 10
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be > 0")
        if self.n_entities <= 0:
            raise ValueError("n_entities must be > 0")
        if self.entity_dim <= 0:
            raise ValueError("entity_dim must be > 0")


class BRMARolloutStorage:
    """Per-timestep storage for BRMA mask generator and dual-policy outputs.

    When ``enabled=False`` no arrays are allocated; ``store_step`` raises.
    """

    def __init__(self, config: BRMARolloutSchemaConfig):
        self._cfg = config
        self._enabled = config.enabled
        T, E, A, N, D = (config.num_steps, config.num_envs,
                         config.num_agents, config.n_entities,
                         config.entity_dim)
        if not self._enabled:
            self._arrays: dict[str, np.ndarray] = {}
            self._valid = None
            return

        self._arrays: dict[str, np.ndarray] = {
            # mask generator outputs
            "p":                  np.zeros((T, E, A, N), dtype=np.float32),
            "msoft":              np.zeros((T, E, A, N), dtype=np.float32),
            "mhard":              np.zeros((T, E, A, N), dtype=np.float32),
            "mR_count":           np.zeros((T, E, A), dtype=np.int64),
            "mB_count":           np.zeros((T, E, A), dtype=np.int64),
            "friendly_drop_mask": np.zeros((T, E, A, N), dtype=bool),
            "enemy_drop_mask":    np.zeros((T, E, A, N), dtype=bool),
            "key_padding_mask":   np.zeros((T, E, A, N), dtype=bool),
            "keep_mask":          np.zeros((T, E, A, N), dtype=bool),
            # dual log-prob placeholders
            "log_prob_unmasked":  np.zeros((T, E, A), dtype=np.float32),
            "log_prob_masked":    np.zeros((T, E, A), dtype=np.float32),
            "entropy_unmasked":   np.zeros((T, E, A), dtype=np.float32),
            "entropy_masked":     np.zeros((T, E, A), dtype=np.float32),
            # next observation placeholders
            "next_entities":       np.zeros((T, E, A, N, D), dtype=np.float32),
            "next_entity_masks":   np.ones((T, E, A, N), dtype=np.int64),
        }
        self._valid = np.zeros((T, E, A), dtype=bool)

    @property
    def has_storage(self) -> bool:
        return self._enabled

    def store_step(
        self,
        step: int,
        env_idx: int,
        agent_idx: int,
        *,
        p: np.ndarray,
        msoft: np.ndarray,
        mhard: np.ndarray,
        mR_count: int,
        mB_count: int,
        friendly_drop_mask: np.ndarray,
        enemy_drop_mask: np.ndarray,
        key_padding_mask: np.ndarray,
        keep_mask: np.ndarray,
        log_prob_unmasked: float = 0.0,
        log_prob_masked: float = 0.0,
        entropy_unmasked: float = 0.0,
        entropy_masked: float = 0.0,
        next_entities: np.ndarray | None = None,
        next_entity_masks: np.ndarray | None = None,
    ) -> None:
        """Store BRMA fields for one agent-timestep."""
        if not self._enabled:
            raise RuntimeError("BRMA rollout storage is disabled")

        T, E, A = self._cfg.num_steps, self._cfg.num_envs, self._cfg.num_agents
        N, D = self._cfg.n_entities, self._cfg.entity_dim

        if not (0 <= step < T and 0 <= env_idx < E and 0 <= agent_idx < A):
            raise IndexError(
                f"step/env_idx/agent_idx out of range: "
                f"({step}, {env_idx}, {agent_idx}) vs ({T}, {E}, {A})")

        for name, arr in [("p", p), ("msoft", msoft), ("mhard", mhard),
                          ("friendly_drop_mask", friendly_drop_mask),
                          ("enemy_drop_mask", enemy_drop_mask),
                          ("key_padding_mask", key_padding_mask),
                          ("keep_mask", keep_mask)]:
            if arr.shape != (N,):
                raise ValueError(
                    f"{name} must have shape ({N},), got {arr.shape}")

        self._arrays["p"][step, env_idx, agent_idx] = p
        self._arrays["msoft"][step, env_idx, agent_idx] = msoft
        self._arrays["mhard"][step, env_idx, agent_idx] = mhard
        self._arrays["mR_count"][step, env_idx, agent_idx] = mR_count
        self._arrays["mB_count"][step, env_idx, agent_idx] = mB_count
        self._arrays["friendly_drop_mask"][step, env_idx, agent_idx] = friendly_drop_mask
        self._arrays["enemy_drop_mask"][step, env_idx, agent_idx] = enemy_drop_mask
        self._arrays["key_padding_mask"][step, env_idx, agent_idx] = key_padding_mask
        self._arrays["keep_mask"][step, env_idx, agent_idx] = keep_mask
        self._arrays["log_prob_unmasked"][step, env_idx, agent_idx] = log_prob_unmasked
        self._arrays["log_prob_masked"][step, env_idx, agent_idx] = log_prob_masked
        self._arrays["entropy_unmasked"][step, env_idx, agent_idx] = entropy_unmasked
        self._arrays["entropy_masked"][step, env_idx, agent_idx] = entropy_masked
        if next_entities is not None:
            if next_entities.shape != (N, D):
                raise ValueError(
                    f"next_entities must have shape ({N}, {D}), got {next_entities.shape}")
            self._arrays["next_entities"][step, env_idx, agent_idx] = next_entities
        if next_entity_masks is not None:
            if next_entity_masks.shape != (N,):
                raise ValueError(
                    f"next_entity_masks must have shape ({N},), got {next_entity_masks.shape}")
            self._arrays["next_entity_masks"][step, env_idx, agent_idx] = next_entity_masks
        self._valid[step, env_idx, agent_idx] = True

    def get_step(self, step: int, env_idx: int, agent_idx: int) -> dict:
        """Return a copy of stored fields for one slot."""
        if not self._enabled:
            raise RuntimeError("BRMA rollout storage is disabled")
        T, E, A = self._cfg.num_steps, self._cfg.num_envs, self._cfg.num_agents
        if not (0 <= step < T and 0 <= env_idx < E and 0 <= agent_idx < A):
            raise IndexError(
                f"step/env_idx/agent_idx out of range: "
                f"({step}, {env_idx}, {agent_idx}) vs ({T}, {E}, {A})")
        if not self._valid[step, env_idx, agent_idx]:
            raise KeyError(
                f"Slot ({step}, {env_idx}, {agent_idx}) is not valid")
        return {
            k: arr[step, env_idx, agent_idx].copy()
            for k, arr in self._arrays.items()
        }

    def summary(self) -> dict:
        """Return a diagnostic summary of stored data."""
        if not self._enabled:
            return {"enabled": False, "valid_count": 0, "total_slots": 0}
        valid = int(self._valid.sum())
        total = int(self._valid.size)
        mean_enemy = 0.0
        mean_friendly = 0.0
        mean_mR = 0.0
        mean_mB = 0.0
        if valid > 0:
            flat_valid = self._valid.ravel()
            flat_ed = self._arrays["enemy_drop_mask"].reshape(total, -1)
            flat_fd = self._arrays["friendly_drop_mask"].reshape(total, -1)
            mean_enemy = float(flat_ed[flat_valid].sum() / valid)
            mean_friendly = float(flat_fd[flat_valid].sum() / valid)
            flat_mR = self._arrays["mR_count"].ravel()
            flat_mB = self._arrays["mB_count"].ravel()
            mean_mR = float(flat_mR[flat_valid].mean())
            mean_mB = float(flat_mB[flat_valid].mean())
        return {
            "enabled": True,
            "valid_count": valid,
            "total_slots": total,
            "mean_mR_count": mean_mR,
            "mean_mB_count": mean_mB,
            "mean_friendly_drop_count": mean_friendly,
            "mean_enemy_drop_count": mean_enemy,
        }
