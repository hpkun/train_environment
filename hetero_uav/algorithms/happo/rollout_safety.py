"""Shared rollout safety helpers for inactive-agent sanitization.

These helpers prevent dead/inactive agents from feeding non-finite or stale
observations into the policy network.  They are used by both the training
runner and the eval runner.
"""
from __future__ import annotations

import numpy as np


def _ctx_str(ctx: dict | None, extra: str = "") -> str:
    if ctx is None:
        return ""
    parts = [
        f"iter={ctx.get('iteration','?')}",
        f"env={ctx.get('env_idx','?')}",
        f"step={ctx.get('total_steps','?')}",
        f"ep={ctx.get('episode_id','?')}",
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


def sanitize_policy_inputs(
    actor_obs: np.ndarray,
    active: np.ndarray,
    critic_state: np.ndarray | None = None,
    rnn_hidden: np.ndarray | None = None,
    context: dict | None = None,
) -> dict:
    """Sanitize actor observations and hidden state before policy.act.

    Inactive (active <= 0.5) rows are zeroed.  Active rows are checked for
    finite values and raise an error if any NaN/Inf is found.
    """
    ctx = context or {}
    actor_obs = np.asarray(actor_obs, dtype=np.float32)
    active = np.asarray(active, dtype=np.float32).reshape(-1)
    critic_state = np.asarray(critic_state, dtype=np.float32) if critic_state is not None else None

    inactive_rows = active <= 0.5
    inactive_count = int(inactive_rows.sum())
    active_rows = ~inactive_rows

    diag: dict = {
        "inactive_count": inactive_count,
        "inactive_nonfinite_count": 0,
        "inactive_critic_nonfinite_count": 0,
        "active_obs_nonfinite_count": 0,
        "active_hidden_nonfinite_count": 0,
    }

    # ---- Inactive rows: zero out (NaN/Inf allowed — zeroed safely) ----
    if inactive_count > 0:
        inactive_nonfin = int((~np.isfinite(actor_obs[inactive_rows])).sum())
        diag["inactive_nonfinite_count"] = inactive_nonfin
        actor_obs[inactive_rows] = 0.0
        if rnn_hidden is not None:
            rnn_hidden[inactive_rows] = 0.0

    # ---- Centralized critic chunks mirror fixed-width actor observations ----
    if critic_state is not None:
        critic_flat = critic_state.reshape(-1)
        actor_dim = int(actor_obs.shape[-1])
        if actor_obs.ndim != 2 or actor_dim <= 0 or critic_flat.size % actor_dim != 0:
            raise ValueError(
                f"critic_state cannot be partitioned into actor chunks: "
                f"{_ctx_str(ctx)} critic_dim={critic_flat.size} actor_dim={actor_dim}"
            )
        critic_chunks = critic_flat.reshape(-1, actor_dim)
        for row_idx in range(critic_chunks.shape[0]):
            inactive_or_padding = row_idx >= active.shape[0] or inactive_rows[row_idx]
            if inactive_or_padding:
                diag["inactive_critic_nonfinite_count"] += int(
                    (~np.isfinite(critic_chunks[row_idx])).sum()
                )
                critic_chunks[row_idx] = 0.0

        bad_active_chunks = []
        for row_idx in np.where(active_rows)[0]:
            row_fin = np.isfinite(critic_chunks[row_idx])
            if not row_fin.all():
                bad_cols = np.where(~row_fin)[0]
                bad_active_chunks.append(
                    f"row={int(row_idx)} cols={[int(c) for c in bad_cols[:8]]}"
                )
        if bad_active_chunks:
            raise ValueError(
                f"Non-finite critic_state for active agent: "
                f"{_ctx_str(ctx)} {'; '.join(bad_active_chunks[:4])}"
            )

    # ---- Active actor_obs: must be finite ----
    if active_rows.any():
        obs_fin = np.isfinite(actor_obs[active_rows])
        if not obs_fin.all():
            bad_detail = []
            active_indices = np.where(active_rows)[0]
            for i, row_idx in enumerate(active_indices):
                row_fin = obs_fin[i]
                if not row_fin.all():
                    bad_cols = np.where(~row_fin)[0]
                    diag["active_obs_nonfinite_count"] += len(bad_cols)
                    bad_detail.append(
                        f"row={int(row_idx)} cols={[int(c) for c in bad_cols[:8]]}"
                    )
            raise ValueError(
                f"Non-finite actor_obs for active agent: "
                f"{_ctx_str(ctx)} "
                f"{'; '.join(bad_detail[:4])}"
            )
        if critic_state is not None and not np.isfinite(critic_state).all():
            bad_positions = np.where(~np.isfinite(critic_state))
            raise ValueError(
                f"Non-finite critic_state: "
                f"{_ctx_str(ctx)} "
                f"bad_idx={[int(p[0]) for p in zip(*bad_positions)][:8]}"
            )

    # ---- Active rnn_hidden: must be finite (inactive already zeroed) ----
    if rnn_hidden is not None and active_rows.any():
        hid_fin = np.isfinite(rnn_hidden[active_rows])
        if not hid_fin.all():
            bad_detail = []
            active_indices = np.where(active_rows)[0]
            for i, row_idx in enumerate(active_indices):
                row_fin = hid_fin[i]
                if not row_fin.all():
                    bad_cols = np.where(~row_fin)[0]
                    diag["active_hidden_nonfinite_count"] += len(bad_cols)
                    bad_detail.append(
                        f"row={int(row_idx)} hidden_idx={[int(c) for c in bad_cols[:8]]}"
                    )
            raise ValueError(
                f"Non-finite rnn_hidden for active agent: "
                f"{_ctx_str(ctx)} "
                f"{'; '.join(bad_detail[:4])}"
            )

    return {
        "actor_obs": actor_obs,
        "critic_state": critic_state,
        "rnn_hidden": rnn_hidden,
        "diagnostics": diag,
    }


def zero_inactive_actions(actions: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Zero out actions for inactive agents (active <= 0.5)."""
    actions = np.asarray(actions, dtype=np.float32)
    active = np.asarray(active, dtype=np.float32).reshape(-1)
    inactive_rows = active <= 0.5
    if inactive_rows.any():
        actions = actions.copy()
        actions[inactive_rows] = 0.0
    return actions


def zero_inactive_hidden(hidden: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Zero out recurrent hidden state for inactive agents."""
    hidden = np.asarray(hidden, dtype=np.float32)
    active = np.asarray(active, dtype=np.float32).reshape(-1)
    inactive_rows = active <= 0.5
    if inactive_rows.any():
        hidden = hidden.copy()
        hidden[inactive_rows] = 0.0
    return hidden
