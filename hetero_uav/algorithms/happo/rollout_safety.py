"""Shared rollout safety helpers for inactive-agent sanitization.

These helpers prevent dead/inactive agents from feeding non-finite or stale
observations into the policy network.  They are used by both the training
runner and the eval runner.
"""
from __future__ import annotations

import numpy as np


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
        "active_nonfinite_count": 0,
    }

    # ---- Inactive rows: zero out (NaN/Inf allowed → zeroed safely) ----
    if inactive_count > 0:
        inactive_nonfin = int((~np.isfinite(actor_obs[inactive_rows])).sum())
        diag["inactive_nonfinite_count"] = inactive_nonfin
        actor_obs[inactive_rows] = 0.0
        if rnn_hidden is not None:
            rnn_hidden[inactive_rows] = 0.0

    # ---- Active rows: must be finite ----
    if active_rows.any():
        act_fin = np.isfinite(actor_obs[active_rows]).all()
        crit_fin = np.isfinite(critic_state).all() if critic_state is not None else True
        if not act_fin:
            bad_rows = np.where(active_rows)[0]
            for row_idx in bad_rows:
                row = actor_obs[row_idx]
                if not np.isfinite(row).all():
                    bad_cols = np.where(~np.isfinite(row))[0]
                    diag["active_nonfinite_count"] += len(bad_cols)
            raise ValueError(
                f"Non-finite actor_obs for active agent: "
                f"iter={ctx.get('iteration','?')} env={ctx.get('env_idx','?')} "
                f"step={ctx.get('total_steps','?')} ep={ctx.get('episode_id','?')} "
                f"bad_rows={[int(r) for r in bad_rows]} "
                f"first_bad_cols={[int(c) for c in bad_cols[:8]]}"
            )
        if not crit_fin:
            bad_positions = np.where(~np.isfinite(critic_state))
            raise ValueError(
                f"Non-finite critic_state: "
                f"iter={ctx.get('iteration','?')} env={ctx.get('env_idx','?')} "
                f"step={ctx.get('total_steps','?')} "
                f"first_bad_idx={[int(p[0]) for p in zip(*bad_positions)][:8]}"
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
