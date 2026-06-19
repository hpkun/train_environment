# role_v1 Reward

## Purpose

`role_v1` is a heterogeneous MAV/UAV role-aware reward ablation.  It adds
role-specific shaping on top of `brma_legacy` to make MAV survival and
UAV engagement visible in the training signal.

`brma_legacy` remains the baseline reward.  `role_v1` is an ablation —
not the default and not a method module.

## Design Principles

- Does **NOT** modify termination, missile launch, missile dynamics,
  action space, PID, or aircraft XML
- Adds small per-step shaping and one-time event bonuses/penalties
- Scales are kept small to avoid critic explosion

## MAV Components

| Component | Trigger | Magnitude |
|---|---|---|
| `r_role_mav_survival` | MAV alive each step | +0.01 |
| `r_role_mav_death` | MAV first death (any cause) | -10.0 (once) |
| `r_role_mav_support` | MAV sees alive enemies or UAVs use shared tracks | ≤ +0.05/step |
| `r_role_mav_team_contribution` | Red UAV kills + MAV alive | ≤ +5.0/step |

## UAV Components

| Component | Trigger | Magnitude |
|---|---|---|
| `r_role_uav_attack_window` | UAV has enemy in engagement geometry | ≤ +0.03/step |
| `r_role_uav_kill_bonus` | UAV kills blue aircraft | +8.0/kill (capped +10) |
| `r_role_uav_death_penalty` | UAV first death (any cause) | -5.0 (once) |
| `r_role_uav_missile_warning` | Incoming missile detected | -0.005/step |

## Constraints

- No full dodge reward (actor lacks missile entity observation)
- No modification to termination logic
- No modification to missile launch/dynamics
- This is an ablation, not a paper claim
