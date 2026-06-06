# Hetero Minimal Reward Overlay (minimal_v1)

## Purpose

`minimal_v1` is an optional hetero reward overlay that adds small
role-aware shaping on top of the BRMA legacy reward.  Default mode is
`brma_legacy` — the overlay must be explicitly enabled via config.

## Why Minimal

- Preserves BRMA comparability
- Does not change termination, missile, evasion, action, or PID
- MAV survival reward is small enough not to overwhelm terminal outcome
- Shared-track bonuses are capped

## Components

| Component | When | Magnitude |
|---|---|---|
| r_mav_survival | MAV alive each step | +0.005 |
| r_mav_death | MAV death detected | -2.0 |
| r_mav_support | MAV provides shared tracks to attack UAV | +0.01 × count (capped 0.05) |
| r_shared_track_used | attack UAV uses MAV shared track | +0.005 × count (capped 0.02) |
| r_attack_kill_bonus | attack UAV kills blue | TODO: 0.0 currently |

## How to Enable

```yaml
hetero_reward_mode: minimal_v1
```

## How to Disable

Omit `hetero_reward_mode` or set it to `brma_legacy`.

## Relation to Papers

- BRMA base is preserved
- TAM-HAPPO MAV support / UAV attack concept expressed via minimal role-aware shaping
- Not a full reproduction
