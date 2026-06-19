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
| r_mav_death | MAV first death detected (any cause: crash, missile kill, etc.) | -2.0 (once per episode) |
| r_mav_support | MAV provides shared tracks to attack UAV | +0.01 × count (capped 0.05) |
| r_shared_track_used | attack UAV uses MAV shared track | +0.005 × count (capped 0.02) |
| r_attack_kill_bonus | attack UAV kills blue | TODO: 0.0 currently |

### r_mav_death — First-Death Detection

`r_mav_death` uses a first-death detection mechanism (`self._mav_death_penalized`):

- On reset (MAV alive): the penalized flag is cleared automatically.
- When MAV transitions from alive → dead (any cause: crash, GCAS,
  missile kill): a one-time penalty of -2.0 is applied.
- Subsequent steps where MAV remains dead: no additional penalty.
- This covers crash-style death AND missile-kill death, unlike the
  old `_crashed_this_step` check which only covered crashes.

### r_mav_support — One-Step-Lag

`r_mav_support` uses the **previous decision-frame observation cache**
(`self._last_step_obs`). This is a **one-step-lag** pattern:

- **Reset**: `_last_step_obs` is cleared, then seeded with the reset
  observation (for `minimal_v1` mode). This means the first step after
  reset uses the reset observation as the "previous decision-frame"
  observation, enabling support reward from step 1.
- **Step**: After each call to `super().step()`, `_last_step_obs` is
  updated with the new observation.
- Tests should not assume `r_mav_support > 0` on step 1 (it depends on
  whether the reset observation already contains shared tracks).

### Reset / Cache Semantics

The `HeteroUavCombatEnv.reset()` override ensures clean episode boundaries:

1. **Clear stale cache**: `_last_step_obs = {}` — prevents
   cross-episode support reward leakage from the previous episode.
2. **Reset death flag**: `_mav_death_penalized = False` — ensures a
   fresh first-death detection per episode.
3. **Seed cache** (minimal_v1 only): `_last_step_obs = obs` — caches
   the reset observation so the first reward computation has a valid
   previous decision-frame reference.
4. **brma_legacy**: `_last_step_obs` is NOT seeded — reward behavior
   is unchanged.

Termination remains unchanged.

## Protocol Review

The `minimal_v1` overlay is an **optional** role-aware shaping layer.
It should NOT automatically become the main protocol default.  The
protocol review (`docs/hetero_environment_protocol_review.md`) documents
the decision: `brma_legacy` remains the default baseline reward for
paper-aligned and balanced configs.

## How to Enable

```yaml
hetero_reward_mode: minimal_v1
```

## How to Disable

Omit `hetero_reward_mode` or set it to `brma_legacy`.  All main configs
explicitly declare `hetero_reward_mode: "brma_legacy"` for protocol
clarity, even though `brma_legacy` is already the default.

## Configs

| Config | hetero_reward_mode | Purpose |
|---|---|---|
| hetero_mav_shared_geo_3v2.yaml | brma_legacy (default) | Paper-aligned baseline |
| hetero_mav_shared_geo_3v2_reward_minimal.yaml | minimal_v1 | Minimal overlay, paper-aligned range |
| hetero_diagnostic_close_range_mav_shared_geo_3v2_reward_minimal.yaml | minimal_v1 | Diagnostic, close-range, overlay |

## Relation to Papers

- BRMA base is preserved
- TAM-HAPPO MAV support / UAV attack concept expressed via minimal role-aware shaping
- Not a full reproduction
