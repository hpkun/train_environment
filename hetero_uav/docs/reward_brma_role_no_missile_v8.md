# brma_role_no_missile_reward_v8

## Purpose

`brma_role_no_missile_reward_v8` is a minimal diagnostic reward for the
heterogeneous MAV/UAV setting.  It keeps the BRMA-MAPPO active reward trunk and
only adapts the MAV role by removing the attack-situation term from the
missile-less MAV.

This mode is intended to test whether the policy can learn from a clean BRMA
flight/situation/terminal signal without TAM-style event shaping or missile
process rewards.

## Active Reward

The parent BRMA reward computes weighted components:

- `r_pitch`
- `r_roll`
- `r_alt`
- `r_bound`
- `r_vel`
- `r_adv`
- `r_end`

Those component values are already weighted by the parent environment.  This
mode does not multiply them again.

Attack UAV active reward:

```text
BRMA flight + BRMA situation + BRMA terminal
```

MAV active reward:

```text
BRMA flight + BRMA terminal
```

The MAV uses the same parent BRMA reward and then subtracts the already-weighted
`r_adv` situation term.  The removed value is logged as
`brma_role_removed_situation`.

## Explicitly Excluded From Active Reward

The following are not active reward terms in this mode:

- TAM MAV safety/support/event rewards
- TAM UAV event rewards
- fire, launch, lock, guided, dodge, near-hit, or shared-track rewards
- missile warning or missile threat rewards
- role-weighted terminal
- loss-fraction terminal
- v7 binary altitude/speed envelope rewards
- asymmetric positive/negative scaling
- low-speed exploit penalty

## Diagnostics

The mode logs:

- `brma_role_no_missile_total`
- `brma_role_no_missile_active`
- `brma_role_active_brma_flight`
- `brma_role_active_brma_situation`
- `brma_role_active_brma_terminal`
- `brma_role_removed_situation`
- `brma_role_situation_active`
- `brma_role_removed_situation_is_weighted`
- `brma_role_is_mav`

These fields are diagnostics only.  They do not call TAM v7 helpers and do not
mutate TAM death/team-credit state.

## Configs

- `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_brma_role_no_missile_reward_v8.yaml`
- `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_dynamics_f22_visual_mav_brma_role_no_missile_reward_v8.yaml`

Both keep F16 dynamics with F22 MAV visual label and `mav.num_missiles = 0`.
