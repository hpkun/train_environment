# Paper-Aligned Baseline Plan

## Why switch from rule_nearest to brma_rule

`rule_nearest` is a simplified opponent that only uses `enemy_states` for
nearest-target selection. `brma_rule` delegates to the parent project's
`rule_based_agent.py`, the BRMA-MAPPO original blue policy with a
four-layer state machine (search, combat, cruise, climb), boundary
patrol, altitude safety, and coordinated target deconfliction.

Diagnostics show `brma_rule` actually fires missiles (vs 0 for
rule_nearest in 400-step tests), has lower heading error, and engages
more like a paper-aligned opponent.

## Why use no_mav_trim config

MAV pitch trim=0.10 was introduced as an A-4 stability carryover. It is
not part of the BRMA-MAPPO paper. It pushes MAV effective pitch from
0.976 to saturated 1.0. Removing it is the minimal paper-aligned
isolation step.

## Why keep brma_legacy reward weights unchanged

`brma_legacy` uses BRMA-MAPPO paper Table 4 weights (r_pitch=0.01,
r_roll=0.002, etc.). Changing them would deviate from the paper baseline.

## Why not role-conditioned or attention now

The shared MLP baseline must be stable before adding architectural
complexity. The paper-aligned config + brma_rule opponent is the minimal
fair comparison point.

## Observation Metrics

- red_elimination_win_rate
- red_timeout_alive_advantage_rate
- blue_elimination_win_rate
- MAV survival
- red0 action saturation
- blue missile launches
- kill_death_ratio

## 50k is a pilot, not a final claim
