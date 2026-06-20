# TAM Categorical Trainer Semantics and Diagnostics Design

## Scope

Work stays inside `tam_uav`. The change fixes categorical HAPPO optimizer ownership,
makes the requested and effective policy architecture explicit, adds read-only
environment and missile diagnostics, adds trend analysis, and documents the formal
4D categorical route. Reward, missile dynamics, aircraft data, initial states,
target selection, and action execution remain unchanged.

## Trainer semantics

The actor is split into three disjoint ownership groups: encoder plus GRU are owned
by an optional `shared_actor_opt`; the MAV and UAV heads are owned by `mav_opt` and
`uav_opt`. Each role update clears the shared and role-head gradients, computes one
loss, clips shared and head gradients separately, steps the role head, then steps the
shared optimizer. Metrics preserve the aggregate actor norm and expose shared,
per-head, and critic norms.

## Architecture identity and checkpoints

Categorical environments always use the effective architecture
`tam_categorical_recurrent`. `brma_recurrent_masked` remains a command-line alias;
metadata records both requested and effective names and whether the alias was used.
Explicit categorical requests record no alias. Evaluation loads categorical
checkpoints from the effective architecture and rejects metadata that identifies a
continuous distribution or a conflicting effective architecture. Legacy continuous
routes retain their existing behavior.

## Diagnostics and trend analysis

Airborne initialization diagnostics observe reset FCS checkpoints and run fixed
neutral flight plus three formal episodes without changing initialization. Missile
threat diagnostics compare deterministic, stochastic, no-blue-missile, and formal
blue-missile scenarios and aggregate launches, hits, warnings, post-warning outcomes,
launch geometry, survival, and termination reasons. Trend analysis consumes existing
CSV and rich logs, reports staged and rolling metrics, and assigns one of the requested
A/B/C/D stage decisions using finite, documented rules.

## Verification

All behavior changes are test-driven. Verification uses the `brmamappo` CUDA Python,
then runs the requested test set, both environment audits, a 2k smoke, a 200k trend
probe, and the trend analyzer. If CUDA becomes unavailable, the same commands fall
back to CPU. No 1M or larger run is started.

