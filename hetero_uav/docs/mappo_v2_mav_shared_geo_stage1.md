# MAPPO V2 MAV-Shared-Geo Stage 1

This stage is an evaluation-diagnostics runner for the existing plain MAPPO
baseline with `HeteroObsAdapterV2`.

It is not HAPPO, attention, GRU, role-aware MAPPO, or a formal win-rate
experiment.

## Dimensions

- Observation mode: `mav_shared_geo`
- Adapter: `HeteroObsAdapterV2`
- Actor observation dimension: `96`
- Critic state dimension: `480`

## Short Trainability Diagnostics

`scripts/diagnose_mappo_v2_trainability.py` calls
`scripts/train_mappo_baseline.py` with:

- `--obs-adapter-version v2`
- `--max-steps`
- V2 shared-geo config

A short diagnostic must complete at least one episode before its return values
can be inspected even informally. If `episodes_completed == 0`, the result only
means the rollout was too short or `max_steps` was too long for that diagnostic
window. It must not be interpreted as a learning trend.

## Zero-Shot Smoke

`scripts/diagnose_mappo_v2_zero_shot_smoke.py` and
`scripts/eval_mappo_zero_shot.py` run a saved V2 model over multiple
`mav_shared_geo` configs and multiple episodes.

These smoke checks verify:

- model/meta compatibility;
- actor and critic dimensions;
- environment reset/step;
- obs / actor_obs / critic_state / action / return NaN checks.

They are not formal zero-shot success claims.

## Formal Experiment Requirements

Formal experiments still require:

- multiple seeds;
- multiple episodes;
- fixed training step budgets;
- consistent opponent policy;
- baseline comparisons.

This stage does not change action, missile, evasion, reward, termination, PID,
aircraft XML, or MAV GCAS.
