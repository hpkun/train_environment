# Final Paper Experiment Plan

This plan is for the final paper-oriented experiment stage. It does not define
new environment mechanics, reward terms, missile logic, or aircraft changes.

## Experiment Matrix

| ID | Experiment | Policy Arch | Mask | Total Env Steps | Purpose |
|---|---|---|---|---:|---|
| A | `flat_baseline_long` | `flat` | none | existing run | Existing weak MLP baseline: `outputs/full_10m_normal_geometry_max1000_env1`. |
| B | `brma_recurrent_masked_nomask_500k_probe` | `brma_recurrent_masked` | dead/padding/observed mask only | 500000 | Primary BRMA-style recurrent entity-attention probe with safe PPO log-prob replay. |
| C | `brma_recurrent_masked_biased_500k_probe` | `brma_recurrent_masked` | biased mask forward path | 500000 | Optional diagnostic only; not a full BRMA biased-mask objective. |

If time is limited, run B first and postpone C. Do not use
`--brma-random-scale-mask` for the main experiment.

## Shared Settings For B/C

- config: `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml`
- rollout length: `256`
- num envs: `1`
- max steps: `1000`
- device: `cuda`
- eval during training: enabled
- eval interval steps: `50000`
- train eval episodes: `5`
- no imitation/pretrain/heading loss
- no reward or environment changes
- no `--brma-random-scale-mask`

## Evaluation

Each run should be evaluated on:

- 3v2 seen evaluation;
- 5v4 fixed-capacity zero-shot evaluation.

Core metrics:

- `red_win_rate`
- `blue_win_rate`
- `draw_rate`
- `timeout_rate`
- `red_elimination_win_rate`
- `red_timeout_alive_advantage_rate`
- `red_missiles_fired_mean`
- `red_missile_hits_mean`
- `blue_dead_mean`
- `mav_survival_rate`
- reward curve
- win-rate curve
- trajectory plot
- attitude plot if available

## Interpretation Boundary

The masked policy can be used as a BRMA-style architecture in method figures
because it uses entity construction, attention, GRU, role-wise actor heads, and
dead/padding/observed masks. It should not be claimed as a full BRMA-MAPPO
reproduction because the complete mask objective and strict BRMA training loop
are simplified. Existing random-scale-mask runs are diagnostic unsafe-mask runs:
they re-sampled masks between rollout and PPO update and are not final main
results.
