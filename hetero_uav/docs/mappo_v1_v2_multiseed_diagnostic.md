# MAPPO V1 vs V2 Multi-Seed Diagnostic

## Purpose

Multi-seed, medium-length diagnostic to observe V1/V2 training pipeline
stability.  This is **not** a formal experiment and does not produce
win-rate conclusions.

The current project priority is MAPPO baseline environment stability. The V1/V2
comparison is an auxiliary diagnostic and does not replace the mainline V2
environment stability validation in
`docs/mappo_baseline_environment_stability.md`.

## V1/V2 Settings

| Version | Observation | Actor Dim | Critic Dim |
|---|---|---|---|
| v1 | brma_sensor | 140 | 700 |
| v2 | mav_shared_geo | 96 | 480 |

## Config Scope

Paper-aligned configs remain available as TAM-HAPPO scenario references:

- `hetero_paper_3v2_mav_2uav_vs_2uav.yaml`
- `hetero_paper_5v4_mav_4uav_vs_4uav.yaml`

The current diagnostic defaults use balanced red/blue counts:

- V1 train: `hetero_balanced_brma_sensor_3v3.yaml`
- V1 eval: `hetero_balanced_brma_sensor_3v3.yaml`,
  `hetero_balanced_brma_sensor_4v4.yaml`
- V2 train: `hetero_balanced_mav_shared_geo_3v3.yaml`
- V2 eval: `hetero_balanced_mav_shared_geo_3v3.yaml`,
  `hetero_balanced_mav_shared_geo_4v4.yaml`

Balanced configs are preferred for the current stability gate because they avoid
red/blue count asymmetry and make later MAPPO-vs-method comparisons cleaner.

## Diagnostic Protocol

- seeds: [0, 1] (configurable)
- iterations: 50
- rollout_length: 32
- max_steps: 128
- eval_episodes: 3
- opponent: rule_nearest

## Outputs

- `train_summary.csv` — per-seed training metrics
- `eval_summary.csv` — per-config evaluation metrics
- `aggregate_summary.json` — mean/std across seeds, grouped by version and config
- per-seed `train_log.csv` and checkpoints

## Role in the Pipeline

This diagnostic is a stability check before method innovation.  If V2
is stable (no NaN, reasonable variance) across multiple seeds, it can
serve as the main experimental observation mode.  Current results do
not prove V2 is better than V1.

Do not enter attention, HAPPO, GRU, or role-aware algorithm work before the
MAPPO baseline environment stability validation is clean.
Entity attention remains only a possible later method direction after this
baseline stability gate; it is not part of the current diagnostic stage.

## Caveats

- This is a diagnostic, **not** a win-rate experiment.
- 50 iterations is too short for convergence.
- 2 seeds are insufficient for statistical conclusions.
- Formal experiments need larger training budget, more seeds,
  repeated evaluation, and proper baselines.
