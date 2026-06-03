# MAPPO V1 vs V2 Multi-Seed Diagnostic

## Purpose

Multi-seed, medium-length diagnostic to observe V1/V2 training pipeline
stability.  This is **not** a formal experiment and does not produce
win-rate conclusions.

## V1/V2 Settings

| Version | Observation | Actor Dim | Critic Dim |
|---|---|---|---|
| v1 | brma_sensor | 140 | 700 |
| v2 | mav_shared_geo | 96 | 480 |

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

Next method innovation priority: **entity attention encoder**, not HAPPO,
because:
- The research target is composition zero-shot transfer
- Variable entity counts and masks are the core challenge
- HAPPO primarily addresses heterogeneous policy updates, not the
  current bottleneck

See `docs/entity_attention_method_plan.md` for the detailed design.

## Caveats

- This is a diagnostic, **not** a win-rate experiment.
- 50 iterations is too short for convergence.
- 2 seeds are insufficient for statistical conclusions.
- Formal experiments need larger training budget, more seeds,
  repeated evaluation, and proper baselines.
