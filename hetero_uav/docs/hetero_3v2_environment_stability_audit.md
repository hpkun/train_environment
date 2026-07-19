# Heterogeneous 3V2 Environment Stability Audit

## Purpose

This audit verifies reproducibility, reset isolation, finite numerical state,
death/active-mask accounting, termination semantics and missile-event
accounting for the frozen formal heterogeneous 3V2 environment. It does not
evaluate Reward V5 learnability and does not use win rate as a PASS condition.

The default environment is:

```text
uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2_reward_v5.yaml
```

The audit is read-only with respect to environment state transitions. It does
not modify Reward V5, dynamics, PID, observations, fire control, missiles,
`PaperGreedyOpponent`, termination logic or Pure HAPPO.

## Ubuntu CLI

```bash
cd /path/to/train_environment/hetero_uav
conda activate brmamappo

python -u scripts/audit_hetero_3v2_environment_stability.py \
  --config uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2_reward_v5.yaml \
  --output-dir outputs/hetero_3v2_environment_stability \
  --seed 3000 \
  --deterministic-cases 10 \
  --reset-episodes 30 \
  --rule-episodes 30 \
  --zero-episodes 10 \
  --random-episodes 20
```

Resume an interrupted audit with the identical command plus `--resume`.
Configuration path, seed, requested case counts, environment contracts,
dimensions, tolerances and the audit-only step cap must match `audit_meta.json`.
The runner rejects a non-empty output directory unless `--resume` is explicit.

`--max-case-steps` is an audit-only smoke control. Its default is zero, which
uses the full environment `max_steps`; it must remain zero for the formal
audit.

Regenerate reports without running the environment:

```bash
python -u scripts/report_hetero_3v2_environment_stability.py \
  --input-dir outputs/hetero_3v2_environment_stability
```

## Reproducibility contract

Initial perturbations reuse
`scripts/hetero_3v2_v2_audit_common.py::perturbation`. Stable JSON signatures
must be unique within each requested scenario group. Deterministic run A and B
share the same seed, perturbation and red rule policy. Discrete fields are
compared exactly. Continuous fields use:

```text
rtol = 1e-7
atol = 1e-8
```

Deterministic A/B runs each use an independent fresh environment, avoiding an
asymmetric first-construction versus reload comparison. Reset/reload stability
is audited separately by consecutive reset cases. Discrete alive state, masks,
targets, gates, events, deaths and episode boundaries remain exact. Arrays are
compared numerically with `numpy.allclose`, never by string or hash.

## Resumable artifacts

- `environment_stability_raw.jsonl`: one flushed record per completed case.
- `environment_stability_failures.jsonl`: one flushed record per failure.
- `environment_stability_episodes.csv`: compact case summary.
- `environment_stability_audit.json`: structured aggregate result.
- `environment_stability_report.md`: concise human-readable report.
- `audit_meta.json`: immutable audit and environment contract.

Case IDs are stable: `deterministic_000_run_a`,
`deterministic_000_run_b`, `reset_000`, `rule_000`, `zero_000` and
`random_000`. Resume strictly parses JSONL, rejects corrupt lines and duplicate
case IDs, and skips completed cases. A completed run A trace is sufficient to
resume and compare a missing run B.

## Report checks

- `DETERMINISTIC_REPLAY`
- `RESET_ISOLATION`
- `NUMERICAL_FINITE`
- `DYNAMICS_BOUNDARY`
- `ACTIVE_MASK_MONOTONICITY`
- `TERMINATION_CONSISTENCY`
- `DEATH_ACCOUNTING`
- `MISSILE_EVENT_ACCOUNTING`
- `RULE_POLICY_RUNTIME`
- `ZERO_ACTION_RUNTIME`
- `RANDOM_ACTION_RUNTIME`

Each check reports `PASS`, `FAIL` or `N/A` with sample and failure counts.
Missing samples remain `N/A`; they are never converted into PASS. Any failed
core consistency check makes the overall result FAIL. A low win rate, no red
win, no launch or no hit does not by itself fail runtime stability.

## Interpretation boundary

A PASS supports only that the frozen implementation was stable, internally
consistent and reproducible under the audited cases. It does not establish
policy quality, Reward V5 alignment, convergence or combat effectiveness.
