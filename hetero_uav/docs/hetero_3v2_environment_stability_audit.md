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
  --random-episodes 20 \
  --report-every-cases 5
```

Resume an interrupted audit with the identical command plus `--resume`.
Configuration path, seed, requested case counts, environment contracts,
dimensions, tolerances, the audit-only step cap and report interval must match
`audit_meta.json`.
The runner rejects a non-empty output directory unless `--resume` is explicit.

`--max-case-steps` is an audit-only smoke control. Its default is zero, which
uses the full environment `max_steps`; it must remain zero for the formal
audit. When a positive value below `env.max_steps` is used, reaching it records
`audit_step_limit_reached=true` and `environment_episode_complete=false`; it
does not require the environment to emit timeout truncation. A formal case that
reaches `env.max_steps` must instead finish with `truncated=true`,
`team_done=true`, `outcome=draw` and `end_reason=timeout`.

Raw case and failure JSONL records are appended and flushed after every case.
The aggregate JSON, CSV and Markdown reports are rebuilt every
`--report-every-cases` completed cases, and always on normal completion,
KeyboardInterrupt or an unhandled exception.

Regenerate reports without running the environment:

```bash
python -u scripts/report_hetero_3v2_environment_stability.py \
  --input-dir outputs/hetero_3v2_environment_stability
```

## Reproducibility contract

Initial perturbations reuse
`scripts/hetero_3v2_v2_audit_common.py::perturbation`. Stable JSON signatures
must be unique within each requested scenario group. Deterministic run A uses
`PaperGreedyOpponent` and records the complete red action sequence. Run B uses
an independent fresh environment with the same seed and perturbation, and
strictly replays run A's recorded actions without calling the rule policy.
The replay sequence is validated as an input contract before environment
outputs are compared. Discrete output fields are compared exactly. Continuous
output fields use:

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

- `CASE_RUNTIME_INTEGRITY`
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
core consistency check makes the overall result FAIL. Runtime exceptions are
fail-closed: deterministic, reset, rule, zero and random cases fail their
corresponding fixed checks, and every unmatched failure is surfaced by
`CASE_RUNTIME_INTEGRITY`. Failure counts are JSONL failure-record counts, not
counts of cases containing a failure. A low win rate, no red win, no launch or
no hit does not by itself fail runtime stability.

The frozen hard-state checks are exact: `numeric_anomaly` covers non-finite
position, velocity, RPY or geodetic state; `crash` requires altitude below
100 m; and `out_of_zone` requires horizontal distance above 50 km or altitude
above 10 km. Missile-hit and other permitted death reasons are not reclassified
as boundary failures.

## Interpretation boundary

A PASS supports only that the frozen implementation was stable, internally
consistent and reproducible under the audited cases. It does not establish
policy quality, Reward V5 alignment, convergence or combat effectiveness.
