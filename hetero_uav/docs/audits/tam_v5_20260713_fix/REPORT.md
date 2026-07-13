# TAM-HAPPO paper formula v5 deterministic audit

## Baseline and scope

- Expected baseline commit: `53c4e70dcac31af7a5f764429a5620e71962025a`.
- Only `hetero_uav` was changed. No Git command was executed.
- No 200K run was started.
- Frozen: UAV `10:10:15:10:30`, UAV events `+200/-200/-100`, global scale `1/200`, MAV Safety `0.5:0.3:0.2`, MAV Support `0.6:0.4`, target assessment `0.35:0.25:0.20:0.20`, PPO/GAE/network/action/observation/PID/XML/missile/blue-rule/formal geometry contracts.

## Deterministic implementation fixes

1. No-target speed now returns zero with `speed_target_valid=0`; an invalid speed with a valid target still returns the paper three-segment penalty.
2. MAV awareness used by reward is the paper AO sum, one contribution per visible alive blue aircraft. Marginal shared-not-direct awareness is retained as log-only diagnostics.
3. Launch, hit, and kill are separate events. Hit requires a known launch; kill requires production kill/death attribution. Missing attribution is recorded and never guessed.
4. MAV team credit uses real kills only and is zero on the MAV death transition and after death.
5. YAML `global_reward_scale` is a strict `0.005` contract.
6. All `V5_TRAIN_FIELDS` are exported to summary/full rich `train_metrics.csv` with alive-before role denominators.
7. `v5_identity_max_abs` is a rollout maximum, not a divided mean.
8. Episode aggregation uses sum for reward terms, last value for terminal/final/unique totals, and max for identity error.
9. Launch gate diagnostics now preserve a real launch frame before lock reset/cooldown mutation. This is diagnostic-only and does not change fire control.

## Full rich smoke

- Output: `outputs/tam_v5_rich_smoke_512_20260713_fix`.
- Status: normal, 512 steps, no NaN/non-finite value.
- `rich_log_mode=full`; train metrics, reward components, episode components, missile events, target diagnostics, and aircraft time series exist.
- All ten v5 train fields are present, finite, and non-empty. Last identity error is below `1e-8`.

## Read-only event-enriched evaluation

Output: `outputs/tam_v5_paper_formula_evaluation_20260713_fix`.

| Source | Episodes | MAV alive | Blue loss | Launch | Hit | Kill | Shared-only kill |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| fixed_straight | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3_seed0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3_seed1 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3_seed2 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| historical event checkpoint | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| paper_pursuit_diagnostic | 20 | 2 | 37 | 41 | 37 | 20 | 0 |
| mav_shared_diagnostic_fixture | 20 | 16 | 40 | 46 | 40 | 36 | 36 |

Totals: 160 terminal episodes, zero censored episodes, 87 launches, 77 hits, 56 reliable kills, and 36 reliable shared-only kills. All numeric collection fields are finite; maximum reward identity error is `2.220446049250313e-16`.

The last two sources are evaluation diagnostics only. They are not training policies and are not formal environment performance results.

### Launch gate funnel

| Source | Track rate | Range rate | ATA rate | TA rate | Geometry rate | Lock mature rate | Actual launch rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3_seed0 | 0.6932 | 0.3394 | 0.0893 | 0.0010 | 0 | 0 | 0 |
| v3_seed1 | 1.0000 | 0.4573 | 0.1871 | 0 | 0 | 0 | 0 |
| v3_seed2 | 1.0000 | 0.3375 | 0.1424 | 0 | 0 | 0 | 0 |
| paper_pursuit_diagnostic | 0.8994 | 0.7192 | 0.5757 | 0.3334 | 0.0484 | 0.0409 | 0.0025 |
| mav_shared_diagnostic_fixture | 0.4489 | 0.0794 | 0.4219 | 0.4035 | 0.0093 | 0.0038 | 0.0038 |

## Unknown constants

- Status: `TAM_V5_UNKNOWN_CONSTANTS_FEASIBLE`.
- Complete episodes: 160.
- Feasible points in the project constraint domain: 40,260.
- Generated candidates: lower feasible bound, interval midpoint, upper feasible bound.
- The search domain is a project constraint domain, not a paper constant range.

## 20,480-step probes

All nine required candidate/seed runs completed normally. All reached 20,480 steps, were finite, saved latest checkpoints, and had identity maxima between `2.22e-16` and `8.88e-16`.

| Candidate | Mean return | Mean timeout rate | Mean MAV survival | Mean blue loss | Mean launch | Mean hit | Mean kill |
|---|---:|---:|---:|---:|---:|---:|---:|
| lower_feasible_bound | -14.9344 | 0.0561 | 0.0627 | 2.3333 | 4.6667 | 2.3333 | 2.3333 |
| interval_midpoint | -16.9506 | 0.0160 | 0.0000 | 2.3333 | 3.0000 | 1.0000 | 1.0000 |
| upper_feasible_bound | -23.2701 | 0.0815 | 0.0074 | 0.6667 | 2.0000 | 0.3333 | 0.3333 |

The original upper/seed2 process was externally stopped by the tool timeout and is retained as a partial run. The completed `s2_retry` run is used in the table.

## Readiness decision

Final status: `TAM_HAPPO_PAPER_FORMULA_V5_INSUFFICIENT_EVIDENCE`.

Reasons:

1. Probe runs used summary rich logging, so per-step launch gate diagnostics and dead-before reward growth evidence are absent.
2. The matched v3 first-20,480-step baseline contains terminal outcome/launch data, but it predates v5 Angle/Distance and launch-gate fields. Those values are marked unavailable, not zero.
3. Therefore 2-of-3 seed behavior/geometry/component direction consistency cannot be established without inventing missing evidence.

No 200K run is justified by the current audit contract.

## Verification

- Syntax checks passed for all modified Python files.
- v1-v5 and launch diagnostic regression: `147 passed`.
- v5 focused tests: `33 passed`.
- 3V2 and 5V4 real environment reset/step finite checks are included in the v5 focused suite.

