# Baseline 1M Results

## Scope

This note records the completed shared-MLP MAPPO 1M baseline result. It should
not be interpreted as a successful combat policy.

## Run

- Output directory: `outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim`
- Algorithm: shared MLP MAPPO baseline
- Opponent: `brma_rule`
- Train config: 3v2 heterogeneous MAV/UAV, no MAV trim
- Eval: 3v2 and 5v4
- Best checkpoint was re-evaluated with 100 episodes.

## 100-Episode Best-Checkpoint Result

| config | red_win | blue_win | draw | timeout | red_alive | blue_alive | MAV survival | blue elimination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v2 | 0.05 | 0.00 | 0.95 | 1.00 | 2.00 | 1.95 | 0.00 | 0.00 |
| 5v4 | 0.15 | 0.12 | 0.73 | 1.00 | 3.82 | 3.79 | 0.00 | 0.00 |

The policy survives until timeout in most episodes, but it does not establish
reliable killing ability. The MAV still dies in all evaluated episodes.

## Interpretation

The 1M shared MLP baseline learned a pure survival/timeout style behavior, not
an effective air-combat strategy. The red-win rates come from timeout alive
advantage, not from consistent blue elimination.

Previous small-sample red-win signals should therefore be treated as unstable
or timeout-driven, not as proof that the environment and method are complete.

## Decision

The current shared MLP baseline cannot prove that the heterogeneous MAV/UAV
environment is complete. The next step should be HAPPO 3v2 reference validation:
check whether a paper-informed heterogeneous policy setup can produce MAV
support behavior, UAV attacks, missile hits, and non-timeout outcomes.
