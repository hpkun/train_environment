# Current Baseline and Environment Status

## 1. Pipeline Baseline

**Status: WORKING.**

- Train, eval, save/load pipeline runs end-to-end.
- `train_mappo_baseline.py` and `eval_mappo_zero_shot.py` are functional.
- `ExperimentSpec` framework supports multiple runners.
- ACMI export and granular eval metrics are available.
- Periodic eval during training and best-checkpoint tracking are available.
- Opponent modes include `rule_nearest`, `greedy_fsm`, and `brma_rule`.

## 2. Learning Baseline

**Status: RUNNABLE but NOT an effective combat policy.**

| Run | Opponent | Config | Steps | Main outcome |
|---|---|---|---:|---|
| alive_done_fix 50k | rule_nearest | 3v2/5v4 | 50k | Useful debugging signal, but not final protocol |
| protocol-aligned 200k | brma_rule | 3v2 no_trim | 200k | red_alive stayed near 0; blue_win stayed near 1 |
| protocol-aligned 1M best checkpoint | brma_rule | 3v2 no_trim -> 3v2/5v4 eval | 1M | timeout survival behavior, not combat effectiveness |

The 1M best-checkpoint 100-episode re-evaluation shows:

| eval config | red_win | blue_win | draw | timeout | red_alive | blue_alive | MAV survival | elimination wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v2 | 0.05 | 0.00 | 0.95 | 1.00 | 2.00 | 1.95 | 0.00 | 0.00 |
| 5v4 | 0.15 | 0.12 | 0.73 | 1.00 | 3.82 | 3.79 | 0.00 | 0.00 |

The best checkpoint can survive to timeout, but it does not form reliable kill
ability. MAV survival remains zero. This means the current shared MLP baseline
learned a timeout-survival policy, not a valid air-combat strategy.

## 3. Environment Status

- The environment can run JSBSim heterogeneous 3v2/5v4 with F-22 MAV and F-16
  UAVs.
- `brma_rule` opponent is functional enough to evaluate protocol behavior.
- Current shared MLP results do not prove that the environment is complete.
- The missing signal is not only red survival; it is MAV support, UAV attack,
  missile hits, and non-timeout combat outcomes.

## 4. Next Step

Move to HAPPO 3v2 reference validation. The goal is not full TAM-HAPPO yet, but
to check whether a paper-informed heterogeneous policy setup can support
reasonable 3v2 combat behavior: MAV support, UAV engagement, blue deaths, and
non-timeout outcomes.
