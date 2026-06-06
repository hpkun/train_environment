# Blue Opponent Protocol Validation

## Purpose

Compare `rule_nearest` and `greedy_fsm` as scripted blue opponents under fixed
red policies.  This validation helps decide which opponent should serve as the
default baseline and which should serve as the hard diagnostic opponent.

This is not a training run. This is **NOT**:
- A win-rate experiment
- An algorithm comparison
- A method module

## Relation to Papers

- **BRMA-MAPPO** requires a fixed-rule blue opponent.
- **TAM-HAPPO** motivates a greedy finite-state opponent.
- The current `greedy_fsm` is an engineering approximation, not a full
  paper reproduction.

## Metrics Collected

| Metric | Description |
|---|---|
| nan_detected | Any NaN in observation or reward |
| action_min / action_max | Blue action range (must be in [-1, 1]) |
| action_mean_abs | Mean absolute action magnitude |
| red_alive_final_mean | Average red aircraft alive at episode end |
| blue_alive_final_mean | Average blue aircraft alive at episode end |
| mav_survival_rate | Fraction of episodes where MAV survived |
| red_win_rate / blue_win_rate / draw_rate | Win/loss/draw fractions |
| timeout_rate | Episodes ending by timeout |
| avg_episode_length | Mean steps per episode |
| greedy_fsm state_counts | FSM state distribution (greedy_fsm only) |
| opponent_difficulty_label | Diagnostic label (see below) |

## Opponent Difficulty Labels

| Label | Condition |
|---|---|
| `too_strong_candidate` | blue_win_rate >= 0.9 AND mav_survival_rate <= 0.2 |
| `too_weak_candidate` | red_win_rate >= 0.9 |
| `stable_candidate` | No NaN, not clearly too strong/weak |

These are diagnostic labels only — not formal conclusions.

## Decision Use

- `rule_nearest`: candidate for easy / default baseline opponent
- `greedy_fsm`: candidate for hard / diagnostic opponent
- Final default opponent decision requires **user confirmation**

## Not a Method Module

This validation does not involve attention, HAPPO, GRU, or any
algorithm-level changes. This is not a method module — it is purely an
environment-protocol exercise.
