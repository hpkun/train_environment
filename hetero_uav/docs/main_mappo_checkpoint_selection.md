# Main MAPPO Checkpoint Selection

## Why Not Just `latest`

RL training is non-monotonic.  The final checkpoint (`latest`) often
reflects late-training collapse (e.g. red always loses, MAV never
survives) rather than the best policy.  Evaluating intermediate
checkpoints is standard RL experiment practice.

## What This Does

- Defaults to sampled checkpoint evaluation instead of full 79-checkpoint sweeps
- Evaluates selected checkpoints and `latest/model.pt`
- Ranks evaluated checkpoints with a diagnostic score
- Reports whether any checkpoint shows red_win_rate > 0 or
  mav_survival_rate > 0

## Selection Modes

- `sampled`: default. Select checkpoints by stride, include `latest`, and cap
  the count with `--max-checkpoints`.
- `top-train`: read `train_log.csv`, select checkpoints with high
  `average_team_return`, include `latest`, and cap the count.
- `all`: explicitly evaluate every checkpoint. This may be very slow and is
  not the default experiment path.

## Scoring

`primary_score = red_win_rate + 0.1 * mav_survival_rate + 0.01 * avg_return`

This is a **diagnostic ranking metric**, not a paper performance claim.
It prioritizes red wins and MAV survival to surface checkpoints that
avoid total blue domination.

## Constraints

- Does not modify environment, reward, termination, or training
- Does not implement new algorithms
- This is not a method module
