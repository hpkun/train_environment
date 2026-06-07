# Main MAPPO Checkpoint Selection

## Why Not Just `latest`

RL training is non-monotonic.  The final checkpoint (`latest`) often
reflects late-training collapse (e.g. red always loses, MAV never
survives) rather than the best policy.

## Preferred Workflow: Training-Time Eval

Use `--eval-during-training` with the training script.  This runs
lightweight periodic evaluations and saves the best checkpoint to
`output_dir/best/model.pt`.  This is the recommended path and avoids
lengthy post-hoc checkpoint sweeps.

## Fallback: Full Checkpoint Sweep

The `scripts/evaluate_main_mappo_checkpoints.py` script can evaluate all
saved checkpoints post-hoc.  This is **slow** and intended only for
post-hoc debugging when training-time eval was not enabled.

Full checkpoint sweeps are **not** the default experiment path.

## Scoring

`primary_score = red_win_rate + 0.1 * mav_survival_rate + 0.01 * avg_return`

This is a **diagnostic ranking metric**, not a paper performance claim.

## Constraints

- Does not modify environment, reward, termination, or training
- Does not implement new algorithms
- This is not a method module
