# MARL/PPO Training Dynamics Diagnostics

This note documents the detached diagnostics added for `pure_happo` training.
The diagnostics do not change rollout storage, losses, gradients, optimizer
steps, rewards, environment dynamics, missile logic, action semantics, or
observation dimensions.

## Training Log Fields

New scalar fields are appended to `train_log.csv` before `nan_detected`. Existing
columns keep their original order.

PPO update health:

- `clip_fraction_mav`, `clip_fraction_uav`
- `approx_kl_abs_mav`, `approx_kl_abs_uav`
- `ratio_mean_*`, `ratio_std_*`, `ratio_p95_*`, `ratio_p99_*`
- `actor_grad_norm_*`, `critic_grad_norm`
- `policy_update_norm_*`, `critic_update_norm`

Critic and target quality:

- `critic_loss_unscaled`, `critic_loss_scaled`
- `value_explained_variance`
- `value_pred_mean/std`
- `return_mean/std`
- `advantage_raw_*`, `advantage_norm_*`

Action distribution:

- per-role action mean/std/mean-abs/saturation for pitch, heading, speed
- existing `action_log_std_*` fields remain available

Sample quality:

- `mav_active_sample_count`, `uav_active_sample_count`
- `entropy_mav_valid_count`, `entropy_uav_valid_count`
- `actor_obs_mean/std/abs_max/nan_count`
- `critic_state_mean/std/abs_max/nan_count`

## Update Diagnostics JSONL

`update_diagnostics.jsonl` writes one JSON record per completed PPO update. It
contains arrays that are too wide for stable CSV columns:

- per-agent actor loss, entropy, KL, clip fraction, ratio statistics
- per-agent grad norms and parameter update norms
- HAPPO correction factor summaries after each sequential agent update
- active sample ratios and counts per agent

These records are for offline audit and should not be consumed by the training
loop.

## Reading Rules

Use these diagnostics to locate training dynamics failures before focusing on
missile hit counts:

- High `approx_kl`, high `clip_fraction`, or large grad/update norms usually
  indicates an overly aggressive PPO update.
- Low KL and low clip fraction with poor outcomes suggests the policy is
  changing too little, or the reward/initial state produces a local optimum.
- Poor or negative `value_explained_variance` means critic targets are not being
  fitted reliably. Low critic loss with bad outcomes can still mean the critic
  is fitting a weak or misleading return signal.
- Falling entropy/log standard deviation indicates possible policy collapse.
  High entropy/log standard deviation without outcome improvement indicates
  ineffective exploration rather than insufficient randomness.
- High action saturation, especially in heading or pitch, often indicates a
  boundary-action strategy or action-contract mismatch.
- Low active sample counts mean deaths, truncations, or masks are dominating
  the PPO batch.
- High timeout and high MAV survival with low red win can indicate a survival or
  timeout local optimum.
- Reward component growth without matching outcome improvement is a reward
  semantics warning, not evidence of learned combat behavior.

## Offline Analyzer

Run:

```bash
python scripts/analyze_marl_training_dynamics.py \
  --output-dir outputs/<run_dir> \
  --phase-bins 0,50000,100000,200000
```

Outputs:

- `learning_dynamics_summary.csv`
- `learning_dynamics_summary.json`
- `learning_dynamics_report.md`

The analyzer supports old runs with missing fields by treating unavailable
values as `0.0`. Its flags are heuristic screening signals, not causal proof.
