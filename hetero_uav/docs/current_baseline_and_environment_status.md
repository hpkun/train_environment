# Current Baseline and Environment Status

## 1. Pipeline Baseline

**Status: WORKING.**

- Train → eval → save/load pipeline runs end-to-end
- `train_mappo_baseline.py` + `eval_mappo_zero_shot.py` functional
- `run_main_mappo_experiment.py` with `ExperimentSpec` framework
- ACMI export (one-episode, debug) functional; death display fixed
- Train/eval logging with granular metrics (elimination win rate, etc.)
- Periodic eval during training with best-checkpoint tracking
- Per-role action logging (mav_action_mean_abs, uav_action_saturation_rate)
- Multiple opponent modes: rule_nearest, greedy_fsm, brma_rule
- Multiple reward modes: brma_legacy, minimal_v1, role_v1

## 2. Learning Baseline

**Status: NOT ESTABLISHED.**

| Run | Opponent | Config | Actor | Steps | 3v2 red_win | Conclusion |
|---|---|---|---|---|---|---|
| alive_done_fix 50k | rule_nearest | with trim | mlp | 50k | 0.70 | FALSE POSITIVE — all wins by timeout+alive advantage; rule_nearest does not attack |
| paper-aligned 50k | brma_rule | no trim | mlp | 50k | 0.00 | FAILED — red eliminated every episode from step 3 |
| protocol-aligned 200k | brma_rule | no trim | mlp | 200k | 0.00 (mid-training) | FAILING — 200+ episodes, red_alive=0.0, entropy rising to 0.51 |

**No run using brma_rule has produced a non-zero red_win_rate.**

The shared MLP MAPPO (96→256→128→3 actor, 480→256→128→1 critic) trained
with brma_legacy reward and brma_rule opponent cannot learn any effective
strategy at 50k or 200k steps in the 3v2 heterogeneous setting.

## 3. Environment Validation

### Confirmed
- Blue brma_rule opponent integrated and fires missiles in diagnostic rollouts
- no_mav_trim config created (MAV pitch_trim=0.0)
- ACMI death display fixed (dead aircraft stop logging T=)
- Observation V2: actor_obs_dim=96, critic_state_dim=480, role at indices 7:11

### Not Yet Confirmed
- F-22 zero-action stability (max_roll=180 deg in 200-step fixed-action test)
- Whether F-22's extreme roll is environment artifact or expected aerodynamics
- Whether the MAV/UAV heterogeneous composition is fundamentally learnable

### Conclusion
Environment is NOT fully validated for training readiness.
