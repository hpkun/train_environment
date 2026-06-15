# Full Experiment Logging Plan

This plan defines the data that must be recorded in a complete run so BRMA-MAPPO-style and TAM-HAPPO-style plots can be generated after training. It does not change reward, missile dynamics, PID, aircraft XML, action space, observation dimension, blue rule, or MAV policy.

## 1. Data Required By The Two Reference Paper Styles

BRMA-MAPPO-style reporting needs reward curves, win-rate curves, scale transfer bars, relative win ratio, kill/death ratio, training efficiency, ablation tables, and attention or mask-related diagnostics when available.

TAM-HAPPO-style reporting needs reward curves, 2v2/3v2/5v4 trajectories, aircraft attitude curves, altitude/speed/yaw/pitch curves, heterogeneous reward components, loss and policy-gradient curves, and perturbation generalization evaluation.

## 2. Data Currently Recorded

When `--enable-rich-logging` is passed to `scripts/train_happo_reference.py`, the run writes:

- `train_metrics.csv`
- `training_efficiency.json`
- schema-stable CSV files for eval, aircraft timeseries, missile events/timeseries, reward components, perturbation summary, and attention metrics

The smoke runner also writes minimal rows into eval/timeseries/reward files so the plotting pipeline can be validated without long training.

## 3. Reserved But Not Implemented By Current Algorithm

- `attention_metrics.csv` is written with `availability=not_available` because the current algorithm has no attention module.
- perturbation evaluation is schema-supported, but real perturbation sweeps require a separate evaluation run.
- policy/value gradient norms are schema-supported; current HAPPO trainer does not expose detailed gradient norms yet.

## 4. Complete Experiment Command Pattern

Use the normal training command and add:

```powershell
python scripts/train_happo_reference.py `
  --enable-rich-logging `
  --rich-log-dir outputs/<experiment_name> `
  --timeseries-episodes-limit 3 `
  --timeseries-step-stride 5
```

For long runs after the 2048-step hang investigation, start conservatively:

```powershell
python scripts/train_happo_reference.py `
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml `
  --output-dir outputs/full_10m_normal_geometry_oracle_anchor `
  --total-env-steps 10000000 `
  --rollout-length 256 `
  --num-envs 1 `
  --device cuda `
  --init-checkpoint outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt `
  --uav-imitation-dataset outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz `
  --uav-imitation-coef 0.03 `
  --uav-imitation-until-steps 2000000 `
  --eval-during-training `
  --eval-interval-steps 500000 `
  --train-eval-episodes 5 `
  --enable-rich-logging `
  --rich-log-dir outputs/full_10m_normal_geometry_oracle_anchor/rich_logs `
  --timeseries-episodes-limit 3 `
  --timeseries-step-stride 10 `
  --heartbeat-log outputs/full_10m_normal_geometry_oracle_anchor/heartbeat.log `
  --heartbeat-every-steps 50
```

`--num-envs` defaults to `1` for stability. If a single-env run is stable, try
`--num-envs 2` before returning to `--num-envs 4`. Do not start a 10M run with
4 envs until the shorter stability check passes. `--eval-at-start` is disabled
by default, so `--eval-during-training` now follows `--eval-interval-steps`
instead of forcing an evaluation after the first rollout.

The env1 10M run can continue in the background. Do not overwrite or delete its
output directory while debugging parallel sampling.

For 4-env runs, note that this is single-process synchronous sampling. One
blocked JSBSim environment blocks the entire rollout. Long runs must explicitly
pass `--max-steps 1000`; the training script default `--max-steps 64` is useful
for fast smoke tests but creates high-frequency resets and is not suitable for
long air-combat training. Until 4-env long-run stability is proven, use 4-env
only for debug/stability validation and keep env1 as the formal stable path.

For 4-env debugging, add:

```powershell
--debug-rollout-heartbeat `
--heartbeat-stall-timeout-sec 300 `
--exit-on-heartbeat-stall
```

This writes every transition event to `heartbeat.log` and writes
`heartbeat_stall_report.json`, `heartbeat_stall_report.md`, and
`heartbeat_stall_stack.txt` if heartbeat output stalls.

Then generate plots:

```powershell
python scripts/generate_paper_style_plots.py --input-dir outputs/<experiment_name> --output-dir outputs/<experiment_name>/paper_style_figures
python scripts/check_paper_plot_coverage.py --input-dir outputs/<experiment_name> --output-dir outputs/<experiment_name>
```

## 5. Smoke Validation

The smoke path is:

```powershell
python scripts/run_rich_logging_smoke.py
python scripts/generate_paper_style_plots.py --input-dir outputs/rich_logging_smoke --output-dir outputs/rich_logging_smoke/paper_style_figures
python scripts/check_paper_plot_coverage.py --input-dir outputs/rich_logging_smoke --output-dir outputs/rich_logging_smoke
```

This validates schema and plotting with a short 1024-step run. It is not a training result.

## 6. Rich Logging Audit Gate

Before starting a complete experiment, run:

```powershell
python scripts/audit_rich_logging_outputs.py --input-dir outputs/rich_logging_smoke --figures-dir outputs/rich_logging_smoke/paper_style_figures
```

The latest smoke audit status is `pass_with_warnings`. This is acceptable for starting a full experiment because the warnings are explicit limitations rather than schema failures:

- `policy_gradient_norm` and `value_gradient_norm` columns exist, but the current trainer does not expose values yet.
- smoke has no missile events or missile timeseries rows; headers are present and full experiments can populate them.
- explicit reward component columns are stable, but smoke component values are zero/empty placeholders.
- `peak_gpu_memory_gb` and `peak_cpu_memory_gb` are currently not available.
- perturbation rows are marked `schema_only`, not real perturbation results.

## 7. Smoke-Only Versus Full-Experiment Figures

Smoke can verify that these figures render from the correct files, but they should not be used as formal conclusions:

- `zero_shot_transfer_bar`: requires full 3v2/5v4 evaluation.
- `ablation_reward_win_curve`: requires multiple runs.
- `perturbation_generalization_bar`: requires a real perturbation evaluation.

The following are structurally available in smoke and become meaningful after a complete run:

- reward curves
- win-rate curves
- RWR/KD
- trajectory and attitude curves
- reward component curves
- loss/entropy curves
- training efficiency table

Attention-related plots remain `not_implemented_by_current_algorithm` until an attention module is actually implemented. Do not fabricate attention entropy or attention weights.
