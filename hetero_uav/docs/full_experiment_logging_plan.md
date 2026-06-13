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
