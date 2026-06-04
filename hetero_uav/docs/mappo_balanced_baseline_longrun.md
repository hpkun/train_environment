# MAPPO Balanced Baseline Long-Run

## Purpose

This workflow runs the plain MAPPO MLP baseline for a longer budget on the
balanced V2 heterogeneous environment. The purpose is to validate long-run
environment and baseline stability before adding any new method module.

It is not a new algorithm stage, and it is not a formal zero-shot claim.

## Why 500k Env Steps

Short smoke tests only verify that the training and evaluation pipeline can run.
They do not tell us whether the environment, observation adapter, baseline
training loop, checkpointing, and evaluation remain stable over longer runs.

500k env steps is the first practical long-run gate for checking that the plain
MAPPO baseline can train without NaNs, save/load models, and evaluate on the
balanced configs.

## Step Definition

`total_env_steps` counts calls to `env.step`.

It is not agent-steps. A 3-red-agent rollout of 128 environment steps still
counts as 128 env steps.

## Default Protocol

- Train config: balanced 3v3,
  `hetero_balanced_mav_shared_geo_3v3.yaml`
- Eval configs:
  - balanced 3v3, `hetero_balanced_mav_shared_geo_3v3.yaml`
  - balanced 4v4, `hetero_balanced_mav_shared_geo_4v4.yaml`
- `obs_adapter_version=v2`
- `actor_dim=96`
- `critic_dim=480`
- `actor_arch=mlp`
- `opponent_policy=rule_nearest`

## How To Run 500k

```bash
python scripts/run_mappo_balanced_baseline_longrun.py \
  --seeds 0 \
  --total-env-steps 500000 \
  --rollout-length 128 \
  --max-steps 500 \
  --eval-episodes 20 \
  --device cpu \
  --opponent-policy rule_nearest
```

500k with the default rollout length uses
`ceil(500000 / 128) = 3907` training iterations.

The long-run runner streams train/eval subprocess stdout to the console in real
time. It also saves per-seed logs:

- `seed_0/train_stdout.log`
- `seed_0/train_stderr.log`
- `seed_0/eval_stdout.log`
- `seed_0/eval_stderr.log`

To watch the training CSV from another PowerShell terminal:

```powershell
Get-Content outputs\mappo_balanced_baseline_500k\seed_0\train_log.csv -Wait
```

## Recommended Later Protocol

1. Run seed 0 at 500k env steps.
2. If seed 0 passes, run seeds 0 1 2.
3. Increase evaluation episodes after the basic long-run gate is stable.
4. Still do not call the result a formal experiment without a fixed statistical
   protocol and baselines.

## Outputs

- `longrun_train_summary.csv`
- `longrun_eval_summary.csv`
- `longrun_report.json`
- per-seed `train_log.csv`
- per-seed `train_stdout.log`
- per-seed `train_stderr.log`
- per-seed `eval_stdout.log`
- per-seed `eval_stderr.log`
- per-seed `latest/model.pt`
- per-seed checkpoint files

The report is a stability artifact. It should not be interpreted as proof of
zero-shot success.

## Combat Metrics After 500k

Do not judge the 500k run from return alone. After training, inspect combat
diagnostics:

- `red_win_rate`
- `blue_win_rate`
- `draw_rate`
- `timeout_rate`
- MAV survival rate
- final red/blue alive counts
- final red/blue dead counts
- episode end reasons

The balanced scenarios have equal total aircraft counts, but red has one fewer
shooting attack UAV because red includes a non-shooting MAV. Poor 4v4
generalization is therefore not automatically a code bug; it may indicate that
the balanced task is still difficult for plain MAPPO.

Run posthoc combat-metrics evaluation on a saved 500k model:

```bash
python scripts/evaluate_saved_mappo_with_combat_metrics.py \
  --model outputs/mappo_balanced_baseline_500k/seed_0/latest/model.pt \
  --episodes 100 \
  --device cpu \
  --opponent-policy rule_nearest
```
