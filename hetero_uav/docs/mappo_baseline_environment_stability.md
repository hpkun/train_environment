# MAPPO Baseline Environment Stability

## Purpose

The current stage is to make the plain MAPPO baseline run reliably in the
heterogeneous MAV/UAV environment. The goal is to use ordinary MAPPO training
and evaluation as an environment stability validation path before adding any
new method module.

This is not an attention, HAPPO, GRU, or role-aware algorithm stage.

## What Is Validated

- Environment reset/step stability through MAPPO training and evaluation.
- V2 `mav_shared_geo` observation mode.
- `HeteroObsAdapterV2`.
- `actor_obs_dim=96`.
- `critic_state_dim=480`.
- Plain MLP actor/critic model save and load.
- Balanced 3v3 training/evaluation smoke.
- Balanced 4v4 composition smoke.
- NaN-free training and evaluation.

## Config Scope

Paper-aligned configs are retained as TAM-HAPPO scenario references:

- `hetero_paper_3v2_mav_2uav_vs_2uav.yaml`
- `hetero_paper_5v4_mav_4uav_vs_4uav.yaml`

They are no longer the default stability validation path.

Balanced configs are the current mainline:

- Train/default eval: `hetero_balanced_mav_shared_geo_3v3.yaml`
- Composition smoke eval: `hetero_balanced_mav_shared_geo_4v4.yaml`

The balanced setup avoids mixing environment stability signals with red/blue
count asymmetry. It is a cleaner gate before comparing plain MAPPO against any
future method module.

## What Is Not Claimed

- This is not a formal zero-shot experiment.
- It does not claim that zero-shot transfer has succeeded.
- It does not claim V2 is better than V1.
- It does not report formal win rate.
- It does not show that the policy has converged.

## Pass Criteria

- Every seed trains without subprocess failure.
- `latest/model.pt`, `latest/meta.json`, and `train_log.csv` are present.
- Train logs contain no NaN flags.
- Model metadata reports `obs_adapter_version=v2`, `actor_obs_dim=96`, and
  `critic_state_dim=480`.
- Evaluation on balanced 3v3 and balanced 4v4 produces no NaN.
- Evaluation reports `actor_dim_ok=True` and `critic_dim_ok=True`.
- Summary files are complete:
  - `stability_train_summary.csv`
  - `stability_eval_summary.csv`
  - `stability_report.json`

## How To Run

```bash
python scripts/validate_mappo_baseline_environment_stability.py \
  --seeds 0 1 \
  --iterations 50 \
  --rollout-length 32 \
  --max-steps 128 \
  --eval-episodes 3 \
  --device cpu \
  --opponent-policy rule_nearest
```

## Next Decision

Only after this validation is stable should the project decide whether to enter
the next method stage. A likely next stage may be an entity attention module,
but this document does not require or implement it.
