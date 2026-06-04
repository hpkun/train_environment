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
- Seen 3v2 evaluation smoke.
- Unseen 5v4 evaluation smoke.
- NaN-free training and evaluation.

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
- Evaluation on 3v2 and 5v4 produces no NaN.
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
