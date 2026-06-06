# Paper-Aligned Protocol Smoke

## Purpose

This smoke test verifies that the frozen paper-aligned environment protocol
can be exercised end-to-end through the MAPPO training and evaluation pipeline.

This is **NOT**:
- A formal training run
- A long-run baseline
- A zero-shot success claim
- A method module exercise

## Protocol

| Decision | Value |
|---|---|
| Train config | `hetero_mav_shared_geo_3v2.yaml` |
| Eval configs | `hetero_mav_shared_geo_3v2.yaml`, `hetero_mav_shared_geo_5v4.yaml` |
| Reward mode | `brma_legacy` |
| Observation mode | `mav_shared_geo` (V2) |
| Actor dim | 96 |
| Critic dim | 480 |
| Decision dt | 0.2s (sim_freq=60, agent_interaction_steps=12) |

## Opponent Policies

| Policy | Status |
|---|---|
| `rule_nearest` | Available, default opponent |
| `greedy_fsm` | Diagnostic opponent, not yet final default |

## Pass Criteria

- Training completes without NaN
- Model checkpoint (`model.pt`) saved
- Metadata (`meta.json`) correct:
  - `obs_adapter_version == "v2"`
  - `actor_obs_dim == 96`
  - `critic_state_dim == 480`
- Train log CSV: `nan_detected == 0` for all rows
- Evaluation completes without NaN
- `actor_dim_ok == True` for all eval configs
- `critic_dim_ok == True` for all eval configs
- Combat metrics present:
  - `red_win_rate`
  - `blue_win_rate`
  - `draw_rate`
  - `timeout_rate`
  - `mav_survival_rate`

## Smoke Runner

```bash
python scripts/smoke_paper_aligned_protocol.py \
  --total-env-steps 512 \
  --rollout-length 64 \
  --max-steps 1000 \
  --eval-episodes 2 \
  --opponent-policies rule_nearest greedy_fsm \
  --device cpu \
  --output-dir outputs/paper_aligned_protocol_smoke
```

## What Comes Next

- User decides whether to run a longer baseline or continue opponent refinement
- `greedy_fsm` remains diagnostic; `rule_nearest` remains the default
- **No method module yet** — this is not a method module
- **This is not a zero-shot claim**
