# Main MAPPO Experiment Protocol

## Protocol

| Decision | Value |
|---|---|
| Train config | `hetero_mav_shared_geo_3v2.yaml` |
| Eval configs | `hetero_mav_shared_geo_3v2.yaml`, `hetero_mav_shared_geo_5v4.yaml` |
| Observation adapter | `v2` (`mav_shared_geo`) |
| Reward | `brma_legacy` |
| Blue opponent | `greedy_fsm` |
| Algorithm | Current shared-actor MAPPO baseline (unchanged) |

## Running

### Mainline (brma_legacy)

```bash
python scripts/run_main_mappo_experiment.py --total-env-steps 500000 ...
```

### role_v1 Reward Ablation

```bash
python scripts/run_main_mappo_role_v1_experiment.py
```

No CLI arguments needed — edit the script's top-level constants to adjust
steps, output dir, etc.

## Experiment Scale

| Stage | Total env steps | Purpose |
|---|---|---|
| Pilot | 50k–100k | Quick sanity check |
| Baseline candidate | 500k | Formal baseline for paper |

This is a MAPPO baseline — not a method module.  No attention, HAPPO,
GRU, or role-aware algorithm modifications.
