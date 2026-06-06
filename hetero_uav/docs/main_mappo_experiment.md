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

## Composition

- Train: red = 1 MAV + 2 attack_uav, blue = 2 attack_uav
- Eval 3v2: same as train
- Eval 5v4: red = 1 MAV + 4 attack_uav, blue = 4 attack_uav

## Experiment Scale

| Stage | Total env steps | Purpose |
|---|---|---|
| Pilot | 100k | Quick sanity check |
| Baseline candidate | 500k | Formal baseline for paper |

This is a MAPPO baseline — not a method module.  No attention, HAPPO,
GRU, or role-aware algorithm modifications.
