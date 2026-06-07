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

## Opponent Policy

Default blue opponent is `greedy_fsm`. For baseline trainability
diagnostics, `--opponent-policy rule_nearest` runs a weaker rule-based
opponent pilot. This is not a new method or an environment change.

## Training-Time Diagnostics

New since 2026-06: training now records extended metrics in `train_log.csv`
(action saturation, episode outcome rates, missile stats) and can run
lightweight periodic eval (`--eval-during-training`) with best-checkpoint
selection.  Full checkpoint sweeps are discouraged for regular workflow;
use `eval_log.csv` and `best/model.pt` instead.  Checkpoint sweeps remain
available for post-hoc debug only.

## Experiment Scale

| Stage | Total env steps | Purpose |
|---|---|---|
| Pilot | 100k | Quick sanity check |
| Baseline candidate | 500k | Formal baseline for paper |

This is a MAPPO baseline — not a method module.  No attention, HAPPO,
GRU, or role-aware algorithm modifications.
