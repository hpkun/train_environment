# Zero-Shot Experiment Protocol

## 1. Research Objective

Heterogeneous UAV/MAV composition zero-shot transfer:
train a shared policy on one MAV/UAV composition, evaluate on a
different composition without retraining.

## 2. Paper Roles

- **BRMA-MAPPO**: inspiration for variable-scale training, zero-shot
  generalisation via entity attention with biased random masks.
- **TAM-HAPPO**: inspiration for MAV/UAV heterogeneous environment
  definition, 3v2 and 5v4 scenario settings, role-based coordination.

## 3. What Is Not Claimed

- This is NOT a full reproduction of TAM-HAPPO.
- This is NOT a full reproduction of BRMA-MAPPO.
- Current MAPPO trainability smoke runs do NOT constitute evidence of
  successful zero-shot transfer.
- No win-rate conclusions can be drawn from smoke / diagnostic scripts.

## 4. Scenario Groups

### A. Debug / engineering-stable configs

- `hetero_train_2v2_mav_attack.yaml`
- `hetero_test_3v3_mav_2attack.yaml`
- `hetero_test_3v3_mav_attack_scout.yaml`
- `hetero_test_3v3_mav_attack_interceptor.yaml`

### B. Paper-aligned configs

- `hetero_paper_3v2_mav_2uav_vs_2uav.yaml` (TAM-HAPPO 3v2)
- `hetero_paper_5v4_mav_4uav_vs_4uav.yaml` (TAM-HAPPO 5v4)

### C. Composition zero-shot config pairs

- Train: 2v2, eval: 3v2 (seen roles, unseen scale)
- Train: 2v2, eval: 5v4 (seen roles, unseen scale)
- Train: 3v2, eval: 5v4 (both unseen)

## 5. Proposed Experiment Stages

| Stage | Name | Goal |
|---|---|---|
| E0 | Environment smoke | All configs load, adapter shapes correct, no NaN |
| E1 | MAPPO trainability | 20–200 iterations, loss curves, no NaN |
| E2 | MAPPO baseline formal | Multi-seed, multi-episode, full metrics |
| E3 | Zero-shot composition | Train on X, eval on Y, gap measurement |
| E4 | Role-aware / attention | Incremental method improvement |
| E5 | HAPPO-like extension | Only if role-aware is insufficient |

## 6. Metrics (for future formal experiments)

- Average return (mean over red agents)
- MAV survival rate
- Red alive count / Blue alive count
- Episode length
- Kills / losses (if available in info dict)
- Crash rate
- NaN rate
- Zero-shot generalisation gap (train_test_return_gap)

## 7. Required Cautions

- "Can run on unseen composition" is NOT evidence of zero-shot success.
- Formal zero-shot evaluation needs multiple seeds and episodes.
- Smoke scripts deliberately omit seed averaging and episode statistics.
- Current `OpponentPolicy = rule_nearest` is a placeholder; paper may
  specify a different blue baseline.

## 8. Observation Modes

`brma_sensor` is the compatibility baseline observation mode. It keeps the
BRMA-style raw observation path and is useful for debugging, regression tests,
and ablations.

`mav_shared_geo` is the main experimental candidate observation mode. It adds
geometric self/entity fields and expresses the MAV situation-support role
through MAV-mediated information sharing:

- UAV direct observation is preferred;
- MAV shared observation is used when direct observation is unavailable and the
  red MAV can track the enemy;
- unavailable enemies are zeroed and masked.

This v2 observation mode is still an environment abstraction. It does not claim
that "UAV can run on unseen composition" equals zero-shot success.

`alive_mask` and `observed_mask` are separate. A valid enemy slot may be alive
but currently unobserved (`valid=1, alive=1, observed=0`). In that state, the
actor receives zero enemy geometry while masks preserve the enemy's true alive
status. The default MAV missile count is 0; armed MAV cases must opt in through
config.

Formal future experiments should compare at least:

- `brma_sensor + MAPPO`;
- `mav_shared_geo + MAPPO`;
- `mav_shared_geo + attention/role-aware method`.
