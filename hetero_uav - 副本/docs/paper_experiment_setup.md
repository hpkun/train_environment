# Paper Experiment Setup

## Environment

| Item | Setting |
|---|---|
| Simulator | JSBSim air-combat environment |
| Dynamics interface | High-level control with PID/JSBSim backend |
| Action space | 3D high-level action `[pitch, heading, speed]` |
| Missile launch | Environment-level scripted/fire-control logic |
| Blue opponent | Rule-based opponent in current protocol |
| Maximum episode length | `max_steps=1000` |

## Aircraft And Roles

| Role | Current experiment setup |
|---|---|
| MAV | F-16 MAV surrogate in main training/eval path; visual labels may use MAV/F22 notation in ACMI where documented |
| UAV | F-16 UAV |
| MAV role | survival/support/information role |
| UAV role | attack role |

The F-16 MAV surrogate is an engineering choice for controllability under the
current high-level action + PID interface. It should not be described as the
physical MAV model from the original heterogeneous paper.

## Scenarios

| Scenario | Use | Red team | Blue team |
|---|---|---|---|
| 3v2 | Training / seen evaluation | 1 MAV + 2 UAV | 2 UAV |
| 5v4 | Zero-shot transfer evaluation | 1 MAV + 4 UAV | 4 UAV |

The 5v4 evaluation is fixed-capacity zero-shot scale transfer. No 5v4
fine-tuning is performed.

## Observation And State

| Item | Dimension / behavior |
|---|---|
| Actor observation | 96 |
| Critic state | 480 |
| Observation style | V2 fixed-capacity entity/mask-inspired flat schema |
| Internal entity decode | Used by entity/BRMA policy paths |
| Padding/masks | Valid, alive, and observed masks support 3v2 and 5v4 compatibility |

## Policies

| Policy | Role in experiments |
|---|---|
| `flat` | baseline MLP policy |
| `brma_entity` | BRMA-style entity encoder ablation |
| `brma_recurrent` | entity encoder + GRU ablation |
| `brma_recurrent_masked` | final opt-in BRMA-style recurrent masked policy path |

## Evaluation Metrics

Use the following metrics for final tables and plots:

- `red_win_rate`;
- `blue_win_rate`;
- `draw_rate`;
- `timeout_rate`;
- `red_elimination_win_rate`;
- `red_timeout_alive_advantage_rate`;
- `red_missiles_fired_mean`;
- `red_missile_hits_mean`;
- `blue_dead_mean`;
- MAV survival rate;
- reward curve;
- win-rate curve;
- trajectory plot;
- attitude plot if available.

## Reporting Boundary

The current experiment setup supports a fixed-capacity 3v2-to-5v4 zero-shot
evaluation protocol. It does not support claims of arbitrary-scale
generalization or full reproduction of BRMA-MAPPO/TAM-HAPPO.

