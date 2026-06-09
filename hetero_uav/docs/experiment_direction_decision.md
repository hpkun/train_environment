# Experiment Direction Decision

## 1. Project Goal

The project goal is a zero-shot transfer experiment for heterogeneous MAV/UAV
cooperative air combat. It is not a complete air-combat engineering system, and
it is not low-level flight-control reinforcement learning.

The experiment should answer a focused question: can a red MAV/UAV team trained
on a smaller composition transfer to a larger composition while exploiting
heterogeneous roles and observations?

## 2. Candidate Directions

### A. Move Fully Toward The Heterogeneous Paper Environment

This direction would switch toward the heterogeneous MAV/UAV paper literally:
4D low-level throttle/aileron/elevator/rudder actions, missile-aware
observation, role reward, TAM-HAPPO, temporal features, and attention.

Advantages:

- Closest surface match to the heterogeneous paper.
- Makes missile dodge reward and temporal/attention methods more natural.
- Could support stronger claims if the whole environment were faithfully
reproduced.

Risks:

- Very large environment change before the baseline is stable.
- Cross-aircraft control signs, trim, PID removal, and FDM differences become
new confounders.
- Hard to tell whether results come from heterogeneous method design or from
unsettled low-level control.
- Too much work for the current paper-experiment phase.

### B. Use BRMA-MAPPO As The Environment Base And Add Heterogeneous Method Design

This direction keeps the current BRMA-style environment base: high-level
`[pitch, heading, speed]` actions, PID control, BRMA missile launch logic,
`brma_legacy` reward, and the current F-22 MAV plus F-16 UAV composition. It
uses the heterogeneous paper mainly for role-aware observation, role-conditioned
or entity-attention MAPPO, and zero-shot composition transfer.

Advantages:

- Keeps the currently working JSBSim/PID/missile environment stable.
- High-level actions are already compatible with shared cross-aircraft MAPPO
  baselines.
- Directly supports the current 3v2 train and 5v4 zero-shot evaluation protocol.
- Lets the next experiment isolate the method contribution instead of changing
  reward, action, and dynamics at the same time.

Risks:

- It is not a literal reproduction of the heterogeneous paper environment.
- Missile-aware observation and dodge reward remain future extensions.
- The method claim must be framed as a BRMA-style environment with
  heterogeneous MAV/UAV observation and composition transfer, not a full
  TAM-HAPPO reproduction.

### C. Continue Tuning `role_v1` Reward

This direction keeps the current algorithm and tries to improve results mainly
through role-aware reward shaping.

Advantages:

- Small code surface if each change is minimal.
- Directly targets MAV survival and UAV attack behavior.
- Useful as an ablation after a stable baseline exists.

Risks:

- The completed `role_v1` 50k run is clearly weaker than `brma_legacy`.
- Current `role_v1` did not improve MAV survival.
- Reward tuning can hide method weaknesses and create fragile results.
- The current failure audit found implementation issues such as MAV support
  using `enemy_alive_mask` instead of observed/shared enemy evidence.

## 3. Evidence From Papers

| Topic | BRMA-MAPPO evidence | Heterogeneous MAV/UAV evidence | Current project choice | Temporarily not adopted |
|---|---|---|---|---|
| Action space | High-level target pitch, heading, and velocity with PID control | Low-level 4D throttle/aileron/elevator/rudder style control | Keep `[pitch, heading, speed] + PID` | 4D low-level action, because it would make control/FDM the main variable |
| Dynamics/control | JSBSim aircraft and PID bridge are compatible with BRMA-style action | Low-level action is closer to direct actuator control | Keep JSBSim/PID as environment base | Removing PID or retraining low-level control |
| Missile launch | BRMA-style launch gates, lock delay, cooldown, range/angle checks | Missile and dodge information are important for combat behavior | Keep BRMA missile launch logic | PN redesign or new missile dynamics |
| Missile evasion | Scripted missile warning/evasion exists in the BRMA-style environment | Heterogeneous paper includes missile-related observation/reward ideas | Red-only scripted missile evasion as current information advantage | Full learned dodge reward, because the actor lacks complete missile geometry |
| Reward | BRMA-style flight stability, geometry, missile/combat outcome, terminal reward | MAV emphasizes safety/support/events; UAV emphasizes height/speed/angle/distance/dodge/events | Use `brma_legacy` as baseline; keep `role_v1` as failed ablation evidence | Large reward retuning before method work |
| Observation | BRMA observation supports ego/ally/enemy/missile-warning style state | Heterogeneous method relies on role/entity information and temporal/attention processing | Use `mav_shared_geo` and V2 adapter with role/source/masks | Full missile entity observation in the mainline |
| Method | MAPPO/BRMA-MAPPO motivates a shared-policy baseline | TAM-HAPPO emphasizes temporal feature, attention, and heterogeneous roles | Current shared MLP MAPPO is baseline only | Full TAM-HAPPO/HAPPO until baselines and attention actor are validated |
| Zero-shot protocol | Not the main contribution by itself | Train small composition and transfer to larger composition | Train 3v2, evaluate 5v4 | Treating one latest checkpoint as final proof |

The key interpretation is that the heterogeneous paper's 4D low-level action is
not mandatory for the current stage. Its more important experimental
contribution for this project is role separation, heterogeneous observation,
attention/temporal method design, and zero-shot composition transfer.

## 4. Evidence From Results

The most reliable current baseline is the `brma_legacy` 50k run after the alive
mask and team-done fixes:

| run | 3v2 red_win | 3v2 blue_win | 3v2 MAV survival | 5v4 red_win | 5v4 blue_win | 5v4 MAV survival |
|---|---:|---:|---:|---:|---:|---:|
| `brma_legacy` 50k alive/done fix | 0.70 | 0.00 | 0.00 | 0.40 | 0.30 | 0.00 |
| `role_v1` 50k | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| `brma_legacy` 100k latest | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| `brma_legacy` 200k latest | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| `brma_legacy` 500k latest | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 100k `iter_0580` checkpoint | 0.10 | 0.50 | 0.00 | 0.05 | 0.75 | 0.00 |

Result interpretation:

- The `brma_legacy` 50k run is the strongest current MAPPO baseline signal.
- The `role_v1` implementation is weaker than `brma_legacy` and does not
  improve MAV survival.
- The 50k result is useful for direction screening, not a final convergence
  claim.
- The 100k, 200k, and 500k latest checkpoints show that simply adding steps and
  looking at latest can make the policy worse.
- Checkpoint selection helps diagnose non-monotonic training, but the `iter_0580`
  result is not a method breakthrough.
- The alive mask / team-done fix and red-only evasion change were useful because
  they corrected training semantics and experimental asymmetry without changing
  the core environment.

## 5. Recommended Direction

Use BRMA-MAPPO as the environment base and add heterogeneous method design.

Concretely, keep:

- high-level `[pitch, heading, speed]` actions;
- PID control;
- BRMA missile launch logic;
- `brma_legacy` as the baseline reward;
- F-22 MAV and F-16 UAV;
- `mav_shared_geo` observation;
- 3v2 training and 5v4 zero-shot evaluation.

Use the heterogeneous MAV/UAV paper as the method-design reference: role
conditioning, entity-aware observation processing, attention, temporal features,
and composition transfer. The immediate method step should be a minimal
role-conditioned or entity-attention MAPPO design, not a large environment
rewrite.

Do not continue large `role_v1` tuning now, and do not switch to 4D low-level
control now.

## 6. Next Minimal Steps

1. Freeze the `brma_legacy` 50k alive/done fix run as the current MAPPO baseline
   reference result.
2. Design the minimal role-conditioned actor plan before writing code.
3. Design missile-aware observation as a later extension, outside the current
   mainline.

## 7. Stop Doing List

- Do not keep blindly tuning `role_v1` numeric values.
- Do not keep running 500k and judging only the latest checkpoint.
- Do not switch now to 4D low-level throttle/aileron/elevator/rudder actions.
- Do not implement full TAM-HAPPO now.
- Do not keep adding complex engineering audit scripts before the method
  direction is fixed.
