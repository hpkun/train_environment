# Heterogeneous Environment Finalization Plan

## Project Environment Objective

The project objective is heterogeneous UAV/MAV team composition zero-shot
transfer. The current environment is not a full BRMA-MAPPO reproduction and is
not a full TAM-HAPPO reproduction.

The intended environment design combines:

- a BRMA-style JSBSim, high-level action, missile, reward, and termination
  foundation;
- TAM-HAPPO-style MAV/UAV heterogeneous composition and MAV situation-support
  ideas.

This document is a readiness and finalization plan. The environment is not ready
for method module work until the gaps below are audited and resolved. In short:
not ready for method module changes.

## Protocol Taxonomy

### A. Paper-Aligned Heterogeneous Protocol

This is the main protocol.

Train 3v2:

- red = 1 MAV + 2 attack_uav
- blue = 2 attack_uav

Eval 5v4:

- red = 1 MAV + 4 attack_uav
- blue = 4 attack_uav

The extra red aircraft is the MAV. The number of red attacking UAVs equals the
number of blue attacking UAVs, and the MAV is a support platform.

### B. Balanced Total-Count Protocol

This is a hard ablation, not the main protocol.

Train 3v3:

- red = 1 MAV + 2 attack_uav
- blue = 3 attack_uav

Eval 4v4:

- red = 1 MAV + 3 attack_uav
- blue = 4 attack_uav

The total aircraft counts match, but red has one fewer attacking UAV than blue
because red includes a non-shooting MAV. This makes the task harder and should
not be used as the default environment-readiness conclusion.

### C. BRMA Homogeneous Scale Protocol

This is a future reference protocol, not the current priority.

- train 6v6
- eval 8v8 / 10v10

It can be used later as a homogeneous scale-transfer reference.

## Environment Components Status Table

| component | current implementation | desired status | action needed |
|---|---|---|---|
| aircraft model | MAV uses A-4, attack UAV uses f16 | keep explicit per type | audit model stability and document limitations |
| MAV missile count | MAV defaults to `num_missiles=0` | support-only MAV by default | keep, document armed-MAV as opt-in only |
| UAV missile count | attack UAV defaults to `num_missiles=2` | consistent in paper-aligned and balanced configs | audit every protocol config |
| high-level action | `[pitch, heading, speed]` | BRMA-style high-level command | no method change; continue diagnostics |
| JSBSim/PID | migrated BRMA-style JSBSim/PID base | stable enough for protocol runs | continue stability checks, no XML/PID change here |
| decision frequency | `sim_freq` and `agent_interaction_steps` configurable | consistent per protocol | audit and standardize |
| episode max_steps | config/script values still vary | explicit protocol default | align after audit |
| observation V1 brma_sensor | compatibility adapter, actor 140 / critic 700 | ablation/reference | keep available |
| observation V2 mav_shared_geo | actor 96 / critic 480 | main observation candidate | audit required fields and masks |
| MAV shared information | abstract direct/shared observation logic | support MAV situation sharing | audit communication assumptions |
| blue opponent policy | zero/random/rule_nearest plus diagnostic greedy_fsm | greedy finite-state / situation-based baseline | greedy_fsm design in progress; rule_nearest remains default |
| missile/fire-control | inherited BRMA-style mechanics | explicit protocol documentation | audit against heterogeneous objective |
| evasion | inherited BRMA logic | explicit heterogeneous audit | audit, do not change blindly |
| reward | inherited BRMA reward with hetero metadata | explicit MAV/UAV role objective | reward/termination audit needed |
| termination | inherited elimination/timeout mechanics | clear end reason metrics | audit max_steps and outcome semantics |
| combat metrics | posthoc win/end-reason metrics exist | train/eval diagnostics with outcome statistics | add per-episode outcome logging later |
| initial states | configured per scenario | paper-aligned mainline and ablation separated | audit every protocol config |
| scenario protocols | paper-aligned and balanced configs both exist | paper-aligned mainline, balanced hard ablation | document and enforce defaults |

## Immediate Environment Gaps

- Blue opponent currently uses `rule_nearest`, not a greedy finite-state or
  situation-based policy.
- Episode `max_steps` is still inconsistent across scripts/configs.
- Paper-aligned protocol is not yet designated everywhere as the default
  environment protocol.
- V2 MAV shared observation is abstract and has no communication delay/noise.
- Reward/termination are inherited from BRMA and need explicit audit against
  the heterogeneous objective.
- Win/end-reason metrics exist, but training logs still do not include
  per-episode outcome statistics.
- Current balanced 4v4 failure should not be used to judge the algorithm until
  the protocol is finalized.

## Next Environment Tasks In Priority Order

E1. Environment protocol audit and config readiness.

E2. Paper-aligned 3v2/5v4 smoke and diagnostics.

E3. Episode length / decision frequency consistency.

E4. Blue opponent logic alignment.

Status: in progress.

- E4a paper-alignment audit.
- E4b controlled branch diagnostics.
- E4c visibility/geometry decision.
- E4d search/acquisition behavior.
- E4e target assignment and candidate maneuver scoring.
- E4f only then consider training with `greedy_fsm`.

The initial `greedy_fsm` diagnostic is implemented, but `rule_nearest` remains
the default. No training run should switch to `greedy_fsm` without explicit user
confirmation.

E5. Reward/termination audit for MAV/UAV roles.

E6. Long-run baseline after environment protocol is frozen.

E7. Only after E1-E6, ask the user whether to enter a method module.
