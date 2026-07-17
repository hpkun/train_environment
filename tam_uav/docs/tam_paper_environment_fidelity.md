# TAM paper environment fidelity

Current revision: `published_rules_simplified_v3`. Outputs from v1/v2 and older
low-perturbation 2v2 experiments are **pre-v3 fidelity diagnostics**. Formal v3
checkpoint loading rejects missing or mismatched environment lineage.

## A. PUBLISHED_EXACT

- Tables 5–7 nominal initial states.
- 60 Hz physics, 5 Hz decisions, 12 physics frames per action, and 1000 steps.
- Four direct-control dimensions, 40 levels each, throttle `[0.4, 0.9]`, and
  aileron/elevator/rudder `[-1, 1]`.
- The published UAV speed, angle, distance, dodge, and event reward formulas and
  weights, except for the separately identified paper-silent height function.
- Point-mass PN equations, gains, 30 g limit, 14 km attack range, 25 s launch
  interval, and missile mass/length/diameter metadata.

## B. UNIQUELY_DERIVED

- Missile timeout is `2 * 14000 / 500 = 56 s`.
- Incoming-missile observation capacity is scenario-derived: four opposing
  attack UAVs times two missiles equals eight fixed slots.

## C. PAPER_SILENT_SIMPLIFICATION

- UAV direct detection equals the published attack range: 14 km.
- MAV detection is twice that range: 28 km. Alive-MAV tracks are shared with red
  attack UAVs perfectly, instantaneously, without delay, loss, noise, bandwidth,
  or link modelling. Sharing stops immediately when the MAV dies; blue has no
  MAV sharing.
- Horizontal combat-zone radius and position normalization are 28 km.
- Altitude and situation-height normalization use the table-based nominal 6000 m.
- Height reward is `-1` at ground, `0` at 750 m, rises linearly to `1` at 6000 m,
  and remains `1` above 6000 m.
- MAV distances derive from `R=14 km`: danger `0.5R`, safe/optimal `R`, maximum
  `2R`. MAV death is `-200`; each same-step eligible team kill is `+200`, without
  an additional cap.
- The blue policy is a **minimal greedy basic-manoeuvre reconstruction** over
  level, accelerate, decelerate, left turn, right turn, climb, and dive. Its
  0.2 s aircraft/constant-velocity threat predictor remains paper-silent.
- Automatic launch against the selected visible target requires an alive attack
  UAV, inventory, a live target, `1e-6 m < distance <= 14 km`, and a 25 s interval.
- MAV is excluded from combat-unit elimination. Mutual elimination is a draw;
  episode limit is a truncated draw. Resolution occurs at
  `termination_resolution = decision_step_boundary`: all 12 physics frames finish
  before one outcome is computed.

## D. PAPER_SILENT_REQUIRED_PARAMETER

- F-16/F-22 JSBSim model selection.
- Missile initial speed, powered duration, powered acceleration, effective
  quadratic drag, and hit radius.
- Missile-speed reward normalization and global reward scale.
- The remaining MAV safety/support form and blue predictor constants.

## E. REMOVED_UNSUPPORTED_COMPLEXITY

- 20/80 km independent detection assumptions and 50 km independent combat zone.
- Independent position/altitude/situation normalization parameters.
- Three-segment quadratic height curve and optimal/maximum-altitude parameters.
- Independent MAV distance/event parameters and cumulative bonus cap state.
- Four compound blue manoeuvres.
- Duplicate weapon-layer visibility gate and the old 300 m minimum range.
- Independent maximum missile speed and missile lifetime parameters.
- Structural grace and structural death reasons; limit exceedances remain
  diagnostics only and `structural_failures` remains a zero compatibility field.

## F. OPEN_FIDELITY_ITEMS

- Aircraft identity, sensor/communication physics, MAV reward constants, blue
  predictor constants, and missile propulsion/drag/hit engineering remain
  unpublished.
- The blue reconstruction is not claimed to be reference [8]'s exact FSM.
- `paper_nominal` uses only `none`; `paper_5v4_generalization` evaluates only
  low/medium/large with a v3 nominal 5v4 vanilla-HAPPO baseline checkpoint.
- Vanilla-HAPPO baseline results are not TAM-HAPPO results.

PAPER_ENVIRONMENT_EXACTLY_REPRODUCED = false
