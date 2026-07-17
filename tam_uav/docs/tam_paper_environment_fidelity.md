# TAM paper environment fidelity

Current revision: `published_rules_simplified_v4`. Outputs and checkpoints from
v1-v3 are **pre-v4 fidelity diagnostics** and are rejected by formal resume,
nominal evaluation, and 5v4 generalization entry points.

## A. PUBLISHED_EXACT

- Tables 5-7 nominal initial states.
- 60 Hz physics, 5 Hz decisions, 12 physics frames per action, and 1000 steps.
- Four direct-control dimensions with 40 levels each; throttle endpoints
  `[0.4, 0.9]` and surface endpoints `[-1, 1]`.
- Published UAV speed, angle, distance, dodge, and event reward formulas and
  weights, except for the separately identified paper-silent height function.
- Point-mass PN equations, `Ky=Kz=3`, 30 g limit, 14 km attack range, 25 s
  launch interval, and missile mass/length/diameter metadata.
- The MAV reward contains a capped team-contribution-bonus structure. The
  numerical event scale and scenario-derived cap below are not published values.

## B. SCENARIO_SET_DERIVED

- Incoming-missile capacity is four opposing attack UAVs times two missiles:
  eight fixed observation slots.
- MAV team-kill cap is `200 * initial opposing attack-UAV count`: 400 in 2v2,
  400 in 3v2, and 800 in 5v4. This is `SCENARIO_DERIVED_REWARD_CAP`.

## C. PAPER_SILENT_SIMPLIFICATION

- UAV direct detection is 14 km; MAV detection is 28 km. Alive-MAV tracks are
  shared with red attack UAVs perfectly and instantaneously. Sharing has no
  delay, loss, noise, bandwidth, or link model and stops when the MAV dies.
- Horizontal combat-zone radius and position normalization are 28 km. Leaving
  the zone causes the existing boundary death. Altitude and situation-height
  normalization use 6000 m.
- Height reward is `-1` at ground, `0` at 750 m, rises linearly to `1` at
  6000 m, and remains `1` above 6000 m.
- MAV distance thresholds derive from `R=14 km`: danger `0.5R`, safe/optimal
  `R`, maximum `2R`. Death is `-200`; each eligible unique enemy attack-UAV hit
  before MAV death contributes `+200`, subject to the scenario-derived cap.
- Blue uses a **minimal greedy basic-manoeuvre reconstruction** over exactly
  level, accelerate, decelerate, left turn, right turn, climb, and dive. Its
  0.2 s aircraft and constant-velocity threat predictor remains paper-silent.
- Automatic launch against the selected visible target requires an alive attack
  UAV, inventory, a live target, `1e-6 m < distance <= 14 km`, and 25 s interval.
- MAV is excluded from combat-unit elimination. Mutual elimination is a draw;
  episode limit is a truncated draw. Resolution is at the decision-step boundary.

## D. PAPER_SILENT_DERIVED_SIMPLIFICATION

- Missile timeout is `2 * maximum_attack_range_m / missile_initial_speed_mps`.
  With 14 km and the inferred 500 m/s initial speed it is 56 s. The factor 2
  and initial speed are not published, so this is not uniquely derived.
- `timeout_derivation` is
  `2_times_attack_range_over_inferred_initial_speed`.

## E. PAPER_SILENT_REQUIRED_PARAMETER

- F-16/F-22 JSBSim model selection.
- Missile initial speed, powered duration, powered acceleration, effective
  quadratic drag, and hit radius.
- Missile-speed reward normalization, global reward scale, remaining MAV
  safety/support form, and blue predictor constants.

## Action-grid mathematical semantics

For each surface axis, `control(i) = -1 + 2i/39`. Therefore index 19 is
`-1/39`, index 20 is `+1/39`, and no one of the 40 equally spaced endpoint-
inclusive levels is exactly zero. Index 20 is the `nearest_positive_center`,
not exact neutral or zero surface. The level action remains `[24,20,20,20]` as
a level approximation using nearest-positive-center control surfaces. A dead
agent's `[20,20,20,20]` is an inactive placeholder and is never applied to a
living-aircraft dynamics step. The action space is not changed.

## F. REMOVED_UNSUPPORTED_COMPLEXITY

- Independent 20/80 km detection, 50 km combat zone, and independent
  normalization parameters.
- Quadratic height curve and optimal/maximum-altitude parameters.
- Configurable MAV distance/death/kill/cap parameters. The cap state now only
  implements the published structural requirement with scenario-derived scale.
- Four compound blue manoeuvres, duplicate weapon visibility gate, old 300 m
  minimum range, independent missile maximum speed, and configured lifetime.
- Structural grace/death logic; limit exceedances remain diagnostics only.

## Reference [8] evidence boundary

Jian-dong Zhang, Yi-fei Yu, Li-hui Zheng, Qi-ming Yang, Guo-qing Shi, Yong Wu,
“Situational continuity-based air combat autonomous maneuvering
decision-making”, *Defence Technology*, Volume 29, 2023, Pages 66-79,
DOI: 10.1016/j.dt.2022.08.010.

Reference [8] primarily studies an LSTM-DQN method based on situational
continuity and includes comparison with a statistical-principle decision
method. The TAM paper describes blue as a rule-based finite-state machine that
evaluates predefined basic manoeuvres by immediate reward. Available repository
evidence does not uniquely recover the FSM state set, transitions, state
durations, complete manoeuvre set, four-dimensional indices, or coupling between
shooting and evasion. The current seven-manoeuvre policy is therefore only a
`minimal_greedy_basic_manoeuvre_reconstruction`; its predictor remains
paper-silent. It is not an exact reference-[8] FSM.

`REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED = false`

## G. OPEN_FIDELITY_ITEMS

- Exact reference-[8] blue FSM, state transitions, durations, complete
  manoeuvre/action mapping, and shooting/evasion coupling.
- Aircraft identity, sensor and communication physics, MAV numerical reward
  constants, blue predictor constants, and missile propulsion/drag/hit details.
- Vanilla-HAPPO baseline results are not TAM-HAPPO results.

`PAPER_ENVIRONMENT_EXACTLY_REPRODUCED = false`
