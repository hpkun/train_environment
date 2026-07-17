# TAM paper environment fidelity

This document records the current fidelity boundary for the isolated
`tam_paper_env_v1` environment relative to Chen, Luo, and Guo,
*Aerospace Science and Technology* 176 (2026) 112537.

## A. Explicitly implemented from the paper

- JSBSim aircraft dynamics at 60 Hz.
- Agent decisions at 5 Hz, with 12 physics frames per action.
- A 1000-decision-step episode limit.
- Four-dimensional actions ordered as throttle, aileron, elevator, and rudder.
- Forty discrete levels per action dimension.
- Throttle range `[0.4, 0.9]`; aileron, elevator, and rudder ranges `[-1, 1]`.
- Published aircraft limits and parameters from Table 2.
- Published missile metadata and guidance parameters from Table 3.
- Nominal scenario initial states from Tables 5, 6, and 7.
- Situation assessment equations 10-12 and the published target-selection weights.
- The published UAV reward equations and weights.
- `low`, `medium`, and `large` perturbations are reserved for the paper's 5v4
  generalization experiments. Ordinary training and evaluation use `none`.

## B. Extra mechanisms removed in this alignment pass

- `structural_limit_grace_s`.
- Direct aircraft destruction after exceeding 400 m/s.
- Direct aircraft destruction after exceeding 9 g.
- The 300 m tactical minimum launch range.
- Episode-limit winner selection based on surviving-unit or kill counts.
- `low` perturbation as the default for ordinary evaluation.

Speed and overload exceedances remain diagnostic counters. Actual ground impact,
non-finite state handling, combat-zone handling, elimination termination, the
14 km maximum attack range, and the 25 s launch interval remain active.

## C. Necessary assumptions not published by the paper

The following implementation choices remain because the paper does not provide
enough detail to remove or uniquely determine them:

- Selection of the F-16 and F-22 JSBSim models.
- Combat-zone radius.
- UAV and MAV detection ranges.
- Missile initial speed.
- Missile powered-flight duration.
- Missile powered acceleration.
- Effective missile drag parameter.
- Maximum missile speed.
- Missile lifetime.
- Hit radius.
- Observation normalization constants.
- Maximum incoming-missile observation slots.
- MAV reward thresholds and event constants.
- The blue-side basic manoeuvre set.
- The role of the MAV in episode termination semantics.

These assumptions were not tuned or otherwise changed in this alignment pass.
This pass also does not resolve the remaining fidelity questions concerning the
blue FSM, sensor engineering assumptions, MAV reward assumptions, or missile
engineering parameters.
