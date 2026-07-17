# TAM paper environment fidelity

This document defines the fidelity boundary of `tam_paper_env_v1` relative to
Chen, Luo, and Guo, *Aerospace Science and Technology* 176 (2026) 112537.
New outputs are identified by `environment_fidelity_revision =
published_rules_simplified_v2`.

## A. Explicitly implemented from the paper

- The nominal initial states in Tables 5, 6, and 7.
- A 60 Hz simulation loop, 5 Hz decisions, and 12 physics frames per action.
- Four direct-control dimensions with 40 categorical levels per dimension.
- The published action endpoints, situation equations and weights, UAV reward
  equations and weights, PN guidance gains, attack range, launch interval, and
  missile mass and geometry metadata.

The action mapping is consumed by `paper.task`, situation weights by
`paper.situation`, and reward weights by `paper.reward`; these remain code
constants where the existing implementation defines them as such.

## B. Implementations directly determined by published rules

- Episode-limit outcomes are truncated draws when both sides retain combat UAVs.
- Simultaneous elimination of both sides' combat UAVs is a terminated draw.
- The 400 m/s and 9 g values are diagnostic performance-limit exceedances only.
- The 750 m value is consumed by reward and low-altitude diagnostics. The paper
  does not publish a low-level hard-constraint or protection implementation.
- Missile mass, length, and diameter are published metadata, not consumed by the
  current minimal point-mass dynamics.

## C. Removed non-paper mechanisms

- The 0.2 s structural-limit grace and structural death reasons.
- Direct destruction after exceeding 400 m/s or 9 g.
- The 300 m tactical minimum launch range.
- Episode-limit winner selection from survivor or kill counts.
- Perturbed ordinary 2v2/3v2 evaluations.

`structural_failures` remains only as a compatibility output fixed to zero.

## D. Retained minimal paper-silent assumptions

- F-16 and F-22 JSBSim model selection; the paper does not publish these models.
- Combat-zone radius, UAV/MAV detection ranges, observation normalization, and
  incoming-missile slot count.
- Missile initial speed, powered duration, powered acceleration, effective drag,
  maximum speed, lifetime, and hit radius.
- MAV reward thresholds and event constants, plus height-reward approximation
  parameters.
- The blue-side basic manoeuvre set and termination treatment of the MAV role.

These inferred values were not changed or tuned in this revision.

## E. Nominal experiment protocol

`paper_nominal` permits 2v2, 3v2, and 5v4 only with
`initial_perturbation = none`. Ordinary training, periodic evaluation,
independent evaluation, baseline diagnosis, and nominal rule evaluation use this
protocol. Old low-perturbation 2v2 outputs are pre-fidelity diagnostics and must
not be combined with this revision.

## F. 5v4 generalization protocol

`paper_5v4_generalization` is evaluation-only, fixed to scenario 5v4 and levels
`low`, `medium`, and `large`. The formal protocol uses 50 episodes per level,
one trained 5v4 checkpoint, and independent reproducible seed sequences. `none`,
2v2, and 3v2 cannot enter this result set.

## G. Open fidelity items

- The blue FSM is not yet closed against reference [8].
- Sensor ranges remain engineering assumptions.
- MAV reward thresholds remain engineering assumptions.
- Missile initial speed, propulsion, drag, lifetime, and hit radius remain
  engineering assumptions.
- Aircraft model identity remains unpublished.

Therefore this revision identifies and narrows paper-silent assumptions but does
not claim an exact reproduction.

PAPER_ENVIRONMENT_EXACTLY_REPRODUCED = false
