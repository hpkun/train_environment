# Blue Greedy FSM Opponent Design

## Purpose

`greedy_fsm` is a scripted blue opponent for environment diagnostics. Its goal
is to be closer to a greedy finite-state controller than `rule_nearest`, while
remaining a low-intrusion policy layer that only outputs the existing high-level
action `[pitch, heading, speed]`.

This is environment completion work, not a new algorithm. It is also not final opponent behavior for paper results; it is an environment diagnostic opponent
until its state coverage and action saturation are validated.
It is an environment component for pre-training readiness checks, not a learned
policy or a neural network.

The current implementation is an initial version. Live paper-aligned diagnostics
can be patrol-only because blue may have no visible red tracks, which is a
visibility/geometry issue rather than automatically an FSM branch bug.

## Relationship To Papers

BRMA-MAPPO uses a rule-based blue side in its training environment. TAM-HAPPO
describes a greedy rule-based or finite-state style opponent. The current
`greedy_fsm` is only a minimal approximation for diagnostics. It is not a full
reproduction of either paper's controller.

## States

- `evade`: entered when the blue observation reports `missile_warning > 0`.
  The policy commands climb, high speed, and a lateral turn intent. The actual
  missile warning and evasion implementation remain owned by the environment.
- `recover_altitude`: entered when the observation indicates low altitude. The
  policy commands climb and medium-high speed. The altitude check is currently
  heuristic because the script-layer policy may receive normalized or variant
  altitude fields; it must be rechecked against the environment's real altitude
  definition before any training use.
- `attack_mav_priority`: entered when visible enemy role/type metadata marks a
  red MAV. The policy turns toward the MAV target. If `enemy_roles` and
  `enemy_types` are missing, it falls back to nearest-target attack.
- `attack_nearest`: entered when any visible enemy exists and no MAV-priority
  target is available. It turns toward the nearest observed enemy with segmented
  speed intent.
- `target assignment`: within one `act()` call, visible targets already assigned
  to another blue agent are skipped when alternatives exist. If the environment
  exposes engaged-target information through `refresh_engaged_targets()`, the
  policy can use it as an additional deconfliction hint.
- `search_acquire`: entered when no visible enemy target exists. It keeps a
  small alternating heading offset and high speed to express contact/search
  intent without reading hidden state.
- `patrol`: entered when no visible enemy is available. It uses a mild turning
  command and medium speed. This is retained as a legacy fallback concept, but
  the default no-target greedy_fsm branch is `search_acquire`.

## Difference From rule_nearest

`rule_nearest` always steers toward the nearest non-zero enemy state and uses a
fixed attack speed. `greedy_fsm` first checks missile warning, altitude recovery,
and optional MAV target metadata before falling back to nearest-target attack.
If no target is visible, it uses `search_acquire` instead of passive patrol. It
also records `OpponentPolicy.last_states` for diagnostics.

## What It Does Not Change

- missile fire-control
- direct missile control
- evasion implementation
- reward
- termination
- PID
- aircraft model
- action dimensionality

## Usage

Use the mode explicitly when running diagnostics:

```powershell
python scripts/diagnose_greedy_fsm_opponent.py --steps 50 --output-json outputs/environment_audit/greedy_fsm_opponent_diagnostic.json
```

The mode name is:

```text
--opponent-policy greedy_fsm
```

`rule_nearest remains default`: `rule_nearest` remains the default
training/evaluation opponent until
`greedy_fsm` is validated and the user explicitly confirms switching.

Controlled branch diagnostics must pass before using `greedy_fsm` in training:

```powershell
python scripts/diagnose_greedy_fsm_controlled_branches.py --output-json outputs/environment_audit/greedy_fsm_controlled_branches.json
```

The geometry/range decision remains unresolved. Do not change initial states or
observation ranges solely because live rollout diagnostics show patrol-only
behavior.

No-target behavior should not be passive patrol-only. `search_acquire` is a
low-intrusion intercept intent: keep the initial heading, add only a small
per-agent deconfliction offset, and fly fast enough to close range. It still
does not read hidden state, does not control missiles, and does not enter the
training protocol by default.

Heading action is a circular absolute heading: `0=north`, `0.5=east`,
`1/-1=south`, and `-0.5=west`. Any `current_heading + offset` command in
`greedy_fsm` must wrap to `[-1, 1]`, not clip. The wrap fix applies to
`search_acquire`, `turn_back`, and attack heading corrections. This matters
near `+/-1`, where clipping can prevent a real turn-back maneuver.

## Open Issues

- Whether blue should prioritize MAV attack in every protocol still needs
  confirmation.
- Target assignment across multiple blue aircraft is not implemented.
- Candidate maneuver sets are not implemented.
- Candidate maneuver scoring should wait until heading wrap and post-pass
  separation diagnostics are verified.
- Candidate actions are not evaluated by immediate reward or lookahead.
- Finite-state transition rules are still minimal and need validation against
  BRMA-MAPPO/TAM-HAPPO assumptions.
- Explicit opponent validation before training is required.
- Visibility/geometry alignment with paper-aligned 3v2/5v4 remains unresolved.
- `greedy_fsm` is not final opponent behavior.
- Alignment with the original BRMA rule opponent may need a separate audit.
