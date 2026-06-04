# Blue Greedy FSM Opponent Design

## Purpose

`greedy_fsm` is a scripted blue opponent for environment diagnostics. Its goal
is to be closer to a greedy finite-state controller than `rule_nearest`, while
remaining a low-intrusion policy layer that only outputs the existing high-level
action `[pitch, heading, speed]`.

This is environment completion work, not a new algorithm.

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
  policy commands climb and medium-high speed.
- `attack_mav_priority`: entered when visible enemy role/type metadata marks a
  red MAV. The policy turns toward the MAV target.
- `attack_nearest`: entered when any visible enemy exists and no MAV-priority
  target is available. It turns toward the nearest observed enemy with segmented
  speed intent.
- `patrol`: entered when no visible enemy is available. It uses a mild turning
  command and medium speed.

## Difference From rule_nearest

`rule_nearest` always steers toward the nearest non-zero enemy state and uses a
fixed attack speed. `greedy_fsm` first checks missile warning, altitude recovery,
and optional MAV target metadata before falling back to nearest-target attack.
It also records `OpponentPolicy.last_states` for diagnostics.

## What It Does Not Change

- missile fire-control
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

`rule_nearest` remains the default training/evaluation opponent until
`greedy_fsm` is validated and the user explicitly confirms switching.

## Open Issues

- Whether blue should prioritize MAV attack in every protocol still needs
  confirmation.
- Target assignment across multiple blue aircraft is not implemented.
- Candidate actions are not evaluated by immediate reward or lookahead.
- Alignment with the original BRMA rule opponent may need a separate audit.
