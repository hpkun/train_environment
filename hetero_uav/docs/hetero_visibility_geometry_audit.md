# Hetero Visibility / Geometry Audit

## Purpose

Diagnose initial geometry and observation visibility in paper-aligned and
balanced scenarios.  Determine whether greedy_fsm patrol-only behaviour
is caused by a strategy bug or by blue agents lacking visible enemy tracks.

## Metrics

- red_enemy_observed / blue_enemy_observed — any agent has visible enemy
- direct tracks vs MAV shared tracks
- first observed step
- visibility fractions over rollout

## Interpretation

- red sees via MAV shared but blue sees nothing: information asymmetry
- both sides see nothing for long: initial geometry too distant or
  patrol policy too weak
- blue sees enemy but greedy_fsm stays in patrol: greedy_fsm target
  selection bug

## Relationship to Papers

- TAM-HAPPO uses MAV for situation support
- BRMA-MAPPO uses sensor/rule opponent
- V2 is an abstract visibility/share model, so auditing visibility
  is required before training

## Next Actions After Audit

1. Adjust initial geometry if too distant
2. Adjust observation ranges if needed
3. Improve blue patrol / target acquisition
4. Only then validate greedy_fsm as training opponent
