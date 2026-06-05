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

One 100-step diagnostic is not enough to decide geometry or sensor changes.
Use a horizon sweep such as 100/250/500 steps before changing initial states or
observation ranges.

- If blue observes red by 250 or 500 steps, the issue is mainly contact time
  under the current no-training policy.
- If blue still cannot observe red by 500 steps, then initial geometry or direct
  observation range becomes an environment protocol decision.
- Do not adjust geometry before greedy_fsm has explicit search/acquisition
  behavior.

1. Add explicit blue search/acquisition behavior.
2. Run horizon-sweep visibility diagnostics.
3. Decide whether initial geometry or observation ranges are protocol issues.
4. Only then validate greedy_fsm as a possible training opponent.
