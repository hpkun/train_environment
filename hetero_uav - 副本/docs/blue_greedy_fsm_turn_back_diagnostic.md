# Blue Greedy FSM Turn-Back Diagnostic

## Scope

This note documents the heading-wrap fix for the diagnostic `greedy_fsm`
opponent. It is environment-completion work only. It does not change action
space, initial states, observation range, missile dynamics, reward,
termination, PID, or aircraft XML.

## Heading Semantics

The high-level action is `[pitch, heading, speed]`. The environment maps
`heading` to an absolute heading with `heading * pi`.

Normalized heading is circular:

- `0.0` means north.
- `0.5` means east.
- `1.0` and `-1.0` both mean south.
- `-0.5` means west.

Because heading is circular, heading offsets must wrap, not clip. For example,
`0.9 + 0.5` should become a negative heading near `-0.6`, not a saturated
`1.0`. Clipping near `+/-1` can prevent a real turn-back maneuver because the
command gets stuck at the boundary.

## Current Fix

`OpponentPolicy` uses `_wrap_heading_norm()` for heading increments in:

- `_search_acquire_action`
- `_turn_back_action`
- `_attack_action_from_obs`

Pitch and speed remain clipped to `[-1, 1]`, and the final action vector remains
clipped for safety.

## Diagnostics

Controlled branch diagnostics include explicit wrap cases:

- turn-back from `0.9` with `+0.5` must produce a negative heading;
- turn-back from `-0.9` with `-0.5` must produce a positive heading;
- search-acquire from `0.99` must not get stuck at `1.0`.

Live diagnostics additionally record:

- `heading_wrap_used`
- `turn_back_count`
- `turn_back_heading_delta_mean_abs`
- `turn_back_heading_values_sample`
- `post_pass_separation_m`

If turn-back is triggered but post-pass separation remains large, the next step
is not training. Candidate maneuver scoring should wait until heading wrap is
verified and the remaining separation issue is diagnosed.
