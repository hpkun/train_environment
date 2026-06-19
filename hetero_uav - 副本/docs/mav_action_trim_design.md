# MAV Action Trim Design

## Purpose

Heterogeneous aircraft share the same high-level action space, but A-4 and f16
do not respond identically to the same zero action. A configurable trim lets the
MAV/A-4 zero action sit closer to level flight without changing the action
space.

This is not an RL method and is not reward shaping. It is an environment
calibration for aircraft-type mismatch.

## Evidence

The red-policy zero ACMI path does not load a MAPPO actor, so the observed A-4
altitude loss is not caused by untrained RL. The A-4 zero action drops much more
than f16, while a pitch bias near 0.10 improves MAV altitude retention.

## Design

The trim is config-driven through `action_trim_by_role`:

```yaml
action_trim_by_role:
  mav:
    pitch: 0.10
    heading: 0.0
    speed: 0.0
```

The current paper-aligned V2 configs enable MAV pitch `+0.10`. Balanced configs
are not changed. If a config has no trim entry, behavior remains unchanged.
Effective actions are clipped to `[-1, 1]`.

Agent-specific trim has priority over role trim, and role trim has priority over
type trim when those optional maps are used.

## What It Does Not Change

There is no aircraft XML change, no PID change, no missile change, no reward
change, no termination change, no observation change, and no action space
dimension change.

## How To Diagnose

Use `diagnose_mav_action_trim_effect.py` to compare trim disabled and enabled.
Use `diagnose_mav_flight_stability.py` for the broader case matrix. Use
`export_hetero_tacview_acmi.py` with and without `--disable-config-trim` for
Tacview comparison.
