# F-22 MAV Control Response Audit

## Purpose

This audit verifies whether the F-22 MAV responds to high-level actions inside
the main `HeteroUavCombatEnv` paper-aligned configuration. It does not modify
the model, PID, action space, reward, termination, missile logic, evasion, or
MAPPO.

## Why This Matters

If F-22 does not respond to control actions in the actual heterogeneous
environment, missile-launch auditing and training results are not trustworthy.
The model must first be proven to react to climb, descend, turn, and speed
commands in the same environment used by the main experiment.

## Checks

The audit checks:

- climb / descend response
- turn-left / turn-right heading response
- speed-up / slow-down response
- crash and NaN status
- F100 / F119 / f15 / f22 resource status
- `action_trim_by_role.mav.pitch`

The current MAV pitch trim is reported because it was introduced as an A-4
stability carryover. Whether F-22 still needs that trim is not verified in this
round, and the audit does not change it.

## Decision Rule

Missile audit should wait until F-22 control response is verified. If response
is not clearly separated across scenarios, the next step should be F-22 control
path, trim, and throttle diagnosis. Do not train or run long experiments from a
failed or inconclusive control-response audit.
