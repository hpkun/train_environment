# Missile Launch Contract Audit

## Purpose

This audit checks the current F-22 MAV mainline environment's missile launch
and hit logic. It is diagnostic only and does not modify missile, reward,
termination, action, evasion, PID, aircraft XML, MAPPO, or method modules.

## Paper Launch Conditions

The launch contract checked here is:

- 10 km electro-optical / infrared detection range
- 0.25 s continuous detection / lock before launch
- 0.5 s launch interval
- same-target deconfliction so multiple recent missiles do not attack the same
  target
- rear hemisphere / 3-9 line gate

## Current Implementation Summary

The current environment exposes launch constants and rollout diagnostics through
`uav_env/JSBSim/env.py`. It uses range, AO, TA, lock delay, cooldown, engaged
target deconfliction, launch quality records, and missile termination reasons.

The mainline setup uses F-22 as the MAV with zero missiles and f16 attack UAVs
with two missiles.

## Open Questions

- Whether AO cone angle 45 degrees is explicitly specified by the paper.
- Whether the 500 m minimum launch range is paper-specified or engineering
  protection.
- Whether the launch range should be 2D or 3D; current range comes from
  `get2d_AO_TA_R`.
- No closing-speed gate is currently required by the paper text used here.

## Decision Rule

If there is no blocking mismatch, do not change missile logic. Use the audit
result to decide whether to continue the 100k pilot or first fix a concrete
blocking mismatch.
