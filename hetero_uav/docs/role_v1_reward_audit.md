# role_v1 Reward Audit

## Purpose

This document records the diagnostic scope for the `role_v1` reward ablation.
The audit compares the completed `role_v1` 50k run against the completed
`brma_legacy` 50k run and checks whether the current role-aware overlay matches
the intended MAV/UAV reward direction.

This round does not modify reward, termination, missile launch, missile
dynamics, action space, PID, aircraft XML, or MAPPO network code.

## Current 50k Result Summary

The completed `role_v1` 50k run finished without NaN and with valid actor/critic
dimensions, but latest evaluation is weaker than the comparable `brma_legacy`
50k run:

- `brma_legacy` 3v2: red_win 0.7, blue_win 0.0, timeout 1.0, red_alive 2.0
- `brma_legacy` 5v4: red_win 0.4, blue_win 0.3, timeout 1.0, red_alive 3.5
- `role_v1` 3v2: red_win 0.0, blue_win 1.0, MAV survival 0.0
- `role_v1` 5v4: red_win 0.0, blue_win 1.0, MAV survival 0.0

The training curve shows some mid-run survival signal, but the latest policy
does not retain it. This makes immediate long training a poor next step.

## Relation To BRMA-MAPPO Reward

The `brma_legacy` reward is the inherited BRMA-style baseline. It mainly
preserves flight stability, combat geometry, missile/combat outcomes, and
terminal outcome incentives. It does not explicitly distinguish MAV support
from UAV attack roles.

`role_v1` is intended to be an overlay on that baseline, not a replacement of
the base environment mechanics.

## Relation To Heterogeneous MAV/UAV Reward

The heterogeneous MAV/UAV reward idea emphasizes different role objectives:

- MAV: survival, support, and event/team contribution.
- UAV: height/speed/angle/distance/dodge/event style combat behavior.

Current `role_v1` covers some of this direction through MAV survival/support
and UAV attack-window/kill/death signals, but it is not a full reproduction of
the heterogeneous paper reward. It also intentionally avoids complex missile
dodge reward because the current actor does not have full missile geometry
observation.

## Why No Direct Reward Change In This Round

The failed `role_v1` 50k result does not by itself identify which component is
wrong. The immediate task is to measure component trigger frequency, compare
scales against `brma_legacy`, and identify implementation issues such as support
being based on alive enemies rather than observed enemies.

Reward changes should follow the audit findings instead of empirical guessing.

## Next Minimal Change Direction

The expected next step is to apply the smallest reward change supported by
`outputs/reward_audit/role_v1_reward_effects_audit.json`, then run a short pilot
again. The first likely target is to make MAV support depend on observed/shared
enemy tracks instead of `enemy_alive_mask`.
