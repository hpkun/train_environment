# Reward / Termination Audit

## Purpose

This audit checks whether the current BRMA-inherited reward and termination
behavior is suitable for the heterogeneous MAV/UAV task. It is diagnostic only:
this round is not modifying reward, termination, missile, evasion, action, PID,
or aircraft XML behavior.

## Relation To Papers

BRMA-MAPPO uses team reward, win rate, survival, and combat metrics around a
cooperative air combat environment. TAM-HAPPO-style heterogeneous settings
emphasize different role objectives: MAV survival and shared support value,
UAV attack behavior, and cooperative task completion.

For this project, the key question is whether the current reward actually
expresses those role objectives, or whether MAV/UAV differences are only
visible in metadata and post-hoc metrics.

## Current Suspected Issues

- Reward and termination are largely inherited from the BRMA environment.
- MAV has no missiles, but the reward may not explicitly value MAV shared
  observation support.
- Terminal reward is suspected to be alive-count based, which may not separate
  MAV support from UAV attack.
- MAV survival may currently be a metric rather than an explicit training
  objective.
- Timeout outcome needs a clear interpretation, especially when alive advantage
  is used in combat metrics.

## Audit Outputs

The diagnostic script writes:

- `outputs/environment_audit/reward_termination_audit.json`
- `outputs/environment_audit/reward_termination_audit.md`

Warnings are design inputs for the next environment step. They do not mean the
code failed, and they should not automatically trigger reward changes without
review.

## Next Step After Audit

Review the audit findings and decide whether minimal heterogeneous reward
shaping is needed. This should happen before training and before any
attention/HAPPO/GRU or other method-module work.
