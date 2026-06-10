# HAPPO 3v2 Reference Validation Plan

## Purpose

This project is a zero-shot heterogeneous MAV/UAV cooperative air-combat
experiment. It is not a full air-combat engineering system and it is not a
low-level flight-control reinforcement-learning project.

The next validation path is HAPPO-style 3v2 reference validation: check whether
a paper-informed heterogeneous policy setup can produce reasonable 3v2 combat
behavior before running larger 5v4 transfer experiments.

## Why Shared MLP Is Not Enough

The current shared MLP MAPPO baseline is runnable, but the 1M best checkpoint
does not prove combat effectiveness. The 100-episode re-evaluation is dominated
by timeout outcomes, MAV survival remains zero, and blue elimination remains
zero.

Timeout survival should not be reported as successful air combat. A valid
environment-method pair should show MAV support, UAV engagement, missile hits,
blue deaths, and non-timeout combat outcomes.

## Why HAPPO 3v2 Is The Right Validation Step

The heterogeneous MAV/UAV paper is mainly useful here as a method and protocol
reference: MAV and UAV have different roles, observations should support
cooperation, and the method should handle heterogeneous agents. HAPPO-style
separate actors are a cleaner test than a single shared MLP actor because the
MAV support role and UAV attack role should not be forced through one identical
policy head.

This is still only a reference validation step, not full TAM-HAPPO.

## Similarities With The Heterogeneous Paper

- 3v2 setup: one MAV with two UAVs against two UAV opponents.
- MAV is a support agent and should not be the main missile shooter.
- UAVs are the attack agents.
- The key experimental question is heterogeneous composition and later
  zero-shot transfer.
- The method direction is role-aware or heterogeneous-policy learning.

## Differences From The Heterogeneous Paper

- Current action interface remains high-level `[pitch, heading, speed]` with
  PID, not low-level throttle/aileron/elevator/rudder.
- Current missile launch and evasion remain BRMA-style scripted mechanics.
- Current first-stage plan does not include temporal attention or full
  TAM-HAPPO.
- Current aircraft are engineering approximations: F-22 MAV and F-16 UAVs.
- Current first-stage reward should stay `brma_legacy` unless a separate
  reference reward is explicitly implemented and audited.

These differences are acceptable for the next validation step because changing
action interface, missile mechanics, and algorithm at the same time would make
the result hard to interpret.

## HAPPO Reference v0 Scope

HAPPO reference v0 should be the smallest coherent method change:

- separate MAV actor;
- separate UAV actor;
- centralized critic;
- sequential HAPPO-style actor update;
- continuous 3D Gaussian action output;
- active-agent mask and team-done logic preserved;
- no attention in the first stage;
- no GRU or temporal module in the first stage;
- no observation-dimension change in the first stage.

The first smoke test should only verify that this setup can collect rollout,
update once, save, load, and evaluate without NaN.

## Reward Position

`happo_ref_v0` reward is a design item, not implemented in this audit.

The current recommended first HAPPO validation should keep `brma_legacy` to
avoid mixing method validation with reward redesign. A role reward can be
revisited only after the HAPPO smoke path is stable.

## Success Criteria

HAPPO 3v2 validation should not be judged by timeout red-win rate alone. The
minimum useful signals are:

- MAV does not always die early;
- UAVs create attack windows and fire effective missiles;
- blue deaths are not always zero;
- episodes are not all timeout draws;
- combat deaths dominate over crash or control anomalies;
- ACMI shows MAV supporting while UAVs engage.

## Training Protocol

1. Run HAPPO reference v0 smoke only.
2. If smoke passes, run a 200k 3v2 validation.
3. If the 200k run shows MAV survival, UAV engagement, and non-timeout combat,
   then consider a 1M run and 5v4 zero-shot transfer.

## Stop Doing

- Do not keep tuning shared MLP MAPPO as the main evidence path.
- Do not run more 500k or 1M jobs that only inspect the latest checkpoint.
- Do not switch now to low-level 4D control.
- Do not implement full TAM-HAPPO before HAPPO reference v0 is validated.
- Do not add more broad engineering audit scripts unless they directly answer
  the HAPPO 3v2 validation question.
