# MAPPO Baseline Implementation Audit

## Purpose

This document records the baseline implementation audit for the current MAPPO
training code. The audit is about training logic only. It does not change the
environment, reward, missile logic, action space, evasion, PID, aircraft XML, or
network structure.

## Relation To BRMA-MAPPO

The current implementation is a MAPPO baseline in the BRMA-MAPPO sense: it uses
a shared actor, centralized critic, GAE, PPO clipping, Gaussian continuous
actions, and parameter sharing across red agents.

This is acceptable as a baseline after basic training-logic issues are fixed.

## Relation To TAM-HAPPO

The current implementation is not the final proposed method for the
heterogeneous TAM-HAPPO paper. It does not implement temporal features,
attention, GRU memory, or HAPPO-style sequential heterogeneous policy updates.

## Expected Blocking Issues

The audit script is expected to flag these implementation risks when present:

- dead agents may contribute to policy loss if a slot-valid mask is used instead
  of an alive mask;
- single red agent death may truncate team done / GAE if team done is computed
  from any per-agent done;
- V2 actor/critic dimensions or action_dim mismatches would block baseline use;
- train/eval observation adapter mismatch would block baseline use.

## Recommended Fix Order

1. alive mask
2. team done / GAE
3. action distribution / log_prob
4. if the MAPPO baseline still fails, then consider entity attention

The clipped Gaussian issue should be audited before changing the policy:
sampling from a Normal distribution, clipping the action, and computing log_prob
on the clipped action can create a PPO likelihood mismatch.

This is not the final method module.
