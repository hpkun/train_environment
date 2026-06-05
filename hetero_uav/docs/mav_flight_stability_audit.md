# MAV Flight Stability Audit

## Purpose

This audit separates MAV/A-4 flight instability causes from policy training
effects. The specific Tacview case uses red-policy zero and does not load a
MAPPO actor. Therefore an A-4 crash in that scene is not caused by untrained RL.

This is not training and does not apply a fix. It only records whether the A-4
MAV can remain controllable under zero action, small pitch bias, speed-up, and
bounded random actions.

## Key Distinction

The ACMI export command with red-policy zero sends scripted high-level actions
directly to the environment. It does not use MAPPO network output. An untrained
RL actor can also crash aircraft during training, but that is a separate issue.

## Relationship To Papers

In the TAM-HAPPO-style heterogeneous setting, the MAV is a support platform. It
should not disappear from a basic scripted route before the combat setting is
meaningful.

The current BRMA-style JSBSim/PID base must be diagnosed with A-4 because that
model may not match the F-16-oriented high-level action and PID assumptions.

## Diagnostic Cases

The environment-level audit runs:

- `zero_all`
- `mav_zero_attack_zero`
- `mav_pitch_bias_005`
- `mav_pitch_bias_010`
- `mav_level_speed_up`
- `red_random_bounded`

The single-aircraft comparison additionally checks A-4 and f16 under zero
action and pitch bias.

## How To Interpret

A zero action crash suggests an environment, controller, model, or initial
condition mismatch. It should not be attributed to an untrained RL actor in the
red-policy zero ACMI scenario.

If pitch bias improves stability, the next discussion should be MAV trim,
target-pitch bias, or safe default behavior. If only bounded random action
crashes, exploration range is the likely issue. If A-4 drops much more than f16,
the issue is model-specific control integration.

The follow-up calibration uses config-driven `action_trim_by_role` for the MAV
in paper-aligned V2 configs. The diagnostic can still reproduce old behavior
with `--disable-config-trim`.

## Next Actions

If zero action fails but pitch bias works, discuss a minimal MAV trim or safe
default after the audit. If all A-4 cases fail, revisit A-4 model/control
integration. Do not immediately change reward or algorithm.
