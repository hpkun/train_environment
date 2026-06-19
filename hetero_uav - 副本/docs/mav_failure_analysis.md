# MAV Failure Analysis

## Current Status

MAV failure remains the blocking issue before any HAPPO reference v0 1M run.
The 200k HAPPO reference run has some red missile hits and blue deaths, but the
MAV survival rate is still zero in checkpoint evaluation.

This document summarizes the post-200k diagnostics:

- death-event instrumentation;
- F-22 fixed-action stability;
- F-16 MAV surrogate comparison;
- HAPPO checkpoint MAV survival ablations;
- blue target preference against MAV;
- 1M gate decision.

## Death Event Instrumentation

`info["death_events"]` is now emitted on every step. It is an empty list when no
aircraft dies and contains structured records when any aircraft transitions
from alive to dead.

Recorded fields include:

- `agent_id`, `side`, `role`, `aircraft_model`;
- `death_reason`, `death_reason_source`;
- missile hit metadata when available;
- low-altitude, over-G, out-of-bounds, and crash flags when inferable;
- altitude, speed, roll, pitch, and heading snapshots.

This is diagnostic logging only. It does not change reward, termination,
missile dynamics, PID, action space, aircraft XML, observation dimensions, or
HAPPO policy/trainer code.

## Control Stability Sweep

Command:

`python scripts/audit_mav_control_stability_sweep.py --episodes 3 --steps 500`

Summary:

- F-22 MAV stable: `False`;
- F-16 MAV surrogate stable: `False` under the strict roll/stability gate;
- F-22 mean death rate: `1.0`;
- F-16 surrogate mean death rate: `0.142857`;
- F-16 surrogate is more stable than F-22 in the fixed-action sweep.

Representative fixed safe-speed case `[0.0, 0.0, 0.3]` with zero blue:

- F-22 MAV: death rate `1.0`, reason `Crash_LowAlt`;
- F-16 MAV surrogate: death rate `0.0`, but max roll is still near `180 deg`,
  so it does not pass the strict stability gate.

Interpretation: F-22 MAV failure is strongly tied to control/dynamics
stability, but the F-16 surrogate is not clean enough to be declared a final
fix. It is only a candidate diagnostic surrogate.

## MAV Survival Ablation

Command:

`python scripts/audit_happo_mav_survival_ablation.py --episodes 20`

All tested cases have MAV survival rate `0.0`:

- learned MAV/UAV policy;
- fixed safe MAV action `[0.0, 0.0, 0.3]`;
- MAV action scale `0.3`;
- MAV action scale `0.1`;
- F-16 MAV surrogate learned;
- F-16 MAV surrogate with fixed safe MAV action.

Death reasons are now mostly `Crash_LowAlt`. In the best-checkpoint fixed
safe-action cases, many MAV deaths are recorded as `missile_hit`, so a fixed
MAV can become predictable and vulnerable even when the actor is bypassed.

Interpretation:

- fixed safe MAV action does not solve MAV survival;
- action scaling does not solve MAV survival;
- F-16 surrogate does not solve MAV survival in the full HAPPO ablation;
- the failure is not explained by actor output alone.

## Blue Target Preference

Command:

`python scripts/audit_blue_target_preference_against_mav.py --episodes 20`

Result:

- `mav_missile_target_fraction = 0.0`;
- no blue missile launches were recorded in this audit;
- exact blue lock target and selected target fields are unavailable.

Interpretation: current available missile metadata does not support the claim
that blue systematically prioritizes the MAV. Because lock/selected target are
not exposed, the audit cannot fully rule out non-missile pursuit preference.

## 1M Gate Decision

Command:

`python scripts/decide_mav_failure_fix_path.py`

Decision:

- primary failure hypothesis: `f22_control_or_dynamics_instability`;
- recommended next action: use F-16 MAV surrogate for algorithm validation,
  then return to F-22;
- `run_1m_allowed = false`.

Blocking issues:

1. no realistic ablation case reaches MAV survival above `0.5`;
2. safe fixed MAV action does not prove survival;
3. previous HAPPO 1M readiness decision is false.

## Next Step

Do not run 1M yet. The next step should be a narrow F-22 control/dynamics
stability fix or a temporary F-16 MAV surrogate validation branch, not more
HAPPO training.
