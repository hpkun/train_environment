# PID design boundary

This directory implements only the flight-control structure described in
BRMA-MAPPO Section 2.4. It does not implement reinforcement learning, combat,
weapons, sensors, rewards, or multi-agent behavior.

## Explicitly stated by the paper

The command consists of target pitch, target heading, and target speed. The
target direction vector is
`[cos(theta)cos(psi), cos(theta)sin(psi), sin(theta)]` in the paper's z-up
inertial coordinates. It is transformed into body coordinates; roll and pitch
direction errors are computed with `atan2(y, x)` and `atan2(z_up, x)`.

The structure has roll, pitch, and speed PID channels. Rudder has no feedback
and remains zero. Integral limiting and actuator saturation are part of the
reported structure.

## Not specified by the paper

The numerical PID gains, throttle base, integral-separation thresholds,
derivative filter, F-16 actuator signs, and PID execution frequency are not
reported. Every such value in `f16_pid_v1.yaml` is therefore an engineering
choice or a result of offline tuning on the local JSBSim F-16, never a paper
gain. `initial_guess_only` remains true until the scripts produce accepted
local results.

## Coordinate and actuator conventions

JSBSim and this package use NED and aerospace body z-down axes. Consequently,
the target vector's z component changes sign on entry to NED, and body z is
negated before evaluating the paper's pitch error. The initial elevator sign
is an engineering hypothesis; `check_actuator_signs.py` reports measured
directions but deliberately does not change configuration.

The 60 Hz physics/PID frequency and 5 Hz command frequency follow the local
interpretation recorded in the configuration. The timing structure is cited
to TAM-HAPPO Section 4.2; numerical PID behavior is still locally engineered.
