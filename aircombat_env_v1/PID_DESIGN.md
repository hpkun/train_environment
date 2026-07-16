# PID design and acceptance boundary

This package implements only a JSBSim F-16 flight-control experiment. It has
no reinforcement learning, combat, rewards, weapons, sensors, or multi-agent
behavior.

## Paper structure versus local engineering

The controller structure comes from BRMA-MAPPO Section 2.4: target pitch,
heading, and speed define a desired direction; that direction is transformed
to body coordinates; roll, pitch, and speed PID channels drive aileron,
elevator, and throttle; rudder has no feedback. Integral limiting and actuator
saturation belong to this reported structure.

The numerical PID gains are not paper parameters. Elevator trim, throttle
base, integral thresholds, derivative filtering, actuator signs, PID frequency,
and all validation thresholds are local engineering choices for this JSBSim
F-16 model or results of offline experiments.

JSBSim uses NED and aerospace body z-down. The paper's z-up desired vector is
therefore sign-flipped on entry to NED and body z is negated before evaluating
the published pitch error. Closed-loop elevator output is local trim plus the
signed pitch PID increment, clipped to `[-1, 1]`.

## Dual-frequency execution

JSBSim physics and PID both run at 60 Hz. The high-level command interface is
called at 5 Hz, and each target is held for exactly 12 physics/PID frames. A
“5 Hz update” means the decision interface is invoked every 0.2 seconds; it
does not mean the command value must change every 0.2 seconds. The deterministic
combined task changes heading every 3 seconds, pitch every 4 seconds, and speed
every 5 seconds.

## Parameter lifecycle

`parameter_status` has exactly three states:

- `initial_guess`: engineering starting values, not experimentally accepted.
- `candidate`: trim or PID optimization was accepted, but complete validation
  has not passed.
- `validated`: all 72 short cases and all eight representative 200-second long
  cases passed the configured local thresholds.

Trim and tuning scripts emit candidate YAML by default. Attitude integrals are
currently fixed at zero; formal tuning is deliberately limited to Roll PD,
Pitch PD, and Speed PI. Each stage compares an optimized candidate against its
incoming baseline and adopts it only when every case completes safely and cost
improves. Explicit accept can update the formal configuration only when all
three stages are accepted and joint validation is valid. Only full validation
may set `validated`.

## Metrics

Tuning records roll, pitch, and speed errors separately, normalized by 30 deg,
10 deg, and 50 m/s. Control energy uses increments around the trim point:
aileron, `elevator-elevator_trim`, rudder, and
`throttle-throttle_base`. Control-rate energy additionally penalizes squared
physical-frame changes in aileron, elevator, and throttle. Step overshoot is measured only after the response
first reaches its target; settling requires the error to remain within its
channel tolerance through the end of that step segment. Heading calculations
use circular angle differences.

Multi-case performance integrals and control energies are averaged. Safety,
saturation, altitude loss, and final errors use their worst-case maxima. Failed
optimization cases receive finite, completion-sensitive penalties so the
optimizer retains a useful gradient, while stage acceptance still requires
zero failures.
