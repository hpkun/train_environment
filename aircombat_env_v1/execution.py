"""Shared 60 Hz PID execution for one held 5 Hz command."""

from __future__ import annotations

import numpy as np


def run_command_hold(simulator, controller, target_command, physics_steps):
    """Hold one (pitch, heading, speed) command for ``physics_steps`` frames.

    The caller invokes this once per 5 Hz decision. The PID still runs on every
    JSBSim physics frame, normally 12 times at 60 Hz.
    """
    target_pitch, target_heading, target_speed = map(float, target_command)
    rows = []
    for _ in range(int(physics_steps)):
        try:
            state = simulator.state()
            controls = controller.step(
                state["roll"], state["pitch"], state["heading"],
                state["true_airspeed"], target_pitch, target_heading,
                target_speed, simulator.dt)
            simulator.set_controls(*controls)
            state = simulator.run()
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return rows, f"simulation_error:{type(exc).__name__}"
        numeric = tuple(state.values()) + tuple(controls)
        if not np.all(np.isfinite(numeric)):
            return rows, "nan_or_inf"
        row = dict(state)
        row.update(target_pitch=target_pitch, target_heading=target_heading,
                   target_true_airspeed=target_speed, aileron=controls[0],
                   elevator=controls[1], rudder=controls[2], throttle=controls[3])
        rows.append(row)
        if state["altitude"] < 1000.0:
            return rows, "altitude_below_1000_m"
        if state["true_airspeed"] < 120.0:
            return rows, "airspeed_below_120_mps"
    return rows, None
