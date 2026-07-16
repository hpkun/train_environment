import numpy as np
import pytest

from aircombat_env_v1.pid import PIDLoop, PaperAutopilot


def make_autopilot():
    return PaperAutopilot(
        roll_gains={"kp": 0.8, "ki": 0.01, "kd": 0.03},
        pitch_gains={"kp": 0.8, "ki": 0.01, "kd": 0.03},
        speed_gains={"kp": 0.01, "ki": 0.001, "kd": 0.0},
        throttle_base=0.3,
        actuator_sign_elevator=-1,
    )


def test_reset_reproduces_control_sequence():
    autopilot = make_autopilot()
    inputs = (0.0, 0.0, 0.0, 245.0, 0.1, 0.2, 250.0, 1.0 / 60.0)
    first = [autopilot.step(*inputs) for _ in range(20)]
    autopilot.reset()
    second = [autopilot.step(*inputs) for _ in range(20)]
    assert np.allclose(first, second)


def test_pid_rejects_invalid_time_step():
    loop = PIDLoop(1.0, 0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        loop.step(0.2, 0.0)


def test_rudder_is_always_zero():
    assert make_autopilot().step(0, 0, 0, 250, 0.1, 0.2, 260, 0.02)[2] == 0.0
