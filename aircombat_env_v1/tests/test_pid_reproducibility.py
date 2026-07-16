import numpy as np
import pytest

from aircombat_env_v1.pid import PIDLoop, PaperAutopilot


def make_autopilot():
    return PaperAutopilot(
        roll_gains={"kp": 0.8, "ki": 0.01, "kd": 0.03},
        pitch_gains={"kp": 0.8, "ki": 0.01, "kd": 0.03},
        speed_gains={"kp": 0.01, "ki": 0.001, "kd": 0.0},
        elevator_trim=-0.04,
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


def test_integral_output_limit_clamps_integral_contribution():
    loop = PIDLoop(0.0, 2.0, 0.0, -10.0, 10.0,
                   integral_output_limit=0.15)
    for _ in range(100):
        loop.step(1.0, 0.1)
    assert abs(loop.ki * loop.integral) <= 0.15


def test_pid_reset_clears_integral_state():
    loop = PIDLoop(0.0, 1.0, 0.0, -10.0, 10.0)
    loop.step(1.0, 0.5)
    assert loop.integral != 0.0
    loop.reset()
    assert loop.integral == 0.0


def test_zero_integral_gain_is_unchanged_by_output_limit():
    limited = PIDLoop(1.0, 0.0, 0.0, -1.0, 1.0,
                      integral_output_limit=0.15)
    unlimited = PIDLoop(1.0, 0.0, 0.0, -1.0, 1.0)
    assert [limited.step(0.2, 0.1) for _ in range(5)] == [
        unlimited.step(0.2, 0.1) for _ in range(5)]


def test_rudder_is_always_zero():
    assert make_autopilot().step(0, 0, 0, 250, 0.1, 0.2, 260, 0.02)[2] == 0.0


def test_zero_errors_return_trim_inputs():
    controls = make_autopilot().step(0, 0, 0, 250, 0, 0, 250, 1 / 60)
    assert controls[1] == pytest.approx(-0.04)
    assert controls[3] == pytest.approx(0.3)


def test_pitch_errors_produce_opposite_increments_around_trim():
    positive = make_autopilot().step(0, 0, 0, 250, 0.1, 0, 250, 1 / 60)[1]
    negative = make_autopilot().step(0, 0, 0, 250, -0.1, 0, 250, 1 / 60)[1]
    assert positive < -0.04 < negative
    assert (positive + 0.04) == pytest.approx(-(negative + 0.04))


def test_elevator_output_is_clipped():
    upper = PaperAutopilot(
        {"kp": 0, "ki": 0, "kd": 0}, {"kp": 100, "ki": 0, "kd": 0},
        {"kp": 0, "ki": 0, "kd": 0}, elevator_trim=0.9,
        throttle_base=0.3, actuator_sign_elevator=1)
    lower = PaperAutopilot(
        {"kp": 0, "ki": 0, "kd": 0}, {"kp": 100, "ki": 0, "kd": 0},
        {"kp": 0, "ki": 0, "kd": 0}, elevator_trim=-0.9,
        throttle_base=0.3, actuator_sign_elevator=1)
    assert upper.step(0, 0, 0, 250, 1.0, 0, 250, 1 / 60)[1] == 1.0
    assert lower.step(0, 0, 0, 250, -1.0, 0, 250, 1 / 60)[1] == -1.0
