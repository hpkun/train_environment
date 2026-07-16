import numpy as np
import pytest

pytest.importorskip("jsbsim")

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import load_config
from aircombat_env_v1.pid import PaperAutopilot


pytestmark = pytest.mark.integration


def reset_from_config(simulator, config):
    state = config["initial_state"]
    return simulator.reset(
        altitude_m=state["altitude_m"], speed_mps=state["speed_mps"],
        heading_deg=state["heading_deg"], roll_deg=state["roll_deg"],
        pitch_deg=state["pitch_deg"], elevator_trim=config["trim"]["elevator_trim"],
        throttle_base=config["trim"]["throttle_base"])


def test_real_f16_loads():
    simulator = AircraftSimulator()
    assert simulator.fdm is not None


def test_two_resets_reuse_fdm():
    simulator = AircraftSimulator()
    original = simulator.fdm
    reset_from_config(simulator, load_config())
    reset_from_config(simulator, load_config())
    assert simulator.fdm is original


def test_identical_control_sequence_reproduces_trajectory():
    config, simulator = load_config(), AircraftSimulator()
    controls = (0.01, config["trim"]["elevator_trim"], 0.0,
                config["trim"]["throttle_base"])
    trajectories = []
    for _ in range(2):
        reset_from_config(simulator, config)
        simulator.set_controls(*controls)
        trajectories.append([[simulator.run()[key] for key in
                              ("altitude", "true_airspeed", "roll", "pitch")]
                             for _ in range(60)])
    assert np.allclose(trajectories[0], trajectories[1], rtol=1e-10, atol=1e-10)


def test_current_trim_is_finite_for_five_seconds():
    config, simulator = load_config(), AircraftSimulator()
    reset_from_config(simulator, config)
    simulator.set_controls(0.0, config["trim"]["elevator_trim"], 0.0,
                           config["trim"]["throttle_base"])
    states = [simulator.run() for _ in range(5 * 60)]
    assert np.all(np.isfinite([value for state in states for value in state.values()]))


def test_zero_error_autopilot_outputs_current_trim():
    config, simulator = load_config(), AircraftSimulator()
    state = reset_from_config(simulator, config)
    controls = PaperAutopilot.from_config(config).step(
        state["roll"], state["pitch"], state["heading"], state["true_airspeed"],
        state["pitch"], state["heading"], state["true_airspeed"], simulator.dt)
    assert controls[1] == pytest.approx(config["trim"]["elevator_trim"], abs=1e-12)
    assert controls[3] == pytest.approx(config["trim"]["throttle_base"], abs=1e-12)
