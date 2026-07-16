import numpy as np
from gymnasium import spaces

from aircombat_env_v1.geometry import NEU2LLA
from aircombat_env_v1.observation import build_observation
from aircombat_env_v1.scenario import ORIGIN


def state(north, speed, heading=0.0):
    lon, lat, alt = NEU2LLA(north, 0.0, 6000.0, *ORIGIN)
    return {
        "longitude": lon, "latitude": lat, "altitude": alt,
        "roll": 0.0, "pitch": 0.0, "heading": heading,
        "true_airspeed": speed, "v_north": speed, "v_east": 0.0,
        "v_down": 0.0,
    }


def test_observation_is_finite_float32_and_in_space():
    observation = build_observation(state(0, 270), state(5000, 240), 0.2, 0.0)
    space = spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32)
    assert observation.shape == (20,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert space.contains(observation)
