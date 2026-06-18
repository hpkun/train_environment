from __future__ import annotations

import numpy as np
import pytest

from scripts.train_happo_reference import _build_red_alive_mask
from uav_env.JSBSim.env import UavCombatEnv
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


class FakeAircraft:
    def __init__(self, bad_field: str | None = None, bad_value: float = np.nan):
        self.bad_field = bad_field
        self.bad_value = bad_value
        self.is_alive = True
        self.num_left_missiles = 0
        self.crash_called = False

    def _state(self, field: str, size: int, values: list[float]) -> np.ndarray:
        out = np.asarray(values, dtype=np.float64)
        if self.bad_field == field:
            out[0] = self.bad_value
        assert out.shape == (size,)
        return out

    def get_geodetic(self):
        return self._state("geodetic", 3, [120.0, 60.0, 6000.0])

    def get_position(self):
        return self._state("position", 3, [0.0, 0.0, 6000.0])

    def get_velocity(self):
        return self._state("velocity", 3, [250.0, 0.0, 0.0])

    def get_rpy(self):
        return self._state("rpy", 3, [0.0, 0.0, 0.0])

    def get_property_value(self, _name):
        return 0.0

    def crash(self):
        self.crash_called = True
        self.is_alive = False


def _bare_env(sim: FakeAircraft) -> UavCombatEnv:
    env = object.__new__(UavCombatEnv)
    env.agent_ids = ["red_0"]
    env.red_ids = ["red_0"]
    env.blue_ids = []
    env.red_planes = {"red_0": sim}
    env.blue_planes = {}
    env._overload_timers = {"red_0": 0.0}
    env._crashed_this_step = set()
    env._death_reasons = {}
    env._missile_launch_counts = {"red_0": 0}
    env._missile_term_reasons = {"red": {}, "blue": {}}
    env._launch_diag_step = {"red": {}, "blue": {}}
    env._launch_quality_step_records = []
    env._launch_quality_done_step_records = []
    env._death_events_step = []
    env.current_step = 1
    return env


@pytest.mark.parametrize("bad_field", ["geodetic", "position", "velocity", "rpy"])
def test_nonfinite_alive_aircraft_is_crashed(bad_field):
    sim = FakeAircraft(bad_field)
    env = _bare_env(sim)

    env._check_crash_terminations()

    assert sim.crash_called
    assert "red_0" in env._crashed_this_step
    assert env._death_reasons["red_0"] == "Crash_NonFiniteState"


def test_finite_alive_aircraft_is_not_crashed():
    sim = FakeAircraft()
    env = _bare_env(sim)

    env._check_crash_terminations()

    assert not sim.crash_called
    assert "red_0" not in env._crashed_this_step


def test_infinite_alive_aircraft_is_crashed():
    sim = FakeAircraft("position", np.inf)
    env = _bare_env(sim)

    env._check_crash_terminations()

    assert sim.crash_called
    assert env._death_reasons["red_0"] == "Crash_NonFiniteState"


def test_nonfinite_crash_flows_to_info_active_mask_and_zero_ego_observation():
    sim = FakeAircraft("velocity")
    env = _bare_env(sim)
    env._check_crash_terminations()

    info = env._get_info()
    active = _build_red_alive_mask(info, env, env.red_ids)
    assert info["red_0"]["alive"] is False
    assert active.tolist() == [0.0]

    hetero = object.__new__(HeteroUavCombatEnv)
    hetero.red_planes = {"red_0": sim}
    hetero.blue_planes = {}
    hetero.agent_roles = {"red_0": "mav"}
    obs = hetero._build_mav_shared_geo_obs("red_0", [], [])
    assert np.isfinite(obs["ego_geo_state"]).all()
    assert np.all(obs["ego_geo_state"] == 0.0)
