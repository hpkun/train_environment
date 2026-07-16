from types import SimpleNamespace

from aircombat_env_v1 import aircraft


class FakeEngine:
    def init_running(self):
        pass


class FakePropulsion:
    def get_num_engines(self):
        return 1

    def get_engine(self, index):
        return FakeEngine()

    def get_steady_state(self):
        pass


class FakeFDM:
    constructions = 0

    def __init__(self, data_dir):
        type(self).constructions += 1
        self.values = {}
        self.reset_calls = 0

    def set_debug_level(self, level):
        pass

    def load_model(self, model):
        return True

    def set_dt(self, dt):
        self.dt = dt

    def reset_to_initial_conditions(self, mode):
        self.reset_calls += 1

    def set_property_value(self, name, value):
        self.values[name] = value

    def get_property_value(self, name):
        return self.values.get(name, 0.0)

    def run_ic(self):
        return True

    def get_propulsion(self):
        return FakePropulsion()

    def run(self):
        return True


def test_reset_reuses_the_same_fdm(monkeypatch):
    FakeFDM.constructions = 0
    monkeypatch.setattr(aircraft, "jsbsim", SimpleNamespace(FGFDMExec=FakeFDM))
    simulator = aircraft.AircraftSimulator()
    original = simulator.fdm
    simulator.reset()
    simulator.reset(altitude_m=7000.0)
    assert simulator.fdm is original
    assert FakeFDM.constructions == 1
    assert simulator.fdm.reset_calls == 3  # two cold-start passes, then one reset
