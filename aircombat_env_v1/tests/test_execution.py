from aircombat_env_v1.execution import run_command_hold


class FakeSimulator:
    dt = 1 / 60

    def __init__(self):
        self.steps = 0
        self.controls = (0, 0, 0, 0)

    def state(self):
        return {"time": self.steps * self.dt, "roll": 0.0, "pitch": 0.0,
                "heading": 0.0, "true_airspeed": 250.0, "altitude": 6000.0,
                "alpha": 0.0, "beta": 0.0, "load_factor": 1.0}

    def set_controls(self, *controls):
        self.controls = controls

    def run(self):
        self.steps += 1
        return self.state()


class FakeController:
    def __init__(self):
        self.targets = []

    def step(self, *values):
        self.targets.append(values[4:7])
        return 0.0, -0.04, 0.0, 0.3


def test_command_is_held_for_twelve_physics_steps():
    simulator, controller = FakeSimulator(), FakeController()
    rows, failure = run_command_hold(simulator, controller, (0.1, 0.2, 260), 12)
    assert failure is None
    assert len(rows) == 12
    assert controller.targets == [(0.1, 0.2, 260.0)] * 12
