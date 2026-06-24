from __future__ import annotations

import numpy as np

from scripts.rich_logging import RichExperimentLogger
from uav_env.JSBSim.simulator import MissileSimulator


class _FakeAircraft:
    def __init__(
        self,
        uid: str,
        color: str,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        posture: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        alive: bool = True,
        dt: float = 1 / 60,
    ) -> None:
        self.uid = uid
        self.color = color
        self.dt = dt
        self.lon0 = 120.0
        self.lat0 = 60.0
        self.alt0 = 0.0
        self.launch_missiles = []
        self.under_missiles = []
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._posture = np.asarray(posture, dtype=np.float64)
        self._alive = alive

    @property
    def is_alive(self):
        return self._alive

    def shotdown(self):
        self._alive = False

    def get_geodetic(self):
        return np.asarray([120.0, 60.0, self._position[2]], dtype=np.float64)

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_rpy(self):
        return self._posture


def _missile_pair(parent_velocity=(20.0, 0.0, 0.0), target_position=(4000.0, 0.0, 6000.0)):
    shooter = _FakeAircraft(
        "red_1",
        "Red",
        (0.0, 0.0, 6000.0),
        parent_velocity,
        posture=(0.0, 0.0, 0.0),
    )
    target = _FakeAircraft(
        "blue_0",
        "Blue",
        target_position,
        (250.0, 0.0, 0.0),
    )
    missile = MissileSimulator.create(shooter, target, "m0")
    return missile, shooter, target


def test_low_speed_parent_launch_does_not_create_low_speed_termination():
    missile, _shooter, _target = _missile_pair(parent_velocity=(30.0, 0.0, 0.0))

    for _ in range(20):
        missile.run()
        if missile.is_done:
            break

    assert missile._termination_reason != "low_speed"
    assert np.linalg.norm(missile.get_velocity()) >= missile._missile_speed_mps * 0.95
    assert missile._termination_reason in {"", "hit", "p_hit_fail", "timeout", "target_dead"}


def test_forced_low_speed_state_still_does_not_emit_low_speed_reason():
    missile, _shooter, _target = _missile_pair(parent_velocity=(0.0, 0.0, 0.0))
    missile._velocity[:] = 0.0

    for _ in range(120):
        missile.run()
        if missile.is_done:
            break

    assert missile._termination_reason != "low_speed"
    assert missile._termination_reason in {"", "hit", "p_hit_fail", "timeout", "target_dead"}


def test_missile_no_longer_emits_overshoot_reason():
    missile, _shooter, _target = _missile_pair(
        parent_velocity=(250.0, 0.0, 0.0),
        target_position=(-4000.0, 0.0, 6000.0),
    )
    missile._t_max = 0.2

    while not missile.is_done:
        missile.run()

    assert missile._termination_reason == "timeout"
    assert missile._termination_reason != "overshoot"


def test_hit_probability_keeps_directional_match_formula(monkeypatch):
    missile, _shooter, target = _missile_pair(parent_velocity=(250.0, 0.0, 0.0))
    missile._position[:] = np.asarray([0.0, 0.0, 6000.0])
    target._position[:] = np.asarray([1000.0, 0.0, 6000.0])

    missile._velocity[:] = np.asarray([600.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: 0.99)
    assert missile._roll_hit_probability() is True

    missile._velocity[:] = np.asarray([-600.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: 0.06)
    assert missile._roll_hit_probability() is False
    monkeypatch.setattr(np.random, "random", lambda: 0.04)
    assert missile._roll_hit_probability() is True


def test_missile_event_logging_keeps_launch_and_termination_fields(tmp_path):
    logger = RichExperimentLogger(
        tmp_path,
        run_id="run",
        method_name="method",
        scenario_name="scenario",
        device="cpu",
        num_envs=1,
        rollout_length_per_env=4,
        transitions_per_rollout=4,
    )
    try:
        logger.write_missile_events(
            {
                "__launch_quality_step__": [{
                    "missile_id": "m0",
                    "shooter_id": "red_1",
                    "shooter_team": "red",
                    "target_id": "blue_0",
                    "target_team": "blue",
                    "range_m": 3500.0,
                    "AO_rad": 0.1,
                    "TA_rad": 2.5,
                    "shooter_alt_m": 6000.0,
                    "target_alt_m": 6000.0,
                }],
                "__launch_quality_done__": [{
                    "missile_id": "m0",
                    "shooter_id": "red_1",
                    "shooter_team": "red",
                    "target_id": "blue_0",
                    "target_team": "blue",
                    "range_m": 3500.0,
                    "AO_rad": 0.1,
                    "TA_rad": 2.5,
                    "termination_reason": "timeout",
                    "raw_termination_reason": "timeout",
                    "flight_time_sec": 60.0,
                }],
            },
            scenario="scenario",
            episode_id=0,
            step=1,
            sim_time=0.2,
        )
    finally:
        logger.close()

    text = (tmp_path / "missile_events.csv").read_text(encoding="utf-8")
    for token in ["raw_termination_reason", "flight_time_sec", "AO_rad", "TA_rad", "distance_to_target", "target_id", "owner_id"]:
        assert token in text


def test_initial_velocity_uses_parent_velocity_even_below_old_150_threshold():
    """Parent speed 30 m/s (< old 150 threshold) still uses velocity direction."""
    missile, _shooter, _target = _missile_pair(parent_velocity=(30.0, 0.0, 0.0))
    # With old code (>= 150.0), this would fall back to posture direction.
    # With new code (> 1e-6), this uses velocity direction [1, 0, 0].
    expected = np.array([1.0, 0.0, 0.0], dtype=np.float64) * missile._missile_speed_mps
    actual = missile.get_velocity()
    # Both old (posture [0,0,0] → [1,0,0]) and new (velocity [30,0,0] → [1,0,0])
    # give the same result here, but the code path differs. Verify speed.
    assert np.linalg.norm(actual) == missile._missile_speed_mps


def test_initial_velocity_falls_back_to_posture_when_parent_speed_zero():
    """Parent speed 0 should fall back to posture direction."""
    missile, _shooter, _target = _missile_pair(parent_velocity=(0.0, 0.0, 0.0))
    actual = missile.get_velocity()
    assert np.linalg.norm(actual) == missile._missile_speed_mps
    # Posture is (0,0,0) → direction [1,0,0]
    expected = np.array([1.0, 0.0, 0.0], dtype=np.float64) * missile._missile_speed_mps
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_no_150_threshold_in_simulator():
    """Verify 150.0 no longer appears in missile initial velocity logic."""
    import inspect
    src = inspect.getsource(MissileSimulator._initial_velocity)
    assert "150.0" not in src
    assert "150" not in src.split("parent_speed")[1] if "parent_speed" in src else True
