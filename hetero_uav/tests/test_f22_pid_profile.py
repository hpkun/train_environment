"""Tests for F22 MAV 3D PID profile mechanism."""

import numpy as np
import pytest

from uav_env.JSBSim.pid_controller import (
    PIDController,
    PIDLoop,
    F22MavEnergyPIDController,
    F22_MAV_ENERGY_DEFAULT_GAINS,
)


class TestPIDControllerParametrization:
    """Verify that PIDController accepts configurable gains."""

    def test_f16_default_gains_applied(self):
        pid = PIDController(dt=1.0 / 60)
        assert pid.elevator_sign == -1
        assert pid.throttle_min == 0.0
        assert pid.throttle_max == 1.0
        assert pid._roll_pid.kp == 0.15
        assert pid._pitch_pid.kp == 2.5
        assert pid._velocity_pid.kp == 0.04

    def test_custom_gains_applied(self):
        pid = PIDController(
            dt=1.0 / 60,
            roll_kp=0.2, roll_ki=0.3, roll_kd=0.01,
            pitch_kp=1.5, pitch_ki=0.4, pitch_kd=0.05,
            vel_kp=0.05, vel_ki=0.02, vel_kd=0.001,
            elevator_sign=+1,
            throttle_min=0.3, throttle_max=0.95,
        )
        assert pid.elevator_sign == +1
        assert pid.throttle_min == 0.3
        assert pid.throttle_max == 0.95
        assert pid._roll_pid.kp == 0.2
        assert pid._pitch_pid.kp == 1.5
        assert pid._velocity_pid.kp == 0.05

    def test_partial_custom_gains_fallback_to_f16_defaults(self):
        pid = PIDController(dt=1.0 / 60, roll_kp=0.25, elevator_sign=+1)
        assert pid._roll_pid.kp == 0.25   # custom
        assert pid._pitch_pid.kp == 2.5     # default F16
        assert pid._velocity_pid.kp == 0.04  # default F16
        assert pid.elevator_sign == +1
        assert pid.throttle_min == 0.0  # default F16


class TestF22MavEnergyPIDController:
    """Verify F22-specific PID profile."""

    def test_requires_elevator_sign(self):
        with pytest.raises(ValueError, match="elevator_sign must be set"):
            F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=None)

    def test_default_gains_differ_from_f16(self):
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        # F22 gains are different from F16 defaults
        assert pid._roll_pid.kp != 0.15   # F16 default
        assert pid._pitch_pid.kp != 2.5   # F16 default
        assert pid._velocity_pid.kp != 0.04  # F16 default
        # F22 has throttle floor
        assert pid.throttle_min == 0.65
        assert pid.throttle_max == 1.0
        assert pid.low_speed_throttle_floor == 0.95

    def test_f22_gains_match_expected_defaults(self):
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        g = F22_MAV_ENERGY_DEFAULT_GAINS
        assert pid._roll_pid.kp == g["roll_kp"]
        assert pid._pitch_pid.kp == g["pitch_kp"]
        assert pid._velocity_pid.kp == g["vel_kp"]
        assert pid.elevator_sign == +1
        assert pid.throttle_min == g["throttle_min"]

    def test_custom_gains_override_defaults(self):
        pid = F22MavEnergyPIDController(
            dt=1.0 / 60, elevator_sign=-1,
            roll_kp=0.09, pitch_kp=0.9, vel_kp=0.05,
            throttle_min=0.65, low_speed_throttle_floor=0.90,
        )
        assert pid._roll_pid.kp == 0.09
        assert pid._pitch_pid.kp == 0.9
        assert pid._velocity_pid.kp == 0.05
        assert pid.elevator_sign == -1
        assert pid.throttle_min == 0.65
        assert pid.low_speed_throttle_floor == 0.90

    def test_energy_guard_activates_at_low_speed(self):
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        rpy = np.array([0.0, 0.1, 0.0])
        vel = np.array([100.0, 0.0, 5.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        # Action that requests aggressive pitch
        target_pitch = np.deg2rad(30.0)
        target_heading = 0.0
        target_velocity = 300.0

        pid.compute_control(rpy, current_speed,
                            target_pitch, target_heading, target_velocity,
                            ned_velocity=vel_ned)
        # At 100 m/s (<150 critical), energy guard should activate
        assert pid.last_energy_guard_active is True
        assert pid.last_energy_guard_level == "critical"

    def test_energy_guard_inactive_at_high_speed(self):
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        rpy = np.array([0.0, 0.1, 0.0])
        vel = np.array([250.0, 0.0, 5.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        target_pitch = np.deg2rad(10.0)
        target_heading = 0.0
        target_velocity = 300.0

        pid.compute_control(rpy, current_speed,
                            target_pitch, target_heading, target_velocity,
                            ned_velocity=vel_ned)
        # At 250 m/s (>180), energy guard should be inactive
        assert pid.last_energy_guard_active is False
        assert pid.last_energy_guard_level == ""

    def test_level_action_at_medium_speed_energy_guard_low(self):
        """At 160 m/s (between critical and low), energy guard 'low' should activate."""
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        rpy = np.array([0.0, 0.1, 0.0])
        vel = np.array([160.0, 0.0, 5.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        target_pitch = np.deg2rad(20.0)  # aggressive
        target_heading = 0.0
        target_velocity = 300.0

        pid.compute_control(rpy, current_speed,
                            target_pitch, target_heading, target_velocity,
                            ned_velocity=vel_ned)
        assert pid.last_energy_guard_active is True
        assert pid.last_energy_guard_level == "low"
        assert pid.last_pitch_clamped is True

    def test_outputs_are_finite(self):
        """Smoke test: compute_control returns finite values."""
        pid = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        rpy = np.array([0.0, 0.1, 0.0])
        vel = np.array([200.0, 0.0, 5.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        ail, elev, rud, thr = pid.compute_control(
            rpy, current_speed, 0.0, 0.0, 250.0, ned_velocity=vel_ned,
        )
        assert np.isfinite(ail) and -1.0 <= ail <= 1.0
        assert np.isfinite(elev) and -1.0 <= elev <= 1.0
        assert rud == 0.0
        assert np.isfinite(thr) and 0.0 <= thr <= 1.0

    def test_throttle_floor_at_low_speed(self):
        """At very low speed, throttle should be boosted to floor."""
        pid = F22MavEnergyPIDController(
            dt=1.0 / 60, elevator_sign=+1,
            low_speed_throttle_floor=0.85,
        )
        rpy = np.array([0.0, 0.1, 0.0])
        vel = np.array([90.0, 0.0, 2.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        _, _, _, thr = pid.compute_control(
            rpy, current_speed, 0.0, 0.0, 150.0, ned_velocity=vel_ned,
        )
        # At 90 m/s (<180), throttle should be >= floor
        assert thr >= 0.85
        assert pid.last_throttle_boosted is True

    def test_bijection_f22_and_f16_are_different_classes(self):
        """F22 and F16 controllers should be different types."""
        f16 = PIDController(dt=1.0 / 60)
        f22 = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        assert type(f16) is not type(f22)
        assert isinstance(f22, PIDController)
        assert not isinstance(f16, F22MavEnergyPIDController)

    def test_elevator_sign_used_in_compute(self):
        """Elevator sign should affect the sign of the elevator output."""
        # Use identical inputs with sign=+1 and sign=-1
        rpy = np.array([0.1, 0.2, 0.3])
        vel = np.array([200.0, 10.0, -5.0], dtype=np.float64)
        current_speed = float(np.linalg.norm(vel))
        vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)

        pid_pos = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=+1)
        pid_neg = F22MavEnergyPIDController(dt=1.0 / 60, elevator_sign=-1)

        _, elev_pos, _, _ = pid_pos.compute_control(
            rpy.copy(), current_speed, 0.3, 0.0, 250.0, ned_velocity=vel_ned.copy(),
        )
        _, elev_neg, _, _ = pid_neg.compute_control(
            rpy.copy(), current_speed, 0.3, 0.0, 250.0, ned_velocity=vel_ned.copy(),
        )

        # The signs should be opposite (or both near zero if PID output is tiny)
        assert elev_pos * elev_neg <= 1e-6 or abs(elev_pos) < 1e-3


class TestF22ProfileEnvIntegration:
    """Verify env correctly routes PID profiles."""

    def test_env_creates_f22_controller_for_mav(self):
        from uav_env.make_env import make_env

        config_path = (
            "uav_env/JSBSim/configs/"
            "hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"
        )
        env = make_env(config_path)
        assert env.pid_profile_by_role == {
            "mav": "f22_mav_energy_pid",
            "attack_uav": "f16_default",
            "interceptor_uav": "f16_default",
        }

        # After reset, check PID types
        obs, info = env.reset()
        # red_0 = MAV → F22 controller (fallback elevator_sign=+1)
        mav_pid = env.pid_controllers["red_0"]
        assert isinstance(mav_pid, F22MavEnergyPIDController)
        assert mav_pid.elevator_sign is not None
        # red_1 = attack_uav → F16 controller
        uav_pid = env.pid_controllers["red_1"]
        assert isinstance(uav_pid, PIDController)
        assert not isinstance(uav_pid, F22MavEnergyPIDController)
        env.close()


def test_direct_fcs_applies_aileron_and_elevator_trim_without_changing_mainline():
    from uav_env.JSBSim.env import UavCombatEnv

    class FakeSim:
        is_alive = True

        def __init__(self):
            self.values = {}

        def set_property_value(self, name, value):
            self.values[name] = value

    env = UavCombatEnv.__new__(UavCombatEnv)
    sim = FakeSim()
    env._get_sim = lambda _aid: sim
    env._control_mode_for = lambda _aid: "direct_fcs_3d"
    env._direct_fcs_trim_for = lambda _aid: {"elevator": 0.1, "aileron": -0.2}
    env._apply_pid_controls({"red_0": (0.2, 0.3, 0.6)})

    assert sim.values["fcs/elevator-cmd-norm"] == pytest.approx(0.3)
    assert sim.values["fcs/aileron-cmd-norm"] == pytest.approx(0.1)

    def test_f22_pid_mainline_keeps_three_dimensional_actions(self):
        from uav_env.make_env import make_env

        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"
        )
        env.reset()
        assert env.action_space["red_0"].shape == (3,)
        assert "direct_fcs" not in env.pid_profile_by_role["mav"]
        env.close()

    def test_env_f16_default_without_profile_config(self):
        """Without pid_profile_by_role, all agents use F16 PIDController."""
        from uav_env.make_env import make_env

        # Use a config without pid_profile_by_role
        config_path = (
            "uav_env/JSBSim/configs/"
            "hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
        )
        env = make_env(config_path)
        obs, info = env.reset()
        for aid in env.agent_ids:
            pid = env.pid_controllers[aid]
            assert isinstance(pid, PIDController)
            assert not isinstance(pid, F22MavEnergyPIDController)
        env.close()

    def test_elevator_sign_applied_from_config(self):
        """Verify that pid_profile_config with elevator_sign=-1 flows through."""
        from uav_env.make_env import make_env

        config_path = (
            "uav_env/JSBSim/configs/"
            "hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"
        )
        env = make_env(config_path)
        # Set profile config BEFORE reset (tests override of fallback +1)
        env.pid_profile_config = {
            "f22_mav_energy_pid": {"elevator_sign": -1}
        }
        obs, info = env.reset()
        mav_pid = env.pid_controllers["red_0"]
        assert isinstance(mav_pid, F22MavEnergyPIDController)
        assert mav_pid.elevator_sign == -1
        env.close()
