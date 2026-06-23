import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.JSBSim.pid_controller import F22MavEnergyPIDController
from uav_env.JSBSim.simulator import AircraftSimulator, SuppressOutput


ROOT = Path(__file__).resolve().parents[1]
INIT_STATE = {
    "latitude_deg": 59.98,
    "longitude_deg": 120.02,
    "altitude_ft": 6000.0 / 0.3048,
    "heading_deg": 0.0,
    "speed_fps": 250.0 / 0.3048,
}


def _make_f22():
    with SuppressOutput():
        return AircraftSimulator(
            uid="red_0",
            color="Red",
            model="f22",
            sim_freq=60,
            num_missiles=0,
            init_state=dict(INIT_STATE),
            suppress_jsbsim_output=True,
        )


def _prop(sim, name):
    return float(sim.get_property_value(name))


def _speed(sim):
    return float(np.linalg.norm(sim.get_velocity()))


def _run_direct_elevator(command, frames=90):
    sim = _make_f22()
    for _ in range(frames):
        sim.set_property_value("fcs/elevator-cmd-norm", float(command))
        sim.set_property_value("fcs/aileron-cmd-norm", 0.0)
        sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
        sim.set_property_value("fcs/throttle-cmd-norm", 0.85)
        sim.run()
    return {
        "elevator_pos_rad": _prop(sim, "fcs/elevator-pos-rad"),
        "pitch_deg": math.degrees(sim.get_rpy()[1]),
        "alpha_deg": math.degrees(_prop(sim, "aero/alpha-rad")),
        "speed_mps": _speed(sim),
        "altitude_m": sim.get_geodetic()[2],
        "alive": sim.is_alive,
    }


def _run_pid_action(action, frames=200):
    sim = _make_f22()
    pid = F22MavEnergyPIDController(
        1.0 / 60.0,
        elevator_sign=-1,
        pitch_kp=1.0,
        pitch_ki=0.0,
        pitch_kd=1.0,
        roll_kp=0.06,
        roll_ki=0.08,
        roll_kd=0.03,
        vel_kp=0.04,
        vel_ki=0.006,
        vel_kd=0.002,
        throttle_min=0.72,
        throttle_max=1.0,
        low_speed_throttle_floor=0.96,
    )
    target_pitch = float(action[0]) * math.radians(90.0)
    target_heading = float(action[1]) * math.pi
    target_velocity = 102.0 + (float(action[2]) + 1.0) / 2.0 * (408.0 - 102.0)
    rows = []
    for _ in range(frames):
        rpy = sim.get_rpy()
        vel = sim.get_velocity()
        speed = float(np.linalg.norm(vel))
        ned_velocity = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
        aileron, elevator, rudder, throttle = pid.compute_control(
            rpy,
            speed,
            target_pitch,
            target_heading,
            target_velocity,
            ned_velocity=ned_velocity,
        )
        sim.set_property_value("fcs/aileron-cmd-norm", float(np.clip(aileron, -1.0, 1.0)))
        sim.set_property_value("fcs/elevator-cmd-norm", float(np.clip(elevator, -1.0, 1.0)))
        sim.set_property_value("fcs/rudder-cmd-norm", float(np.clip(rudder, -1.0, 1.0)))
        sim.set_property_value("fcs/throttle-cmd-norm", float(np.clip(throttle, 0.0, 1.0)))
        sim.run()
        rows.append(
            {
                "pitch_deg": math.degrees(sim.get_rpy()[1]),
                "roll_deg": math.degrees(sim.get_rpy()[0]),
                "alpha_deg": math.degrees(_prop(sim, "aero/alpha-rad")),
                "speed_mps": _speed(sim),
                "altitude_m": sim.get_geodetic()[2],
                "alive": sim.is_alive,
            }
        )
    return rows


def test_f22_model_name_loads_f22_xml():
    aircraft_dir = ROOT / "uav_env/JSBSim/data/aircraft/f22"
    assert (aircraft_dir / "f22.xml").exists()
    assert (aircraft_dir / "yf22.xml").exists()
    sim = _make_f22()
    assert sim.model == "f22"
    assert sim.is_alive


def test_f22_direct_fcs_elevator_surface_responds_to_command():
    down = _run_direct_elevator(-1.0)
    neutral = _run_direct_elevator(0.0)
    up = _run_direct_elevator(1.0)

    assert down["alive"] and neutral["alive"] and up["alive"]
    assert down["elevator_pos_rad"] < neutral["elevator_pos_rad"] < up["elevator_pos_rad"]
    assert abs(up["elevator_pos_rad"] - down["elevator_pos_rad"]) > 0.15
    assert all(np.isfinite([down["pitch_deg"], neutral["pitch_deg"], up["pitch_deg"]]))


def test_f22_single_throttle_command_drives_both_engines():
    sim = _make_f22()
    for _ in range(90):
        sim.set_property_value("fcs/elevator-cmd-norm", 0.0)
        sim.set_property_value("fcs/aileron-cmd-norm", 0.0)
        sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
        sim.set_property_value("fcs/throttle-cmd-norm", 0.85)
        sim.run()

    throttle_0 = _prop(sim, "fcs/throttle-pos-norm")
    throttle_1 = _prop(sim, "fcs/throttle-pos-norm[1]")
    assert throttle_0 > 0.40
    assert throttle_1 > 0.40
    assert throttle_1 == pytest.approx(throttle_0, abs=1e-3)


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.0, 0.0, 0.3], dtype=np.float32),
        np.array([0.05, 0.0, 0.4], dtype=np.float32),
    ],
)
def test_f22_pid_200_step_stability(action):
    rows = _run_pid_action(action, frames=200)
    assert all(r["alive"] for r in rows)
    assert all(
        np.isfinite([r["pitch_deg"], r["roll_deg"], r["speed_mps"], r["altitude_m"]]).all()
        for r in rows
    )
    assert max(abs(r["pitch_deg"]) for r in rows) < 55.0
    assert max(abs(r["roll_deg"]) for r in rows) < 80.0
    assert min(r["speed_mps"] for r in rows) > 120.0
    assert min(r["altitude_m"] for r in rows) > 5000.0


@pytest.mark.parametrize(
    "action,max_pitch,min_speed,min_altitude",
    [
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 15.0, 220.0, 5900.0),
        (np.array([0.05, 0.0, 0.4], dtype=np.float32), 20.0, 220.0, 5900.0),
    ],
)
def test_f22_env_200_step_stability(action, max_pitch, min_speed, min_altitude):
    from uav_env.make_env import make_env

    env = make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"
    )
    env.reset()
    rows = []
    try:
        for _ in range(200):
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            actions["red_0"] = action
            env.step(actions)
            sim = env.red_planes["red_0"]
            rpy = sim.get_rpy()
            rows.append(
                {
                    "alive": sim.is_alive,
                    "pitch_deg": math.degrees(rpy[1]),
                    "roll_deg": math.degrees(rpy[0]),
                    "speed_mps": _speed(sim),
                    "altitude_m": sim.get_geodetic()[2],
                }
            )
            if not sim.is_alive:
                break
    finally:
        env.close()

    assert len(rows) == 200
    assert all(r["alive"] for r in rows)
    assert max(abs(r["pitch_deg"]) for r in rows) < max_pitch
    assert max(abs(r["roll_deg"]) for r in rows) < 10.0
    assert min(r["speed_mps"] for r in rows) > min_speed
    assert min(r["altitude_m"] for r in rows) > min_altitude


@pytest.mark.parametrize(
    "config_name,red_count,blue_count",
    [
        ("hetero_mav_shared_geo_3v2_f22_paper_role_reward_v1.yaml", 3, 2),
        ("hetero_mav_shared_geo_5v4_f22_paper_role_reward_v1.yaml", 5, 4),
        ("hetero_mav_shared_geo_7v6_f22_paper_role_reward_v1.yaml", 7, 6),
    ],
)
def test_f22_paper_role_reward_configs_exist_and_keep_contract(config_name, red_count, blue_count):
    path = ROOT / "uav_env/JSBSim/configs" / config_name
    assert path.exists()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert cfg["observation_mode"] == "mav_shared_geo"
    assert cfg["hetero_reward_mode"] == "paper_role_reward_v1"
    assert cfg["mav_observation_range_m"] == 80000
    assert cfg["max_num_red"] == red_count
    assert cfg["max_num_blue"] == blue_count
    assert cfg["aircraft_type_params"]["mav"]["aircraft_model"] == "f22"
    assert cfg["aircraft_type_params"]["mav"]["role"] == "mav"
    assert cfg["aircraft_type_params"]["mav"]["num_missiles"] == 0
    assert cfg["aircraft_type_params"]["attack_uav"]["aircraft_model"] == "f16"
    assert cfg["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2
