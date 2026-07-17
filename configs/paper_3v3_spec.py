"""Single formal contract for the paper-derived deterministic 3V3 environment."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    PAPER_EQUATION,
    PAPER_EXPLICIT,
    AwacsConfig,
    ElectroOpticalConfig,
    MissileConfig,
    MissileWarningConfig,
    PIDConfig,
    PaperEnvironmentConfig,
    RCSConfig,
    RwsConfig,
    ScenarioConfig,
    sv,
)
from my_uav_env.pid_controller import (
    PAPER_PID_DERIVATIVE_SEMANTICS,
    PAPER_PID_ERROR_DEFINITION,
)


PAPER_UNSPECIFIED_ENGINEERING = "paper_unspecified_engineering"
PAPER_EQUATION_OPERATIONAL = "paper_equation_operational"
INTENTIONAL_3V3_DEVIATION = "intentional_3v3_deviation"

PAPER_ENVIRONMENT_PROFILE = "paper_3v3_v1"
PAPER_BLUE_POLICY_PROFILE = "simple_dynamic_pursuit_with_mws"
PAPER_PID_PROFILE = "paper_3v3_pid_v1"
PAPER_REWARD_MODE = "paper_joint"
PAPER_MISSILE_GUIDANCE_MODE = "paper_3v3_eq9_11_v1"
PAPER_CHECKPOINT_SCHEMA = "vanilla_mappo_paper_3v3_v1"

MISSILE_LAUNCH_SPEED_MPS = 800.0
MISSILE_HIT_RADIUS_M = 100.0
MISSILE_OVERSHOOT_WINDOW_S = 0.5
MISSILE_OVERSHOOT_DISTANCE_HYSTERESIS_M = 50.0
MISSILE_POSITIVE_CLOSING_THRESHOLD_MPS = 1.0
PID_THROTTLE_BASE = 0.8
PERSISTENT_EXTREME_G = 30.0
PERSISTENT_EXTREME_FRAMES = 3
CATASTROPHIC_G = 100.0
AIRCRAFT_ENVELOPE_FRAMES = 3
COARSE_HORIZONTAL_GRID_M = 500.0
COARSE_ALTITUDE_GRID_M = 250.0


PAPER_SCENARIO = ScenarioConfig(
    arena_half_width_m=sv(50_000.0, PAPER_EXPLICIT),
    arena_altitude_min_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    arena_altitude_max_m=sv(10_000.0, PAPER_EXPLICIT),
    reward_boundary_half_width_m=sv(40_000.0, PAPER_EQUATION),
    initial_head_on_range_m=sv(10_000.0, PAPER_EXPLICIT),
    initial_altitude_m=sv(6_000.0, PAPER_UNSPECIFIED_ENGINEERING),
    initial_speed_mps=sv(300.0, PAPER_UNSPECIFIED_ENGINEERING),
    formation_spacing_m=sv(1_000.0, PAPER_UNSPECIFIED_ENGINEERING),
    reference_longitude_deg=sv(120.0, PAPER_UNSPECIFIED_ENGINEERING),
    reference_latitude_deg=sv(60.0, PAPER_UNSPECIFIED_ENGINEERING),
    aircraft_model=sv("f16", PAPER_UNSPECIFIED_ENGINEERING),
    missiles_per_aircraft=sv(2, PAPER_UNSPECIFIED_ENGINEERING),
    episode_steps=sv(1_400, PAPER_EXPLICIT),
)

PAPER_RCS = RCSConfig(
    azimuth_grid_deg=sv(
        (-180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 180.0),
        PAPER_UNSPECIFIED_ENGINEERING),
    elevation_grid_deg=sv(
        (-90.0, 0.0, 90.0), PAPER_UNSPECIFIED_ENGINEERING),
    table_m2=sv((
        (0.3, 1.2, 0.15, 0.1, 0.15, 1.2, 0.3),
        (0.2, 2.0, 0.1, 0.1, 0.1, 2.0, 0.2),
        (0.3, 1.2, 0.15, 0.1, 0.15, 1.2, 0.3),
    ), PAPER_UNSPECIFIED_ENGINEERING,
       "Minimal symmetric engineering table; the paper does not publish it."),
    range_constant=sv(40_000.0, PAPER_UNSPECIFIED_ENGINEERING),
)

PAPER_PID = PIDConfig(
    roll_gains=sv((0.15, 0.5, 0.05), PAPER_UNSPECIFIED_ENGINEERING),
    pitch_gains=sv((2.5, 0.5, 0.1), PAPER_UNSPECIFIED_ENGINEERING),
    speed_gains=sv((0.04, 0.01, 0.003), PAPER_UNSPECIFIED_ENGINEERING),
    integral_error_limits=sv(
        (1.5707963268, 1.5707963268, 306.0),
        PAPER_UNSPECIFIED_ENGINEERING),
    throttle_base=sv(PID_THROTTLE_BASE, PAPER_UNSPECIFIED_ENGINEERING),
    anti_windup_mode=sv("back_calculation", PAPER_UNSPECIFIED_ENGINEERING),
    actuator_signs=sv((1.0, -1.0, 0.0, 1.0), PAPER_UNSPECIFIED_ENGINEERING),
)

PAPER_MISSILE = replace(
    MissileConfig(),
    model=sv("paper_3v3_constant_speed_point_mass", PAPER_UNSPECIFIED_ENGINEERING),
    maximum_flight_time_s=sv(60.0, PAPER_UNSPECIFIED_ENGINEERING),
    thrust_time_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    specific_impulse_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    length_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    diameter_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    drag_coefficient=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    initial_mass_kg=sv(1.0, PAPER_UNSPECIFIED_ENGINEERING),
    mass_flow_kg_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    navigation_constant=sv(3.0, PAPER_EQUATION_OPERATIONAL),
    maximum_overload_g=sv(30.0, PAPER_UNSPECIFIED_ENGINEERING),
    hit_radius_m=sv(MISSILE_HIT_RADIUS_M, PAPER_UNSPECIFIED_ENGINEERING),
    minimum_speed_mps=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    arming_time_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    density_model=sv("disabled", PAPER_UNSPECIFIED_ENGINEERING),
    guidance_mode=sv(PAPER_MISSILE_GUIDANCE_MODE, PAPER_EQUATION_OPERATIONAL),
)

PAPER_ENVIRONMENT_CONFIG = PaperEnvironmentConfig(
    version=sv(PAPER_ENVIRONMENT_PROFILE, INTENTIONAL_3V3_DEVIATION),
    scenario=PAPER_SCENARIO,
    aircraft=DEFAULT_PAPER_ENVIRONMENT_CONFIG.aircraft,
    pid=PAPER_PID,
    radar=DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar,
    rcs=PAPER_RCS,
    awacs=AwacsConfig(
        update_period_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
        track_hold_s=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
        horizontal_error_std_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
        vertical_error_std_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING)),
    rws=RwsConfig(
        bearing_error_std_rad=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING)),
    electro_optical=ElectroOpticalConfig(
        maximum_range_m=sv(10_000.0, PAPER_EXPLICIT),
        half_angle_rad=sv(0.7853981633974483, PAPER_UNSPECIFIED_ENGINEERING),
        minimum_launch_range_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING)),
    missile_warning=MissileWarningConfig(
        detection_mode=sv(
            "approaching_targeting_live_missiles_ttc_v1",
            PAPER_UNSPECIFIED_ENGINEERING)),
    missile=PAPER_MISSILE,
    fire_control=DEFAULT_PAPER_ENVIRONMENT_CONFIG.fire_control,
    reward=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.reward,
        altitude_interpretation=sv(
            "paper_unspecified_engineering_mean_over_alive_enemies",
            PAPER_UNSPECIFIED_ENGINEERING)),
)

PAPER_PROFILE_METADATA = {
    "environment_profile": sv(PAPER_ENVIRONMENT_PROFILE, INTENTIONAL_3V3_DEVIATION),
    "paper_training_scale": sv("6V6", PAPER_EXPLICIT),
    "current_scale": sv("3V3", INTENTIONAL_3V3_DEVIATION),
    "aircraft_dynamics": sv("JSBSim_6DoF", PAPER_EXPLICIT),
    "physics_frequency_hz": sv(60, PAPER_UNSPECIFIED_ENGINEERING),
    "decision_frequency_hz": sv(5, PAPER_UNSPECIFIED_ENGINEERING),
    "physics_frames_per_action": sv(12, PAPER_UNSPECIFIED_ENGINEERING),
    "maximum_episode_steps": sv(1400, PAPER_EXPLICIT),
    "maximum_aircraft_speed_mps": sv(600.0, PAPER_EXPLICIT),
    "maximum_aircraft_load_g": sv(9.0, PAPER_EXPLICIT),
    "radar_azimuth_rad": DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar.azimuth_max_rad,
    "radar_elevation_min_rad": DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar.elevation_min_rad,
    "radar_elevation_max_rad": DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar.elevation_max_rad,
    "eo_maximum_range_m": PAPER_ENVIRONMENT_CONFIG.electro_optical.maximum_range_m,
    "eo_half_angle_rad": PAPER_ENVIRONMENT_CONFIG.electro_optical.half_angle_rad,
    "fire_control_lock_time_s": PAPER_ENVIRONMENT_CONFIG.fire_control.lock_time_s,
    "fire_control_launch_interval_s": (
        PAPER_ENVIRONMENT_CONFIG.fire_control.launch_interval_s),
    "fire_control_rear_hemisphere_ta_rad": (
        PAPER_ENVIRONMENT_CONFIG.fire_control.rear_hemisphere_ta_rad),
    "initial_condition_profile": sv(
        "deterministic_head_on_3v3_v1", PAPER_UNSPECIFIED_ENGINEERING),
    "observation_schema": sv("paper_table1_table2_entity_mask_v1", PAPER_EXPLICIT),
    "blue_policy_profile": sv(PAPER_BLUE_POLICY_PROFILE, PAPER_UNSPECIFIED_ENGINEERING),
    "red_mws_mode": sv("scripted_ttc_evasion_v1", PAPER_UNSPECIFIED_ENGINEERING),
    "blue_mws_mode": sv("scripted_ttc_evasion_v1", PAPER_UNSPECIFIED_ENGINEERING),
    "pid_profile": sv(PAPER_PID_PROFILE, PAPER_EQUATION_OPERATIONAL),
    "pid_error_definition": sv(PAPER_PID_ERROR_DEFINITION, PAPER_EQUATION_OPERATIONAL),
    "derivative_semantics": sv(
        PAPER_PID_DERIVATIVE_SEMANTICS, PAPER_UNSPECIFIED_ENGINEERING),
    "missile_profile": sv(PAPER_MISSILE_GUIDANCE_MODE, PAPER_EQUATION_OPERATIONAL),
    "missile_launch_speed_mps": sv(
        MISSILE_LAUNCH_SPEED_MPS, PAPER_UNSPECIFIED_ENGINEERING),
    "missile_hit_radius_m": PAPER_MISSILE.hit_radius_m,
    "missile_maximum_command_g": PAPER_MISSILE.maximum_overload_g,
    "missile_lifetime_s": PAPER_MISSILE.maximum_flight_time_s,
    "reward_mode": sv(PAPER_REWARD_MODE, PAPER_EQUATION_OPERATIONAL),
    "altitude_pair_aggregation": sv(
        "paper_unspecified_engineering_mean_over_alive_enemies",
        PAPER_UNSPECIFIED_ENGINEERING),
    "actor_input_dim": sv(60, PAPER_EXPLICIT),
    "critic_input_dim": sv(30, PAPER_EXPLICIT),
    "coarse_horizontal_grid_m": sv(
        COARSE_HORIZONTAL_GRID_M, PAPER_UNSPECIFIED_ENGINEERING),
    "coarse_altitude_grid_m": sv(
        COARSE_ALTITUDE_GRID_M, PAPER_UNSPECIFIED_ENGINEERING),
    "aircraft_envelope_frames": sv(
        AIRCRAFT_ENVELOPE_FRAMES, PAPER_UNSPECIFIED_ENGINEERING),
    "missiles_per_aircraft": PAPER_SCENARIO.missiles_per_aircraft,
    "same_frame_hit_guard": sv(
        "launch_frame_no_contact_sampling_v1", PAPER_UNSPECIFIED_ENGINEERING),
}


def paper_environment_snapshot(*, seed: int | None) -> dict:
    """Return the serializable formal contract and stable fingerprint."""
    payload = {key: asdict(value) for key, value in PAPER_PROFILE_METADATA.items()}
    payload.update({
        "num_red": 3,
        "num_blue": 3,
        "sim_freq": 60,
        "agent_interaction_steps": 12,
        "decision_frequency_hz": 5.0,
        "max_episode_length": 1400,
        "environment_config": asdict(PAPER_ENVIRONMENT_CONFIG),
        "seed": seed,
    })
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("seed", None)
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["environment_config_fingerprint"] = hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()
    return payload
