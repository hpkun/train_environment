"""Frozen public-paper reproduction contract for the independent 6V6 profile."""
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
    PaperEnvironmentConfig,
    RwsConfig,
    ScenarioConfig,
    sv,
)


PAPER_UNDISCLOSED_ENGINEERING = "paper_undisclosed_engineering"
PAPER_EQUATION_OPERATIONAL = "paper_equation_operational"
PAPER_ENVIRONMENT_PROFILE_6V6 = "paper_6v6_reproduction_v1"
PAPER_BLUE_POLICY_PROFILE_6V6 = "paper_undisclosed_minimal_rule_v1"
PAPER_MWS_PROFILE_6V6 = "paper_undisclosed_symmetric_mws_v1"
PAPER_PID_PROFILE_6V6 = "paper_eq12_eq14_pid_6v6_v1"
PAPER_MISSILE_GUIDANCE_MODE_6V6 = "paper_eq7_eq11_point_mass_v1"
PAPER_REWARD_VERSION_6V6 = "paper_literal_eq15_eq23_sum_6v6_v1"
PAPER_REWARD_MODE_6V6 = "paper_joint"
MISSILE_LAUNCH_SPEED_MPS_6V6 = 800.0
PID_THROTTLE_BASE_6V6 = 0.8


PAPER_SCENARIO_6V6 = ScenarioConfig(
    arena_half_width_m=sv(50_000.0, PAPER_EXPLICIT),
    arena_altitude_min_m=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING),
    arena_altitude_max_m=sv(10_000.0, PAPER_EXPLICIT),
    reward_boundary_half_width_m=sv(40_000.0, PAPER_EQUATION),
    initial_head_on_range_m=sv(10_000.0, PAPER_EXPLICIT),
    initial_altitude_m=sv(6_000.0, PAPER_UNDISCLOSED_ENGINEERING),
    initial_speed_mps=sv(300.0, PAPER_UNDISCLOSED_ENGINEERING),
    formation_spacing_m=sv(1_000.0, PAPER_UNDISCLOSED_ENGINEERING),
    reference_longitude_deg=sv(120.0, PAPER_UNDISCLOSED_ENGINEERING),
    reference_latitude_deg=sv(60.0, PAPER_UNDISCLOSED_ENGINEERING),
    aircraft_model=sv("f16", PAPER_UNDISCLOSED_ENGINEERING),
    missiles_per_aircraft=sv(2, PAPER_UNDISCLOSED_ENGINEERING),
    episode_steps=sv(1_400, PAPER_EXPLICIT),
)

# Eq.7-Eq.11 structure is public. The aerodynamic/propulsion numbers are not.
PAPER_MISSILE_6V6 = replace(
    MissileConfig(),
    model=sv("paper_eq7_eq11_point_mass", PAPER_EQUATION_OPERATIONAL),
    maximum_flight_time_s=sv(60.0, PAPER_UNDISCLOSED_ENGINEERING),
    thrust_time_s=sv(3.0, PAPER_UNDISCLOSED_ENGINEERING),
    specific_impulse_s=sv(240.0, PAPER_UNDISCLOSED_ENGINEERING),
    length_m=sv(2.87, PAPER_UNDISCLOSED_ENGINEERING),
    diameter_m=sv(0.127, PAPER_UNDISCLOSED_ENGINEERING),
    drag_coefficient=sv(0.22, PAPER_UNDISCLOSED_ENGINEERING),
    initial_mass_kg=sv(84.0, PAPER_UNDISCLOSED_ENGINEERING),
    mass_flow_kg_s=sv(6.0, PAPER_UNDISCLOSED_ENGINEERING),
    navigation_constant=sv(3.0, PAPER_UNDISCLOSED_ENGINEERING),
    maximum_overload_g=sv(30.0, PAPER_EQUATION),
    hit_radius_m=sv(300.0, PAPER_UNDISCLOSED_ENGINEERING),
    minimum_speed_mps=sv(150.0, PAPER_UNDISCLOSED_ENGINEERING),
    arming_time_s=sv(0.15, PAPER_UNDISCLOSED_ENGINEERING),
    density_model=sv(
        "rho0_exp_altitude_over_9300", PAPER_UNDISCLOSED_ENGINEERING),
    guidance_mode=sv(PAPER_MISSILE_GUIDANCE_MODE_6V6, PAPER_EQUATION_OPERATIONAL),
)

PAPER_ENVIRONMENT_CONFIG_6V6 = PaperEnvironmentConfig(
    version=sv(PAPER_ENVIRONMENT_PROFILE_6V6, PAPER_EQUATION_OPERATIONAL),
    scenario=PAPER_SCENARIO_6V6,
    aircraft=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.aircraft,
        overspeed_throttle_limit=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.aircraft.overspeed_throttle_limit.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        engineering_overload_termination_s=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.aircraft.engineering_overload_termination_s.value,
            PAPER_UNDISCLOSED_ENGINEERING)),
    pid=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid,
        roll_gains=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.roll_gains.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        pitch_gains=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.pitch_gains.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        speed_gains=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.speed_gains.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        integral_error_limits=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.integral_error_limits.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        throttle_base=sv(PID_THROTTLE_BASE_6V6,
                         PAPER_UNDISCLOSED_ENGINEERING),
        anti_windup_mode=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.anti_windup_mode.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        actuator_signs=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.pid.actuator_signs.value,
            PAPER_UNDISCLOSED_ENGINEERING)),
    radar=DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar,
    rcs=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.rcs,
        azimuth_grid_deg=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.rcs.azimuth_grid_deg.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        elevation_grid_deg=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.rcs.elevation_grid_deg.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        table_m2=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.rcs.table_m2.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        range_constant=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.rcs.range_constant.value,
            PAPER_UNDISCLOSED_ENGINEERING)),
    awacs=AwacsConfig(
        update_period_s=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING),
        track_hold_s=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING),
        horizontal_error_std_m=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING),
        vertical_error_std_m=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING)),
    rws=RwsConfig(
        bearing_error_std_rad=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING)),
    electro_optical=ElectroOpticalConfig(
        maximum_range_m=sv(10_000.0, PAPER_EXPLICIT),
        half_angle_rad=sv(0.7853981633974483, PAPER_UNDISCLOSED_ENGINEERING),
        minimum_launch_range_m=sv(0.0, PAPER_UNDISCLOSED_ENGINEERING)),
    missile_warning=MissileWarningConfig(
        detection_mode=sv(PAPER_MWS_PROFILE_6V6, PAPER_UNDISCLOSED_ENGINEERING)),
    missile=PAPER_MISSILE_6V6,
    fire_control=DEFAULT_PAPER_ENVIRONMENT_CONFIG.fire_control,
    reward=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.reward,
        mach_reference_mps=sv(
            DEFAULT_PAPER_ENVIRONMENT_CONFIG.reward.mach_reference_mps.value,
            PAPER_UNDISCLOSED_ENGINEERING),
        altitude_interpretation=sv(
            "paper_unresolved_sum_over_alive_enemies",
            PAPER_UNDISCLOSED_ENGINEERING)),
)

PAPER_PROFILE_METADATA_6V6 = {
    "environment_profile": sv(PAPER_ENVIRONMENT_PROFILE_6V6, PAPER_EQUATION_OPERATIONAL),
    "scientific_claim": sv(
        "public_paper_information_reproduction_not_private_code_exact",
        PAPER_UNDISCLOSED_ENGINEERING),
    "paper_training_scale": sv("6V6", PAPER_EXPLICIT),
    "entity_feature_dimension": sv(10, PAPER_EXPLICIT),
    "entity_count": sv(12, PAPER_EQUATION_OPERATIONAL),
    "actor_input_dim": sv(120, PAPER_EQUATION_OPERATIONAL),
    "critic_input_dim": sv(60, PAPER_EQUATION_OPERATIONAL),
    "observation_profile": sv("paper_table1_table2_6v6_v1", PAPER_EQUATION_OPERATIONAL),
    "global_state_profile": sv("paper_red_self_state_6x10_v1", PAPER_EQUATION_OPERATIONAL),
    "observation_schema": sv("paper_table1_table2_with_masks_v1", PAPER_EQUATION_OPERATIONAL),
    "blue_policy_profile": sv(PAPER_BLUE_POLICY_PROFILE_6V6, PAPER_UNDISCLOSED_ENGINEERING),
    "mws_profile": sv(PAPER_MWS_PROFILE_6V6, PAPER_UNDISCLOSED_ENGINEERING),
    "red_mws_mode": sv(PAPER_MWS_PROFILE_6V6, PAPER_UNDISCLOSED_ENGINEERING),
    "blue_mws_mode": sv(PAPER_MWS_PROFILE_6V6, PAPER_UNDISCLOSED_ENGINEERING),
    "sensor_support_profile": sv(
        "deterministic_radar_or_quantized_position_6v6_v1",
        PAPER_UNDISCLOSED_ENGINEERING),
    "pid_profile": sv(PAPER_PID_PROFILE_6V6, PAPER_EQUATION_OPERATIONAL),
    "pid_error_definition": sv(
        "eq12_body_transform_atan2_dby_dbx_atan2_dbz_dbx",
        PAPER_EQUATION_OPERATIONAL),
    "derivative_semantics": sv(
        "first_sample_only_error_derivative_v2",
        PAPER_UNDISCLOSED_ENGINEERING,
        "The paper does not specify error versus measurement differentiation."),
    "missile_profile": sv(PAPER_MISSILE_GUIDANCE_MODE_6V6, PAPER_EQUATION_OPERATIONAL),
    "missile_launch_speed_mps": sv(MISSILE_LAUNCH_SPEED_MPS_6V6, PAPER_UNDISCLOSED_ENGINEERING),
    "missile_hit_radius_m": PAPER_MISSILE_6V6.hit_radius_m,
    "missile_model_scope": sv("eq7_eq8_point_mass_eq9_eq11_guidance", PAPER_EQUATION_OPERATIONAL),
    "MissileEquationStructureStatus": sv("EQ7_EQ11_IMPLEMENTED", PAPER_EQUATION_OPERATIONAL),
    "MissileNumericalParameterReproductionStatus": sv(
        "PARTIAL_UNDISCLOSED", PAPER_UNDISCLOSED_ENGINEERING),
    "missile_numerical_parameter_units": sv({
        "maximum_flight_time_s": "s", "thrust_time_s": "s",
        "specific_impulse_s": "s", "length_m": "m", "diameter_m": "m",
        "drag_coefficient": "dimensionless", "initial_mass_kg": "kg",
        "mass_flow_kg_s": "kg/s", "navigation_constant": "dimensionless",
        "maximum_overload_g": "g", "hit_radius_m": "m",
        "minimum_speed_mps": "m/s", "arming_time_s": "s",
        "missile_launch_speed_mps": "m/s",
    }, PAPER_UNDISCLOSED_ENGINEERING),
    "reward_mode": sv(PAPER_REWARD_MODE_6V6, PAPER_EQUATION_OPERATIONAL),
    "reward_version": sv(PAPER_REWARD_VERSION_6V6, PAPER_EQUATION_OPERATIONAL),
    "altitude_reward_engineering_thresholds_m": sv(
        {"h_min_m": 0.0, "h_att_m": 2000.0, "h_adv_m": 5000.0,
         "h_max_m": 10000.0, "d_att_max_m": 15000.0},
        PAPER_UNDISCLOSED_ENGINEERING),
    "altitude_pair_aggregation": sv(
        "paper_unresolved_sum_over_alive_enemies", PAPER_UNDISCLOSED_ENGINEERING),
    "q_los_version": sv("observer_velocity_to_target_los_3d_v1", "paper_inferred"),
}


def paper_6v6_environment_snapshot(*, seed: int | None) -> dict:
    payload = {key: asdict(value) for key, value in PAPER_PROFILE_METADATA_6V6.items()}
    payload.update({
        "num_red": 6, "num_blue": 6, "sim_freq": 60,
        "agent_interaction_steps": 12, "decision_frequency_hz": 5.0,
        "max_episode_length": 1400,
        "environment_config": asdict(PAPER_ENVIRONMENT_CONFIG_6V6),
        "seed": seed,
    })
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("seed", None)
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["environment_config_fingerprint"] = hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()
    return payload
