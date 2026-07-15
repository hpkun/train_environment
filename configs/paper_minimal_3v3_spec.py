"""Isolated specification for the deterministic minimal 3V3 paper profile."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    PAPER_EQUATION,
    PAPER_EXPLICIT,
    PAPER_INFERRED,
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


PAPER_MINIMAL_SOURCE = "paper_unspecified_minimal"
PAPER_MINIMAL_ENVIRONMENT_PROFILE = "paper_minimal_3v3_v1"
REFERENCE_ENVIRONMENT_PROFILE = "brma_paper_profile_v1"
MINIMAL_MISSILE_LAUNCH_SPEED_MPS = 600.0
MINIMAL_MISSILE_HIT_RADIUS_M = 300.0
MINIMAL_MISSILE_OVERSHOOT_WINDOW_S = 0.5
MINIMAL_MISSILE_RNG_VERSION = "seedsequence_team_pair_launch_v1"
MINIMAL_EXTREME_LOAD_INVALID_THRESHOLD_G = 30.0


MINIMAL_SCENARIO = ScenarioConfig(
    arena_half_width_m=sv(50_000.0, PAPER_EXPLICIT),
    arena_altitude_min_m=sv(0.0, PAPER_INFERRED, "Sea-level ground reference."),
    arena_altitude_max_m=sv(10_000.0, PAPER_EXPLICIT),
    reward_boundary_half_width_m=sv(40_000.0, PAPER_EQUATION),
    initial_head_on_range_m=sv(10_000.0, PAPER_EXPLICIT),
    initial_altitude_m=sv(6_000.0, PAPER_MINIMAL_SOURCE),
    initial_speed_mps=sv(300.0, PAPER_MINIMAL_SOURCE),
    formation_spacing_m=sv(1_000.0, PAPER_MINIMAL_SOURCE),
    reference_longitude_deg=sv(120.0, PAPER_MINIMAL_SOURCE),
    reference_latitude_deg=sv(60.0, PAPER_MINIMAL_SOURCE),
    aircraft_model=sv("f16", PAPER_MINIMAL_SOURCE),
    missiles_per_aircraft=sv(999, PAPER_MINIMAL_SOURCE),
    episode_steps=sv(1_400, PAPER_EXPLICIT),
)

MINIMAL_RCS = RCSConfig(
    azimuth_grid_deg=sv((-180.0, 180.0), PAPER_MINIMAL_SOURCE),
    elevation_grid_deg=sv((-90.0, 90.0), PAPER_MINIMAL_SOURCE),
    table_m2=sv(((1.0, 1.0), (1.0, 1.0)), PAPER_MINIMAL_SOURCE),
    range_constant=sv(40_000.0, PAPER_MINIMAL_SOURCE),
)

MINIMAL_PID = PIDConfig(
    roll_gains=sv((0.15, 0.5, 0.05), PAPER_MINIMAL_SOURCE),
    pitch_gains=sv((2.5, 0.5, 0.1), PAPER_MINIMAL_SOURCE),
    speed_gains=sv((0.04, 0.01, 0.003), PAPER_MINIMAL_SOURCE),
    integral_error_limits=sv((1.5707963268, 1.5707963268, 306.0), PAPER_MINIMAL_SOURCE),
    throttle_base=sv(0.0, PAPER_MINIMAL_SOURCE),
    anti_windup_mode=sv("back_calculation", PAPER_MINIMAL_SOURCE),
    actuator_signs=sv((1.0, -1.0, 0.0, 1.0), PAPER_MINIMAL_SOURCE),
)

MINIMAL_MISSILE = replace(
    MissileConfig(),
    model=sv("paper_minimal_point_mass", PAPER_INFERRED),
    maximum_flight_time_s=sv(60.0, PAPER_MINIMAL_SOURCE),
    thrust_time_s=sv(0.0, PAPER_MINIMAL_SOURCE),
    specific_impulse_s=sv(0.0, PAPER_MINIMAL_SOURCE),
    length_m=sv(0.0, PAPER_MINIMAL_SOURCE),
    diameter_m=sv(0.0, PAPER_MINIMAL_SOURCE),
    drag_coefficient=sv(0.0, PAPER_MINIMAL_SOURCE),
    initial_mass_kg=sv(1.0, PAPER_MINIMAL_SOURCE),
    mass_flow_kg_s=sv(0.0, PAPER_MINIMAL_SOURCE),
    hit_radius_m=sv(MINIMAL_MISSILE_HIT_RADIUS_M, PAPER_MINIMAL_SOURCE),
    minimum_speed_mps=sv(0.0, PAPER_MINIMAL_SOURCE),
    arming_time_s=sv(0.0, PAPER_MINIMAL_SOURCE),
    density_model=sv("disabled", PAPER_MINIMAL_SOURCE),
    guidance_mode=sv("paper_minimal_point_mass_v1", PAPER_INFERRED),
)

MINIMAL_PAPER_ENVIRONMENT_CONFIG = PaperEnvironmentConfig(
    version=sv(PAPER_MINIMAL_ENVIRONMENT_PROFILE, PAPER_INFERRED),
    scenario=MINIMAL_SCENARIO,
    aircraft=DEFAULT_PAPER_ENVIRONMENT_CONFIG.aircraft,
    pid=MINIMAL_PID,
    radar=DEFAULT_PAPER_ENVIRONMENT_CONFIG.radar,
    rcs=MINIMAL_RCS,
    awacs=AwacsConfig(
        update_period_s=sv(0.0, PAPER_MINIMAL_SOURCE),
        track_hold_s=sv(0.0, PAPER_MINIMAL_SOURCE),
        horizontal_error_std_m=sv(0.0, PAPER_MINIMAL_SOURCE),
        vertical_error_std_m=sv(0.0, PAPER_MINIMAL_SOURCE),
    ),
    rws=RwsConfig(bearing_error_std_rad=sv(0.0, PAPER_MINIMAL_SOURCE)),
    electro_optical=ElectroOpticalConfig(
        maximum_range_m=sv(10_000.0, PAPER_EXPLICIT),
        half_angle_rad=sv(3.141592653589793, PAPER_MINIMAL_SOURCE,
                          "EO visibility is distance-only in the minimal profile."),
        minimum_launch_range_m=sv(0.0, PAPER_MINIMAL_SOURCE),
    ),
    missile_warning=MissileWarningConfig(
        detection_mode=sv("all_targeting_live_missiles", PAPER_INFERRED)),
    missile=MINIMAL_MISSILE,
    fire_control=DEFAULT_PAPER_ENVIRONMENT_CONFIG.fire_control,
    reward=replace(
        DEFAULT_PAPER_ENVIRONMENT_CONFIG.reward,
        altitude_interpretation=sv(
            "paper_unspecified_scale_invariant_mean", PAPER_MINIMAL_SOURCE)),
)


MINIMAL_PROFILE_METADATA = {
    "environment_profile": sv(PAPER_MINIMAL_ENVIRONMENT_PROFILE, PAPER_INFERRED),
    "initial_condition_profile": sv("paper_minimal_head_on_v1", PAPER_INFERRED),
    "sensor_profile": sv("paper_minimal_deterministic_v1", PAPER_INFERRED),
    "blue_policy_profile": sv("paper_minimal_fixed_pair_v1", PAPER_INFERRED),
    "mws_evasion_profile": sv("paper_minimal_mws_evasion_v1", PAPER_INFERRED),
    "pid_profile": sv("paper_minimal_shared_v1", PAPER_INFERRED),
    "missile_profile": sv("paper_minimal_point_mass_v1", PAPER_INFERRED),
    "reward_version": sv("paper_literal_minimal_unspecified_v1", PAPER_INFERRED),
    "altitude_pair_aggregation": sv(
        "paper_unspecified_scale_invariant_mean", PAPER_MINIMAL_SOURCE),
    "altitude_high_tail": sv(0.0, PAPER_MINIMAL_SOURCE),
    "aircraft_model": MINIMAL_SCENARIO.aircraft_model,
    "constant_rcs": sv(1.0, PAPER_MINIMAL_SOURCE),
    "radar_range_constant": MINIMAL_RCS.range_constant,
    "initial_altitude_m": MINIMAL_SCENARIO.initial_altitude_m,
    "initial_speed_mps": MINIMAL_SCENARIO.initial_speed_mps,
    "formation_spacing_m": MINIMAL_SCENARIO.formation_spacing_m,
    "initial_missile_speed_mps": sv(
        MINIMAL_MISSILE_LAUNCH_SPEED_MPS, PAPER_MINIMAL_SOURCE),
    "missile_hit_radius_m": sv(
        MINIMAL_MISSILE_HIT_RADIUS_M, PAPER_MINIMAL_SOURCE),
    "missile_overshoot_window_s": sv(
        MINIMAL_MISSILE_OVERSHOOT_WINDOW_S, PAPER_MINIMAL_SOURCE),
    "missile_rng_version": sv(
        MINIMAL_MISSILE_RNG_VERSION, PAPER_MINIMAL_SOURCE),
    "target_deconfliction": sv("hot_update_live_missile_target_v1", PAPER_INFERRED),
    "load_limiter_mode": sv("symmetric_reactive_9g_v1", PAPER_MINIMAL_SOURCE),
    "maximum_load_g": sv(9.0, PAPER_EXPLICIT),
    "extreme_load_invalid_threshold_g": sv(
        MINIMAL_EXTREME_LOAD_INVALID_THRESHOLD_G, PAPER_MINIMAL_SOURCE),
    "speed_limiter_mode": sv(
        "throttle_cut_velocity_projection_600mps_v1", PAPER_MINIMAL_SOURCE),
    "vertical_bounds_m": sv((0.0, 10_000.0), PAPER_EXPLICIT),
    "maximum_live_missiles_observed": sv(
        "runtime_episode_metric", PAPER_INFERRED),
}


def minimal_environment_snapshot(*, num_red: int, num_blue: int, sim_freq: int,
                                 agent_interaction_steps: int,
                                 max_episode_length: int,
                                 blue_policy_profile: str, seed: int | None) -> dict:
    """Return a serializable, provenance-preserving minimal-profile snapshot."""
    payload = {
        key: asdict(value) for key, value in MINIMAL_PROFILE_METADATA.items()
    }
    payload.update({
        "num_red": int(num_red),
        "num_blue": int(num_blue),
        "sim_freq": int(sim_freq),
        "agent_interaction_steps": int(agent_interaction_steps),
        "decision_frequency_hz": float(sim_freq / agent_interaction_steps),
        "max_episode_length": int(max_episode_length),
        "blue_policy_profile": asdict(sv(str(blue_policy_profile), PAPER_INFERRED)),
        "environment_config": asdict(MINIMAL_PAPER_ENVIRONMENT_CONFIG),
        "seed": seed,
    })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["environment_config_fingerprint"] = hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()
    return payload
