"""Learnability adaptation of the paper-aligned 3V3 environment."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

from configs.brma_mappo_paper_spec import (
    PAPER_EXPLICIT,
    ElectroOpticalConfig,
    MissileConfig,
    PaperEnvironmentConfig,
    sv,
)
from configs.paper_minimal_3v3_spec import (
    MINIMAL_PAPER_ENVIRONMENT_CONFIG,
    MINIMAL_SCENARIO,
)
from my_uav_env.pid_controller import PAPER_PID_ERROR_DEFINITION


LEARNABILITY_ADAPTATION = "learnability_adaptation"
PAPER_UNSPECIFIED_ENGINEERING = "paper_unspecified_engineering"
PAPER_LEARNABLE_ENVIRONMENT_PROFILE = "paper_learnable_3v3_v1"
LEARNABLE_BLUE_POLICY_PROFILE = "paper_learnable_fixed_pair_v1"
LEARNABLE_MISSILE_GUIDANCE_MODE = "paper_learnable_point_mass_v1"
LEARNABLE_INITIALIZATION_MODES = (
    "deterministic_v1",
    "small_symmetric_jitter_v1",
)

LEARNABLE_MISSILE_LAUNCH_SPEED_MPS = 800.0
LEARNABLE_MISSILE_HIT_RADIUS_M = 100.0
LEARNABLE_MISSILE_OVERSHOOT_WINDOW_S = 0.5
LEARNABLE_MISSILE_OVERSHOOT_DISTANCE_HYSTERESIS_M = 50.0
LEARNABLE_MISSILE_POSITIVE_CLOSING_THRESHOLD_MPS = 1.0
LEARNABLE_PID_THROTTLE_BASE = 0.8
LEARNABLE_LOAD_PROTECTION_START_G = 9.0
LEARNABLE_PERSISTENT_EXTREME_G = 30.0
LEARNABLE_PERSISTENT_EXTREME_FRAMES = 3
LEARNABLE_CATASTROPHIC_G = 100.0


LEARNABLE_MISSILE = replace(
    MissileConfig(),
    model=sv("paper_learnable_point_mass", LEARNABILITY_ADAPTATION),
    maximum_flight_time_s=sv(60.0, LEARNABILITY_ADAPTATION),
    thrust_time_s=sv(0.0, LEARNABILITY_ADAPTATION),
    specific_impulse_s=sv(0.0, LEARNABILITY_ADAPTATION),
    length_m=sv(0.0, LEARNABILITY_ADAPTATION),
    diameter_m=sv(0.0, LEARNABILITY_ADAPTATION),
    drag_coefficient=sv(0.0, LEARNABILITY_ADAPTATION),
    initial_mass_kg=sv(1.0, LEARNABILITY_ADAPTATION),
    mass_flow_kg_s=sv(0.0, LEARNABILITY_ADAPTATION),
    hit_radius_m=sv(LEARNABLE_MISSILE_HIT_RADIUS_M, LEARNABILITY_ADAPTATION),
    minimum_speed_mps=sv(0.0, LEARNABILITY_ADAPTATION),
    arming_time_s=sv(0.25, LEARNABILITY_ADAPTATION),
    density_model=sv("disabled", LEARNABILITY_ADAPTATION),
    guidance_mode=sv(LEARNABLE_MISSILE_GUIDANCE_MODE, LEARNABILITY_ADAPTATION),
    maximum_overload_g=sv(30.0, LEARNABILITY_ADAPTATION),
)

LEARNABLE_PID = replace(
    MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid,
    roll_gains=sv(
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid.roll_gains.value,
        PAPER_UNSPECIFIED_ENGINEERING),
    pitch_gains=sv(
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid.pitch_gains.value,
        PAPER_UNSPECIFIED_ENGINEERING),
    speed_gains=sv(
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid.speed_gains.value,
        PAPER_UNSPECIFIED_ENGINEERING),
    integral_error_limits=sv(
        MINIMAL_PAPER_ENVIRONMENT_CONFIG.pid.integral_error_limits.value,
        PAPER_UNSPECIFIED_ENGINEERING),
    throttle_base=sv(
        LEARNABLE_PID_THROTTLE_BASE, PAPER_UNSPECIFIED_ENGINEERING),
)

LEARNABLE_PAPER_ENVIRONMENT_CONFIG = replace(
    MINIMAL_PAPER_ENVIRONMENT_CONFIG,
    version=sv(PAPER_LEARNABLE_ENVIRONMENT_PROFILE, LEARNABILITY_ADAPTATION),
    scenario=MINIMAL_SCENARIO,
    electro_optical=ElectroOpticalConfig(
        maximum_range_m=sv(10_000.0, PAPER_EXPLICIT),
        half_angle_rad=sv(0.7853981633974483, LEARNABILITY_ADAPTATION),
        minimum_launch_range_m=sv(0.0, PAPER_UNSPECIFIED_ENGINEERING),
    ),
    missile=LEARNABLE_MISSILE,
    pid=LEARNABLE_PID,
)


LEARNABLE_PROFILE_METADATA = {
    "environment_profile": sv(
        PAPER_LEARNABLE_ENVIRONMENT_PROFILE, LEARNABILITY_ADAPTATION),
    "profile_provenance": sv(LEARNABILITY_ADAPTATION, LEARNABILITY_ADAPTATION),
    "fire_control_profile": sv(
        "paper_learnable_fire_control_v1", LEARNABILITY_ADAPTATION),
    "initial_condition_profile": sv(
        "paper_head_on_3v3_v1", LEARNABILITY_ADAPTATION),
    "initial_condition_randomization_mode": sv(
        "deterministic_v1", LEARNABILITY_ADAPTATION),
    "blue_policy_profile": sv(LEARNABLE_BLUE_POLICY_PROFILE, LEARNABILITY_ADAPTATION),
    "red_mws_mode": sv(
        "scripted_minimal_evasion_v1", LEARNABILITY_ADAPTATION),
    "blue_mws_mode": sv("disabled_fixed_opponent_v1", LEARNABILITY_ADAPTATION),
    "missile_profile": sv(LEARNABLE_MISSILE_GUIDANCE_MODE, LEARNABILITY_ADAPTATION),
    "reward_version": sv(
        "paper_literal_minimal_unspecified_v1", LEARNABILITY_ADAPTATION),
    "initial_missile_direction_mode": sv("aircraft_body_x_v1", LEARNABILITY_ADAPTATION),
    "initial_missile_speed_mps": sv(
        LEARNABLE_MISSILE_LAUNCH_SPEED_MPS, LEARNABILITY_ADAPTATION),
    "missile_hit_radius_m": sv(
        LEARNABLE_MISSILE_HIT_RADIUS_M, LEARNABILITY_ADAPTATION),
    "missile_arming_time_s": sv(0.25, LEARNABILITY_ADAPTATION),
    "missile_overshoot_window_s": sv(
        LEARNABLE_MISSILE_OVERSHOOT_WINDOW_S, LEARNABILITY_ADAPTATION),
    "missile_overshoot_distance_hysteresis_m": sv(
        LEARNABLE_MISSILE_OVERSHOOT_DISTANCE_HYSTERESIS_M,
        LEARNABILITY_ADAPTATION),
    "missile_positive_closing_threshold_mps": sv(
        LEARNABLE_MISSILE_POSITIVE_CLOSING_THRESHOLD_MPS,
        LEARNABILITY_ADAPTATION),
    "eo_maximum_range_m": sv(10_000.0, PAPER_EXPLICIT),
    "launch_positive_finite_range_guard": sv(
        "finite_and_strictly_positive_v1", PAPER_UNSPECIFIED_ENGINEERING),
    "launch_deconfliction": sv(
        "paper_launch_deconfliction_live_missile_v1",
        PAPER_UNSPECIFIED_ENGINEERING),
    "missile_los_definition": sv(
        "paper_eq10_quadrant_preserving_operational_v1",
        PAPER_UNSPECIFIED_ENGINEERING),
    "target_assignment_mode": sv(
        "initial_fixed_pair_reallocate_on_death_v1", LEARNABILITY_ADAPTATION),
    "observation_mode": sv("paper_strict", LEARNABILITY_ADAPTATION),
    "actor_input_dim": sv(60, LEARNABILITY_ADAPTATION),
    "critic_input_dim": sv(30, LEARNABILITY_ADAPTATION),
    "setpoint_rate_limiter": sv(
        "disabled_for_paper_eq12_14",
        LEARNABILITY_ADAPTATION),
    "load_command_scaling": sv(
        "disabled_for_paper_eq12_14",
        LEARNABILITY_ADAPTATION),
    "pid_error_definition": sv(
        PAPER_PID_ERROR_DEFINITION,
        PAPER_UNSPECIFIED_ENGINEERING),
    "extreme_finite_load_guard": sv(
        "paper_unspecified_numerical_guard_30g_3frames_100g_immediate_v1",
        LEARNABILITY_ADAPTATION),
}


def learnable_environment_snapshot(
    *, num_red: int, num_blue: int, sim_freq: int,
    agent_interaction_steps: int, max_episode_length: int,
    blue_policy_profile: str, seed: int | None,
    initial_condition_randomization_mode: str,
) -> dict:
    """Return the serializable contract and stable configuration fingerprint."""
    if initial_condition_randomization_mode not in LEARNABLE_INITIALIZATION_MODES:
        raise ValueError(
            "initial_condition_randomization_mode must be one of "
            f"{LEARNABLE_INITIALIZATION_MODES}")
    payload = {key: asdict(value) for key, value in LEARNABLE_PROFILE_METADATA.items()}
    payload["initial_condition_randomization_mode"] = asdict(sv(
        initial_condition_randomization_mode, LEARNABILITY_ADAPTATION))
    payload.update({
        "num_red": int(num_red),
        "num_blue": int(num_blue),
        "sim_freq": int(sim_freq),
        "agent_interaction_steps": int(agent_interaction_steps),
        "decision_frequency_hz": float(sim_freq / agent_interaction_steps),
        "max_episode_length": int(max_episode_length),
        "blue_policy_profile": asdict(sv(
            str(blue_policy_profile), LEARNABILITY_ADAPTATION)),
        "environment_config": asdict(LEARNABLE_PAPER_ENVIRONMENT_CONFIG),
        "seed": seed,
    })
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("seed", None)
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["environment_config_fingerprint"] = hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()
    return payload
