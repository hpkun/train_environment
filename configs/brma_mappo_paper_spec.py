"""Single source of truth for the BRMA-MAPPO paper profile.

Every value carries an explicit provenance.  Values labelled
``paper_unspecified_engineering`` are reproducible project choices, not values
reported by the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from typing import Any


PAPER_EXPLICIT = "paper_explicit"
PAPER_EQUATION = "paper_equation"
PAPER_INFERRED = "paper_inferred"
PAPER_ENGINEERING = "paper_unspecified_engineering"


@dataclass(frozen=True)
class SourcedValue:
    value: Any
    source: str
    note: str = ""


def sv(value: Any, source: str, note: str = "") -> SourcedValue:
    return SourcedValue(value=value, source=source, note=note)


PAPER_SPEC = {
    "environment_version": sv("brma_paper_profile_v1", PAPER_INFERRED),
    "episode_steps": sv(1400, PAPER_EXPLICIT),
    "canonical_team_size": sv(6, PAPER_EXPLICIT),
    "main_team_size": sv(3, PAPER_INFERRED, "User main scale; formulas unchanged."),
    "battlefield_size_m": sv((100_000.0, 100_000.0, 10_000.0), PAPER_EXPLICIT),
    "boundary_reward_half_width_m": sv(
        40_000.0, PAPER_EQUATION,
        "Eq.18 uses +/-40 km although Table 4 states a 100 km square."),
    "initial_head_on_range_m": sv(10_000.0, PAPER_EXPLICIT),
    "maximum_aircraft_speed_mps": sv(600.0, PAPER_EXPLICIT),
    "maximum_aircraft_load_g": sv(9.0, PAPER_EXPLICIT),
    "action_pitch_rad": sv((-1.5707963267948966, 1.5707963267948966), PAPER_EXPLICIT),
    "action_heading_rad": sv((-3.141592653589793, 3.141592653589793), PAPER_EXPLICIT),
    "action_speed_mach": sv((0.3, 1.2), PAPER_EXPLICIT),
    "mach_reference_mps": sv(340.0, PAPER_ENGINEERING, "Paper does not define atmosphere conversion."),
    "radar_azimuth_rad": sv((-1.0471975511965976, 1.0471975511965976), PAPER_EXPLICIT),
    "radar_elevation_rad": sv((-0.17453292519943295, 0.5585053606381855), PAPER_EXPLICIT),
    "radar_range_constant": sv(40_000.0, PAPER_ENGINEERING),
    "radar_rcs_frontal_m2": sv(0.1, PAPER_ENGINEERING),
    "radar_rcs_side_m2": sv(2.0, PAPER_ENGINEERING),
    "electro_optical_range_m": sv(10_000.0, PAPER_EXPLICIT),
    "electro_optical_half_angle_deg": sv(45.0, PAPER_ENGINEERING),
    "minimum_launch_range_m": sv(500.0, PAPER_ENGINEERING),
    "lock_time_s": sv(0.25, PAPER_EXPLICIT),
    "launch_interval_s": sv(0.5, PAPER_EXPLICIT),
    "missile_max_load_g": sv(30.0, PAPER_EQUATION),
    "reward_weights": sv(
        {"pitch": 0.01, "roll": 0.002, "altitude": 0.04,
         "boundary": 0.04, "speed": 0.02, "advantage": 0.15},
        PAPER_EQUATION),
    "altitude_pair_aggregation": sv(
        "sum", PAPER_INFERRED,
        "Pairwise combat rewards are summed consistently with Eq.22; the paper does not spell out Eq.17 aggregation."),
    "q_los_definition": sv(
        "observer_velocity_to_observer_target_los_3d", PAPER_INFERRED,
        "Consistent directional q_ij/q_ji interpretation across Table 2 and Eq.20-22."),
    "ppo": sv({"actor_lr": 2e-4, "critic_lr": 5e-4, "entropy_coef": 0.05,
               "hidden": (128, 128), "buffer_size": 2000,
               "rollout_envs": 32, "total_steps": 10_000_000,
               "evaluation_episodes": 100, "seed_count": 3}, PAPER_EXPLICIT),
    "advantage_normalization": sv(True, PAPER_ENGINEERING),
    "value_normalization": sv(False, PAPER_ENGINEERING),
    "value_clipping": sv(False, PAPER_ENGINEERING),
    "action_distribution": sv("tanh_squashed_diagonal_gaussian", PAPER_ENGINEERING),
    "attention_heads": sv(4, PAPER_EXPLICIT),
    "brma_temperature": sv(0.1, PAPER_EXPLICIT),
    "brma_max_mask_allies": sv(2, PAPER_EXPLICIT),
    "brma_max_mask_enemies": sv(2, PAPER_EXPLICIT),
    "brma_entropy_coef": sv(0.05, PAPER_EXPLICIT),
    "brma_mask_lr": sv(5e-4, PAPER_EXPLICIT),
    "pid_gains": sv("configured in PIDController", PAPER_ENGINEERING),
    "missile_physical_parameters": sv("configured in MissileSimulator", PAPER_ENGINEERING),
}


def paper_value(name: str):
    """Return the configured value while keeping provenance available."""
    return PAPER_SPEC[name].value


@dataclass(frozen=True)
class ScenarioConfig:
    arena_half_width_m: SourcedValue = field(default_factory=lambda: sv(50_000.0, PAPER_EXPLICIT))
    arena_altitude_min_m: SourcedValue = field(default_factory=lambda: sv(0.0, PAPER_INFERRED, "Sea-level ground reference."))
    arena_altitude_max_m: SourcedValue = field(default_factory=lambda: sv(10_000.0, PAPER_EXPLICIT))
    reward_boundary_half_width_m: SourcedValue = field(default_factory=lambda: sv(40_000.0, PAPER_EQUATION))
    initial_head_on_range_m: SourcedValue = field(default_factory=lambda: sv(10_000.0, PAPER_EXPLICIT))
    initial_altitude_m: SourcedValue = field(default_factory=lambda: sv(6096.0, PAPER_ENGINEERING, "20,000 ft."))
    initial_speed_mps: SourcedValue = field(default_factory=lambda: sv(304.8, PAPER_ENGINEERING, "1000 ft/s."))
    formation_spacing_m: SourcedValue = field(default_factory=lambda: sv(500.0, PAPER_ENGINEERING))
    reference_longitude_deg: SourcedValue = field(default_factory=lambda: sv(120.0, PAPER_ENGINEERING))
    reference_latitude_deg: SourcedValue = field(default_factory=lambda: sv(60.0, PAPER_ENGINEERING))
    aircraft_model: SourcedValue = field(default_factory=lambda: sv("f16", PAPER_ENGINEERING))
    missiles_per_aircraft: SourcedValue = field(default_factory=lambda: sv(999, PAPER_ENGINEERING))
    episode_steps: SourcedValue = field(default_factory=lambda: sv(1400, PAPER_EXPLICIT))


@dataclass(frozen=True)
class AircraftConstraintConfig:
    maximum_speed_mps: SourcedValue = field(default_factory=lambda: sv(600.0, PAPER_EXPLICIT))
    maximum_load_g: SourcedValue = field(default_factory=lambda: sv(9.0, PAPER_EXPLICIT))
    overspeed_throttle_limit: SourcedValue = field(default_factory=lambda: sv(0.0, PAPER_INFERRED))
    engineering_overload_termination_s: SourcedValue = field(default_factory=lambda: sv(10.0, PAPER_ENGINEERING))


@dataclass(frozen=True)
class PIDConfig:
    roll_gains: SourcedValue = field(default_factory=lambda: sv((0.15, 0.5, 0.05), PAPER_ENGINEERING))
    pitch_gains: SourcedValue = field(default_factory=lambda: sv((2.5, 0.5, 0.1), PAPER_ENGINEERING))
    speed_gains: SourcedValue = field(default_factory=lambda: sv((0.04, 0.01, 0.003), PAPER_ENGINEERING))
    integral_error_limits: SourcedValue = field(default_factory=lambda: sv((1.5707963268, 1.5707963268, 306.0), PAPER_ENGINEERING))
    throttle_base: SourcedValue = field(default_factory=lambda: sv(0.0, PAPER_ENGINEERING))
    anti_windup_mode: SourcedValue = field(default_factory=lambda: sv("back_calculation", PAPER_ENGINEERING))
    actuator_signs: SourcedValue = field(default_factory=lambda: sv((1.0, -1.0, 0.0, 1.0), PAPER_ENGINEERING))


@dataclass(frozen=True)
class RCSConfig:
    azimuth_grid_deg: SourcedValue = field(default_factory=lambda: sv((-180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 180.0), PAPER_ENGINEERING))
    elevation_grid_deg: SourcedValue = field(default_factory=lambda: sv((-90.0, 0.0, 90.0), PAPER_ENGINEERING))
    table_m2: SourcedValue = field(default_factory=lambda: sv((
        (0.3, 1.2, 0.15, 0.1, 0.15, 1.2, 0.3),
        (0.2, 2.0, 0.1, 0.1, 0.1, 2.0, 0.2),
        (0.3, 1.2, 0.15, 0.1, 0.15, 1.2, 0.3),
    ), PAPER_ENGINEERING, "Replaceable 2D approximation; full paper table is unpublished."))
    range_constant: SourcedValue = field(default_factory=lambda: sv(40_000.0, PAPER_ENGINEERING))


@dataclass(frozen=True)
class RadarConfig:
    azimuth_min_rad: SourcedValue = field(default_factory=lambda: sv(-1.0471975512, PAPER_EXPLICIT))
    azimuth_max_rad: SourcedValue = field(default_factory=lambda: sv(1.0471975512, PAPER_EXPLICIT))
    elevation_min_rad: SourcedValue = field(default_factory=lambda: sv(-0.1745329252, PAPER_EXPLICIT))
    elevation_max_rad: SourcedValue = field(default_factory=lambda: sv(0.5585053606, PAPER_EXPLICIT))


@dataclass(frozen=True)
class AwacsConfig:
    update_period_s: SourcedValue = field(default_factory=lambda: sv(1.0, PAPER_ENGINEERING))
    track_hold_s: SourcedValue = field(default_factory=lambda: sv(3.0, PAPER_ENGINEERING))
    horizontal_error_std_m: SourcedValue = field(default_factory=lambda: sv(500.0, PAPER_ENGINEERING))
    vertical_error_std_m: SourcedValue = field(default_factory=lambda: sv(200.0, PAPER_ENGINEERING))


@dataclass(frozen=True)
class RwsConfig:
    bearing_error_std_rad: SourcedValue = field(default_factory=lambda: sv(0.0872664626, PAPER_ENGINEERING))


@dataclass(frozen=True)
class ElectroOpticalConfig:
    maximum_range_m: SourcedValue = field(default_factory=lambda: sv(10_000.0, PAPER_EXPLICIT))
    half_angle_rad: SourcedValue = field(default_factory=lambda: sv(0.7853981634, PAPER_ENGINEERING))
    minimum_launch_range_m: SourcedValue = field(default_factory=lambda: sv(500.0, PAPER_ENGINEERING))


@dataclass(frozen=True)
class MissileWarningConfig:
    detection_mode: SourcedValue = field(default_factory=lambda: sv("all_targeting_live_missiles", PAPER_ENGINEERING))


@dataclass(frozen=True)
class MissileConfig:
    model: SourcedValue = field(default_factory=lambda: sv("AIM-9L", PAPER_ENGINEERING))
    maximum_flight_time_s: SourcedValue = field(default_factory=lambda: sv(60.0, PAPER_ENGINEERING))
    thrust_time_s: SourcedValue = field(default_factory=lambda: sv(3.0, PAPER_ENGINEERING))
    specific_impulse_s: SourcedValue = field(default_factory=lambda: sv(240.0, PAPER_ENGINEERING))
    length_m: SourcedValue = field(default_factory=lambda: sv(2.87, PAPER_ENGINEERING))
    diameter_m: SourcedValue = field(default_factory=lambda: sv(0.127, PAPER_ENGINEERING))
    drag_coefficient: SourcedValue = field(default_factory=lambda: sv(0.22, PAPER_ENGINEERING))
    initial_mass_kg: SourcedValue = field(default_factory=lambda: sv(84.0, PAPER_ENGINEERING))
    mass_flow_kg_s: SourcedValue = field(default_factory=lambda: sv(6.0, PAPER_ENGINEERING))
    navigation_constant: SourcedValue = field(default_factory=lambda: sv(3.0, PAPER_EQUATION))
    maximum_overload_g: SourcedValue = field(default_factory=lambda: sv(30.0, PAPER_EQUATION))
    hit_radius_m: SourcedValue = field(default_factory=lambda: sv(300.0, PAPER_ENGINEERING))
    minimum_speed_mps: SourcedValue = field(default_factory=lambda: sv(150.0, PAPER_ENGINEERING))
    arming_time_s: SourcedValue = field(default_factory=lambda: sv(0.15, PAPER_ENGINEERING))
    density_model: SourcedValue = field(default_factory=lambda: sv("rho0_exp_altitude_over_9300", PAPER_ENGINEERING))
    guidance_mode: SourcedValue = field(default_factory=lambda: sv("paper_eq9", PAPER_EQUATION))


@dataclass(frozen=True)
class FireControlConfig:
    lock_time_s: SourcedValue = field(default_factory=lambda: sv(0.25, PAPER_EXPLICIT))
    launch_interval_s: SourcedValue = field(default_factory=lambda: sv(0.5, PAPER_EXPLICIT))
    rear_hemisphere_ta_rad: SourcedValue = field(default_factory=lambda: sv(1.5707963268, PAPER_EXPLICIT))


@dataclass(frozen=True)
class RewardConfig:
    weights: SourcedValue = field(default_factory=lambda: PAPER_SPEC["reward_weights"])
    terminal_coefficient: SourcedValue = field(default_factory=lambda: sv(30.0, PAPER_EQUATION))
    mach_reference_mps: SourcedValue = field(default_factory=lambda: PAPER_SPEC["mach_reference_mps"])
    altitude_interpretation: SourcedValue = field(default_factory=lambda: PAPER_SPEC["altitude_pair_aggregation"])


@dataclass(frozen=True)
class PaperEnvironmentConfig:
    version: SourcedValue = field(default_factory=lambda: PAPER_SPEC["environment_version"])
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    aircraft: AircraftConstraintConfig = field(default_factory=AircraftConstraintConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    rcs: RCSConfig = field(default_factory=RCSConfig)
    awacs: AwacsConfig = field(default_factory=AwacsConfig)
    rws: RwsConfig = field(default_factory=RwsConfig)
    electro_optical: ElectroOpticalConfig = field(default_factory=ElectroOpticalConfig)
    missile_warning: MissileWarningConfig = field(default_factory=MissileWarningConfig)
    missile: MissileConfig = field(default_factory=MissileConfig)
    fire_control: FireControlConfig = field(default_factory=FireControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


DEFAULT_PAPER_ENVIRONMENT_CONFIG = PaperEnvironmentConfig()


def environment_config_snapshot(config: PaperEnvironmentConfig, *, num_red: int,
                                num_blue: int, sim_freq: int,
                                agent_interaction_steps: int, seed: int | None,
                                blue_policy_profile: str = "paper_pursuit") -> dict:
    payload = asdict(config)
    payload.update({"num_red": int(num_red), "num_blue": int(num_blue),
                    "sim_freq": int(sim_freq),
                    "agent_interaction_steps": int(agent_interaction_steps),
                    "decision_frequency_hz": float(sim_freq / agent_interaction_steps),
                    "blue_policy_profile": str(blue_policy_profile),
                    "seed": seed})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["environment_config_fingerprint"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def config_value(item: SourcedValue):
    return item.value
