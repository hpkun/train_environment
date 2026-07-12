"""Single source of truth for the BRMA-MAPPO paper profile.

Every value carries an explicit provenance.  Values labelled
``paper_unspecified_engineering`` are reproducible project choices, not values
reported by the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
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
