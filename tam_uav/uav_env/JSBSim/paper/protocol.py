"""Protocol and provenance contract for the isolated TAM paper environment."""

from __future__ import annotations

import math


ENVIRONMENT_FIDELITY_REVISION = "published_environment_reconstruction_v5"
NOMINAL_PERTURBATION = "none"
GENERALIZATION_PERTURBATION_LEVELS = ("low", "medium", "large")
GENERALIZATION_EPISODES_PER_LEVEL = 50
PAPER_NOMINAL_PROTOCOL = "paper_nominal"
PAPER_5V4_GENERALIZATION_PROTOCOL = "paper_5v4_generalization"
PAPER_SILENT_ASSUMPTIONS_PRESENT = True
TERMINATION_RESOLUTION = "decision_step_boundary"
SCENARIOS = ("2v2", "3v2", "5v4")
BLUE_POLICY_FIDELITY = "minimal_greedy_basic_manoeuvre_reconstruction"
REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED = False
BASIC_MANOEUVRE_ACTION_MAPPING_UNPUBLISHED = True


REQUIRED_PUBLISHED = {
    "simulation_frequency_hz", "decision_frequency_hz",
    "physics_frames_per_action", "episode_limit_steps", "maximum_speed_mps",
    "minimum_safe_altitude_m", "maximum_aircraft_overload_g",
    "maximum_attack_range_m", "launch_interval_s", "maximum_overload_g",
    "navigation_gain_y", "navigation_gain_z", "missile_mass_kg",
    "missile_length_m", "missile_diameter_m", "direct_fcs_action_dimensions",
    "action_levels_per_dimension", "throttle_command_range",
    "control_surface_command_range", "uav_reward_weights", "event_rewards",
}
REQUIRED_UNPUBLISHED = {
    "execution_aircraft_models", "missile_initial_speed_mps",
    "powered_duration_s", "powered_acceleration_mps2", "hit_radius_m",
    "effective_quadratic_drag_per_m", "missile_timeout_s",
    "missile_speed_reward_norm_mps", "reward_global_scale",
    "height_reward_approximation", "combat_zone_radius_m",
    "position_normalization_m", "altitude_normalization_m",
    "situation_height_normalization_m", "entity_slot_capacity",
    "mav_reward_distance_thresholds_m", "blue_policy",
    "target_selection_tie_break", "dodge_threat_selection",
    "posthumous_reward_semantics", "termination_semantics",
}


def validate_parameter_provenance(config: dict) -> None:
    """Reject hidden defaults and values assigned to the wrong provenance class."""
    for section in ("published_parameters", "unpublished_parameters",
                    "derived_parameters", "scenario_initial_conditions",
                    "role_definitions"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing required provenance section {section!r}")
    published = config["published_parameters"]
    unpublished = config["unpublished_parameters"]
    derived = config["derived_parameters"]
    missing = sorted(REQUIRED_PUBLISHED - set(published))
    if missing:
        raise ValueError(f"published_parameters missing {missing}")
    missing = sorted(REQUIRED_UNPUBLISHED - set(unpublished))
    if missing:
        raise ValueError(f"unpublished_parameters missing {missing}")
    overlap = (set(published) & set(unpublished)) | (set(published) & set(derived)) | (
        set(unpublished) & set(derived))
    if overlap:
        raise ValueError(f"parameter provenance overlap: {sorted(overlap)}")
    allowed_derived = {"decision_dt_s"}
    if set(derived) != allowed_derived:
        raise ValueError("derived_parameters may contain only decision_dt_s")
    expected_dt = (float(published["physics_frames_per_action"])
                   / float(published["simulation_frequency_hz"]))
    if not math.isclose(float(derived["decision_dt_s"]), expected_dt):
        raise ValueError("derived decision_dt_s is inconsistent with 12/60")
    if not math.isclose(float(published["decision_frequency_hz"]), 1.0 / expected_dt):
        raise ValueError("decision_frequency_hz is inconsistent with decision_dt_s")
    roles = config["role_definitions"]
    for role in ("mav", "attack_uav"):
        if role not in roles:
            raise ValueError(f"role_definitions missing {role!r}")
        if "unpublished_execution_aircraft_model" not in roles[role]:
            raise ValueError(f"role {role!r} lacks explicit execution-model provenance")


def environment_values(unpublished: dict) -> dict:
    slots = unpublished["entity_slot_capacity"]
    mav = unpublished["mav_reward_distance_thresholds_m"]
    return {
        "combat_zone_radius_m": float(unpublished["combat_zone_radius_m"]),
        "position_norm_m": float(unpublished["position_normalization_m"]),
        "altitude_norm_m": float(unpublished["altitude_normalization_m"]),
        "situation_height_norm_m": float(unpublished["situation_height_normalization_m"]),
        "max_incoming_missiles": int(slots["incoming_missiles"]),
        "max_red_agents": int(slots["red"]),
        "max_blue_agents": int(slots["blue"]),
        "mav_d_danger_m": float(mav["danger"]),
        "mav_d_safe_m": float(mav["safe"]),
        "mav_d_opt_m": float(mav["optimal"]),
        "mav_d_max_m": float(mav["maximum"]),
    }


def validate_nominal_protocol(scenario: str, perturbation: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"nominal scenario must be one of {SCENARIOS}, got {scenario!r}")
    if perturbation != NOMINAL_PERTURBATION:
        raise ValueError("paper_nominal only permits perturbation='none'")


def validate_generalization_protocol(scenario: str, perturbation: str) -> None:
    if scenario != "5v4":
        raise ValueError("paper_5v4_generalization requires scenario='5v4'")
    if perturbation not in GENERALIZATION_PERTURBATION_LEVELS:
        raise ValueError("generalization perturbation must be low, medium, or large")


def protocol_metadata(scenario: str, perturbation: str, dynamics_backend: str,
                      experiment_protocol: str) -> dict:
    if experiment_protocol == PAPER_NOMINAL_PROTOCOL:
        validate_nominal_protocol(scenario, perturbation)
        nominal, generalization = True, False
    elif experiment_protocol == PAPER_5V4_GENERALIZATION_PROTOCOL:
        validate_generalization_protocol(scenario, perturbation)
        nominal, generalization = False, True
    else:
        raise ValueError(f"unknown paper experiment protocol {experiment_protocol!r}")
    return {
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "experiment_protocol": experiment_protocol, "scenario": scenario,
        "initial_perturbation": perturbation, "dynamics_backend": dynamics_backend,
        "paper_nominal_experiment": nominal,
        "paper_generalization_experiment": generalization,
        "paper_silent_assumptions_present": PAPER_SILENT_ASSUMPTIONS_PRESENT,
        "termination_resolution": TERMINATION_RESOLUTION,
        "neutral_action_semantics": "nearest_positive_center_not_exact_zero",
        "blue_policy_fidelity": BLUE_POLICY_FIDELITY,
        "reference_8_exact_blue_fsm_reproduced": REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED,
        "basic_manoeuvre_action_mapping_unpublished": BASIC_MANOEUVRE_ACTION_MAPPING_UNPUBLISHED,
        "height_reward_exact_formula_available": False,
    }


def checkpoint_lineage(metadata: dict) -> dict:
    return {f"checkpoint_{key}": metadata.get(key) for key in (
        "environment_fidelity_revision", "experiment_protocol", "initial_perturbation",
        "dynamics_backend", "environment_steps", "episodes", "algorithm_mode",
        "paper_silent_assumptions_present", "blue_policy_fidelity",
        "reference_8_exact_blue_fsm_reproduced")}
