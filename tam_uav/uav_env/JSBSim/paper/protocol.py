"""Experiment protocol constants for the isolated TAM paper environment."""

from __future__ import annotations


ENVIRONMENT_FIDELITY_REVISION = "published_rules_simplified_v4"
NOMINAL_PERTURBATION = "none"
GENERALIZATION_PERTURBATION_LEVELS = ("low", "medium", "large")
GENERALIZATION_EPISODES_PER_LEVEL = 50
PAPER_NOMINAL_PROTOCOL = "paper_nominal"
PAPER_5V4_GENERALIZATION_PROTOCOL = "paper_5v4_generalization"
PAPER_SILENT_ASSUMPTIONS_PRESENT = True
TERMINATION_RESOLUTION = "decision_step_boundary"
SCENARIOS = ("2v2", "3v2", "5v4")
NOMINAL_ALTITUDE_M = 6000.0
MAX_INCOMING_MISSILES = 8  # 4 opposing attack UAVs x 2 missiles.
BLUE_POLICY_FIDELITY = "minimal_greedy_basic_manoeuvre_reconstruction"
REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED = False


def derived_environment_values(maximum_attack_range_m: float) -> dict:
    attack_range = float(maximum_attack_range_m)
    return {
        "uav_direct_detection_range_m": attack_range,
        "mav_detection_range_m": 2.0 * attack_range,
        "combat_zone_radius_m": 2.0 * attack_range,
        "position_norm_m": 2.0 * attack_range,
        "altitude_norm_m": NOMINAL_ALTITUDE_M,
        "situation_height_norm_m": NOMINAL_ALTITUDE_M,
        "max_incoming_missiles": MAX_INCOMING_MISSILES,
        "mav_d_danger_m": 0.5 * attack_range,
        "mav_d_safe_m": attack_range,
        "mav_d_opt_m": attack_range,
        "mav_d_max_m": 2.0 * attack_range,
    }


def validate_nominal_protocol(scenario: str, perturbation: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"nominal scenario must be one of {SCENARIOS}, got {scenario!r}")
    if perturbation != NOMINAL_PERTURBATION:
        raise ValueError(
            "paper_nominal only permits perturbation='none'; use "
            "eval_tam_paper_5v4_generalization.py for 5v4 perturbations")


def validate_generalization_protocol(scenario: str, perturbation: str) -> None:
    if scenario != "5v4":
        raise ValueError("paper_5v4_generalization requires scenario='5v4'")
    if perturbation not in GENERALIZATION_PERTURBATION_LEVELS:
        raise ValueError(
            "paper_5v4_generalization perturbation must be low, medium, or large")


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
        "experiment_protocol": experiment_protocol,
        "scenario": scenario,
        "initial_perturbation": perturbation,
        "dynamics_backend": dynamics_backend,
        "paper_nominal_experiment": nominal,
        "paper_generalization_experiment": generalization,
        "paper_silent_assumptions_present": PAPER_SILENT_ASSUMPTIONS_PRESENT,
        "termination_resolution": TERMINATION_RESOLUTION,
        "neutral_action_semantics": "nearest_positive_center_not_exact_zero",
        "blue_policy_fidelity": BLUE_POLICY_FIDELITY,
        "reference_8_exact_blue_fsm_reproduced": (
            REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED),
    }


def checkpoint_lineage(metadata: dict) -> dict:
    """Return the stable checkpoint lineage fields used by result JSON files."""
    return {
        "checkpoint_environment_fidelity_revision": metadata.get(
            "environment_fidelity_revision"),
        "checkpoint_experiment_protocol": metadata.get("experiment_protocol"),
        "checkpoint_initial_perturbation": metadata.get("initial_perturbation"),
        "checkpoint_dynamics_backend": metadata.get("dynamics_backend"),
        "checkpoint_environment_steps": metadata.get("environment_steps"),
        "checkpoint_episodes": metadata.get("episodes"),
        "checkpoint_algorithm_mode": metadata.get("algorithm_mode"),
        "checkpoint_paper_silent_assumptions_present": metadata.get(
            "paper_silent_assumptions_present"),
        "checkpoint_blue_policy_fidelity": metadata.get("blue_policy_fidelity"),
        "checkpoint_reference_8_exact_blue_fsm_reproduced": metadata.get(
            "reference_8_exact_blue_fsm_reproduced"),
    }
