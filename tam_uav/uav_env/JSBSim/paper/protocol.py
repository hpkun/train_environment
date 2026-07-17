"""Experiment protocol constants for the isolated TAM paper environment."""

from __future__ import annotations


ENVIRONMENT_FIDELITY_REVISION = "published_rules_simplified_v2"
NOMINAL_PERTURBATION = "none"
GENERALIZATION_PERTURBATION_LEVELS = ("low", "medium", "large")
GENERALIZATION_EPISODES_PER_LEVEL = 50
PAPER_NOMINAL_PROTOCOL = "paper_nominal"
PAPER_5V4_GENERALIZATION_PROTOCOL = "paper_5v4_generalization"
PAPER_SILENT_ASSUMPTIONS_PRESENT = True
SCENARIOS = ("2v2", "3v2", "5v4")


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
    }
