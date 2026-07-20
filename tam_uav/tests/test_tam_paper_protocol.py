from __future__ import annotations

import pytest

from scripts.eval_tam_paper_5v4_generalization import parse_args as parse_generalization_args
from scripts.eval_tam_paper_env_v1 import parse_args as parse_nominal_env_args
from uav_env.JSBSim.paper.protocol import (
    BLUE_POLICY_FIDELITY, ENVIRONMENT_FIDELITY_REVISION,
    GENERALIZATION_EPISODES_PER_LEVEL,
    GENERALIZATION_PERTURBATION_LEVELS, PAPER_5V4_GENERALIZATION_PROTOCOL,
    PAPER_NOMINAL_PROTOCOL, REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED,
    validate_generalization_protocol,
    validate_nominal_protocol)


def test_protocol_constants_and_nominal_validation():
    assert ENVIRONMENT_FIDELITY_REVISION == "published_environment_reconstruction_v5"
    assert BLUE_POLICY_FIDELITY == "minimal_greedy_basic_manoeuvre_reconstruction"
    assert REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED is False
    assert PAPER_NOMINAL_PROTOCOL == "paper_nominal"
    for scenario in ("2v2", "3v2", "5v4"):
        validate_nominal_protocol(scenario, "none")
        with pytest.raises(ValueError, match="perturbation"):
            validate_nominal_protocol(scenario, "low")


def test_generalization_protocol_is_only_5v4_and_three_perturbations():
    assert PAPER_5V4_GENERALIZATION_PROTOCOL == "paper_5v4_generalization"
    assert GENERALIZATION_PERTURBATION_LEVELS == ("low", "medium", "large")
    for level in GENERALIZATION_PERTURBATION_LEVELS:
        validate_generalization_protocol("5v4", level)
    with pytest.raises(ValueError, match="5v4"):
        validate_generalization_protocol("3v2", "low")
    with pytest.raises(ValueError, match="low, medium, or large"):
        validate_generalization_protocol("5v4", "none")


def test_nominal_environment_cli_has_no_perturbation_argument():
    args = parse_nominal_env_args([])
    assert args.scenario == "3v2"
    assert args.episodes == 1
    assert not hasattr(args, "perturbation")
    with pytest.raises(SystemExit):
        parse_nominal_env_args(["--perturbation-levels", "low"])


def test_generalization_cli_has_fixed_protocol_defaults():
    args = parse_generalization_args(["--checkpoint", "outputs/test.pt"])
    assert args.episodes_per_level == GENERALIZATION_EPISODES_PER_LEVEL
    assert not hasattr(args, "scenario")
    assert not hasattr(args, "perturbation")
