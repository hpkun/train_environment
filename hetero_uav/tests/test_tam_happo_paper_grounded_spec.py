import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tam_happo_paper_grounded_spec_files_exist():
    assert (ROOT / "docs" / "tam_happo_paper_grounded_spec.md").exists()
    assert (
        ROOT / "outputs" / "protocol_audit" / "tam_happo_paper_grounded_spec.json"
    ).exists()
    assert (
        ROOT / "outputs" / "protocol_audit" / "tam_happo_paper_grounded_spec.md"
    ).exists()


def test_tam_happo_paper_grounded_spec_json_schema_and_boundaries():
    path = ROOT / "outputs" / "protocol_audit" / "tam_happo_paper_grounded_spec.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    for key in [
        "paper_scenario_setup",
        "paper_action_space",
        "paper_happo_algorithm",
        "paper_reward_modules",
        "current_env_gaps",
        "forbidden_claims",
        "allowed_reference_v0_scope",
    ]:
        assert key in data

    forbidden = "\n".join(data["forbidden_claims"])
    for phrase in [
        "full TAM-HAPPO reproduction",
        "paper action-space reproduction",
        "attention-enhanced value network",
        "temporal GRU module",
    ]:
        assert phrase in forbidden

    allowed = "\n".join(data["allowed_reference_v0_scope"])
    for phrase in [
        "high-level action retained",
        "scripted missile retained",
        "no attention in v0",
        "no GRU in v0 unless explicitly implemented later",
        "HAPPO-style sequential update if implemented",
    ]:
        assert phrase in allowed


def test_happo_validation_plan_contains_paper_grounded_limits():
    text = (ROOT / "docs" / "happo_3v2_reference_validation_plan.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "not a TAM-HAPPO reproduction",
        "no-temporal HAPPO ablation",
        "cannot claim an attention-enhanced value network",
        "not reproduce the paper action space",
        "scripted environment mechanics",
        "environment validation",
    ]:
        assert phrase in text
