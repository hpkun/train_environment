"""Verify diagnostic scripts can import and run without crashing."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_ppo_action_direction_imports():
    """Diagnostic script must be importable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diagnose_tam_ppo_action_direction",
        ROOT / "scripts" / "diagnose_tam_ppo_action_direction.py",
    )
    assert spec is not None, "diagnose_tam_ppo_action_direction.py not found"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "diagnostic_1_synthetic"), "diagnostic_1_synthetic missing"
    assert hasattr(mod, "diagnostic_1_synthetic"), "diagnostic_1_synthetic missing"


def test_action_stability_envelope_imports():
    """Diagnostic script must be importable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diagnose_tam_action_stability_envelope",
        ROOT / "scripts" / "diagnose_tam_action_stability_envelope.py",
    )
    assert spec is not None, "diagnose_tam_action_stability_envelope.py not found"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_run_single_action"), "_run_single_action missing"


def test_no_new_reward_loss_controller():
    """Verify no reward/loss/controller/action penalty changes were made."""
    import ast

    trainer_path = ROOT / "algorithms" / "happo" / "tam_categorical_happo_trainer.py"
    trainer_src = trainer_path.read_text(encoding="utf-8")

    # These must NOT appear as new additions
    forbidden_new = [
        "neutral_prior_loss", "action_penalty", "altitude_penalty",
        "scripted_controller", "scripted_evasion", "imitation_loss",
        "bc_loss", "behavioral_cloning", "heading_loss",
    ]
    for term in forbidden_new:
        assert term not in trainer_src.lower(), f"forbidden term '{term}' found in trainer"
