from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "scripts" / "train_tam_happo_direct.py"
EVAL = ROOT / "scripts" / "eval_tam_happo_direct.py"


def _source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    return source


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tam_train_discovers_and_propagates_action_dim():
    source = _source(TRAIN)

    assert "def _action_dim_from_env(env)" in source
    assert "action_dim = _action_dim_from_env(env)" in source
    assert "action_dim=action_dim" in source
    assert "action_dim=3" not in source
    assert '"action_dim": action_dim' in source


def test_tam_eval_discovers_action_dim_and_rejects_checkpoint_mismatch():
    source = _source(EVAL)

    assert "def _action_dim_from_env(env)" in source
    assert "action_dim = _action_dim_from_env" in source
    assert "action_dim=action_dim" in source
    assert "action_dim=3" not in source
    assert "checkpoint action_dim" in source


def test_tam_entrypoints_do_not_enable_disallowed_training_features():
    source = _source(TRAIN)

    assert 'default="tam_direct_fsm"' in source
    assert "--brma-random-scale-mask" not in source
    assert "--brma-biased-mask" not in source
    assert "--uav-imitation-dataset" not in source


def test_fixed_action_audit_uses_required_actions():
    module = _load_script("audit_tam_direct_control_response")

    assert module.FIXED_ACTIONS == {
        "level": [0.65, 0.0, 0.0, 0.0],
        "throttle_high": [0.9, 0.0, 0.0, 0.0],
        "throttle_low": [0.4, 0.0, 0.0, 0.0],
        "climb_pos": [0.8, 0.0, 0.3, 0.0],
        "climb_neg": [0.8, 0.0, -0.3, 0.0],
        "roll_left": [0.75, -0.4, 0.0, 0.0],
        "roll_right": [0.75, 0.4, 0.0, 0.0],
        "rudder_left": [0.75, 0.0, 0.0, -0.4],
        "rudder_right": [0.75, 0.0, 0.0, 0.4],
    }


def test_training_curve_analyzer_reports_required_sections():
    module = _load_script("analyze_tam_training_curves")
    summary = module.summarize_rows(
        [
            {"avg_return": "1", "red_win": "0", "blue_win": "1", "timeout": "0"},
            {"avg_return": "3", "red_win": "1", "blue_win": "0", "timeout": "0"},
        ]
    )

    assert summary["avg_return"]["last"] == 3.0
    assert summary["outcomes"]["red_win_mean"] == 0.5
    assert set(summary) >= {"avg_return", "outcomes", "self_control", "weapon_chain", "evaluation"}
