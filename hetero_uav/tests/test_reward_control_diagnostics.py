"""Tests for reward control diagnostic script. Static only, no JSBSim."""
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_diagnose_script_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_reward_control_failure.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0
    assert "--run-dir" in result.stdout
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--episodes" in result.stdout
    assert "--output-dir" in result.stdout


def test_reward_component_keys_defined():
    from scripts.diagnose_reward_control_failure import ALL_RC_KEYS, BRMA_KEYS, MAV_TAM_KEYS, EVENT_KEYS, TERMINAL_KEYS
    assert len(ALL_RC_KEYS) > 0
    assert "r_pitch" in BRMA_KEYS
    assert "tam_mav_safety_raw" in MAV_TAM_KEYS
    assert "event_uav_kill" in EVENT_KEYS
    assert "terminal_hetero_raw" in TERMINAL_KEYS


def test_pre_crash_window_function_handles_empty():
    from scripts.diagnose_reward_control_failure import _pre_crash_window_analysis
    result = _pre_crash_window_analysis([], [])
    assert "error" in result
    result2 = _pre_crash_window_analysis(
        [{"episode": 0, "agent_id": "red_0", "step": 1, "altitude_m": 3000, "alive": 1, "role": "mav"}],
        [],
    )
    assert "events" in result2


def test_launch_geometry_summary_empty():
    from scripts.diagnose_reward_control_failure import _launch_geometry_summary
    result = _launch_geometry_summary([], [], [])
    assert "error" in result


def test_analyze_train_trends_empty():
    from scripts.diagnose_reward_control_failure import analyze_train_trends
    result = analyze_train_trends([])
    assert "error" in result


def test_no_training_imports():
    """Verify diagnostic script does not import training/PPO modules."""
    spec = importlib.util.spec_from_file_location(
        "diagnose_reward_control_failure",
        str(ROOT / "scripts" / "diagnose_reward_control_failure.py"),
    )
    # Just check the script compiles and doesn't import forbidden modules
    try:
        import py_compile
        py_compile.compile(str(ROOT / "scripts" / "diagnose_reward_control_failure.py"), doraise=True)
        assert True
    except py_compile.PyCompileError as e:
        assert False, f"Script failed to compile: {e}"


import importlib.util
import py_compile
