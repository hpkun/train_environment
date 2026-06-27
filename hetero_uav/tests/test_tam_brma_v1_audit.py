"""Lightweight tests for tam_brma_v1 audit script."""
import sys, subprocess, pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_audit_script_help():
    r = subprocess.run([sys.executable, "scripts/audit_tam_brma_v1_full_pipeline.py", "--help"],
                       cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "--checkpoint" in r.stdout
    assert "--episodes" in r.stdout

def test_mismatch_types_defined():
    from scripts.audit_tam_brma_v1_full_pipeline import _gate_reward_vs_real
    from scripts.audit_tam_brma_v1_full_pipeline import _launch_breakdown
    assert callable(_gate_reward_vs_real)
    assert callable(_launch_breakdown)

def test_launch_breakdown_empty():
    from scripts.audit_tam_brma_v1_full_pipeline import _launch_breakdown
    assert _launch_breakdown([]) == {}

def test_no_checkpoint_error():
    """Script gives clear error when checkpoint missing."""
    r = subprocess.run([sys.executable, "scripts/audit_tam_brma_v1_full_pipeline.py",
                        "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_brma_scripted_reward_v1.yaml",
                        "--checkpoint", "/nonexistent/path/model.pt",
                        "--output-dir", "outputs/_audit_test",
                        "--episodes", "1", "--max-steps", "100", "--device", "cpu"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0  # should fail with clear error, not silently
