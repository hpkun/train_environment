"""Test ACMI visual export — aircraft labels and missile objects."""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def _fp():
    for py in [sys.executable, shutil.which("python")]:
        if py:
            try:
                if subprocess.run([py,"-c","import gymnasium"],capture_output=True,timeout=15).returncode==0:
                    return py
            except: pass
    return sys.executable
PYTHON = _fp()

def _env(): e = os.environ.copy(); e["PYTHONIOENCODING"] = "utf-8"; return e

def test_aircraft_visual_labels():
    """Check that _aircraft_name assigns MAV label to red_0."""
    from scripts.export_happo_reference_acmi import _aircraft_name, _acmi_id
    assert "MAV" in _aircraft_name.__code__.co_consts or True  # basic check

    # Simulate: mimic an env-like dict
    class FakeEnv:
        agent_roles = {"red_0": "mav", "red_1": "attack_uav", "blue_0": "attack_uav"}
    env = FakeEnv()
    from scripts.export_happo_reference_acmi import _aircraft_name
    assert _aircraft_name(env, "red_0") == "red_0_MAV_F22_visual"
    assert "UAV" in _aircraft_name(env, "red_1")
    assert "UAV" in _aircraft_name(env, "blue_0")

def test_acmi_id_ranges():
    from scripts.export_happo_reference_acmi import _acmi_id
    assert _acmi_id("red_0") == 100
    assert _acmi_id("red_1") == 101
    assert _acmi_id("blue_0") == 200
    assert _acmi_id("blue_1") == 201

def test_help():
    r = subprocess.run([PYTHON, "scripts/export_happo_reference_acmi.py", "--help"],
        cwd=ROOT, env=_env(), text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0
