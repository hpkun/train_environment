"""Test blue detection script."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _fp():
    cs = [sys.executable, shutil.which("python")]
    for py in [c for c in cs if c]:
        try:
            if subprocess.run([py,"-c","import gymnasium"],capture_output=True,timeout=15).returncode==0:
                return py
        except: pass
    return sys.executable
PY=_fp()
def _env(): e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; return e
def test_help():
    r=subprocess.run([PY,"scripts/diagnose_blue_detection_and_response.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/diagnose_blue_detection_and_response.py","--steps","10","--output-json","outputs/flight_audit/test_bdr.json","--output-md","outputs/flight_audit/test_bdr.md"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/flight_audit/test_bdr.json").read_text(encoding="utf-8"))
    for k in ["rule_nearest","greedy_fsm"]:
        assert k in d; b0=d[k]["blue_0"]; assert "first_detection_step" in b0; assert "action_change_after_detection" in b0; assert "heading_command_matches_target_bearing" in b0
