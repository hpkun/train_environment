"""Test environment readiness audit."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _fp():
    for py in [sys.executable, shutil.which("python")]:
        if py:
            try:
                if subprocess.run([py,"-c","import gymnasium"],capture_output=True,timeout=15).returncode==0: return py
            except: pass
    return sys.executable
PY=_fp()
def _env(): e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; return e
def test_help():
    r=subprocess.run([PY,"scripts/audit_environment_readiness.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/audit_environment_readiness.py","--steps","20","--output-json","outputs/environment_audit/test_er.json","--output-md","outputs/environment_audit/test_er.md"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/environment_audit/test_er.json").read_text(encoding="utf-8"))
    for k in ["aircraft_stability","opponent_readiness","observation_readiness","reward_readiness","acmi_readiness","environment_ready_for_training"]:
        assert k in d, f"missing {k}"
