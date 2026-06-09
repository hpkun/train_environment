"""Test protocol alignment audit."""
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
    r=subprocess.run([PY,"scripts/audit_original_training_protocol_alignment.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/audit_original_training_protocol_alignment.py","--output-json","outputs/protocol_audit/test_pa.json","--output-md","outputs/protocol_audit/test_pa.md"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/protocol_audit/test_pa.json").read_text(encoding="utf-8"))
    for k in ["original_project_protocol","hetero_uav_paper_aligned_protocol","differences","likely_failure_causes","minimal_next_options"]:
        assert k in d, f"missing {k}"
