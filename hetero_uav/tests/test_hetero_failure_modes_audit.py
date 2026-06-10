"""Test failure modes audit."""
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
    r=subprocess.run([PY,"scripts/audit_hetero_3v2_failure_modes.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/audit_hetero_3v2_failure_modes.py","--episodes","2","--steps","20","--output-json","outputs/environment_audit/test_fm.json","--output-md","outputs/environment_audit/test_fm.md"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/environment_audit/test_fm.json").read_text(encoding="utf-8"))
    for k in ["hetero_f22_mav","f16_mav_surrogate","homo_f16_2v2_reference"]:
        assert k in d["summaries"], f"missing {k}"
        s=d["summaries"][k]
        for sub in ["outcome","first_dead","red0_death_count","missiles"]:
            assert sub in s, f"{k} missing {sub}"
    assert "conclusions" in d
