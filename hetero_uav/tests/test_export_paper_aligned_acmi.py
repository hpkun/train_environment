"""Test paper-aligned ACMI export."""
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
    r=subprocess.run([PY,"scripts/export_paper_aligned_acmi.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    model="outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"
    if not (ROOT/model).exists(): return
    r=subprocess.run([PY,"scripts/export_paper_aligned_acmi.py","--output-acmi","outputs/acmi/test_pa.acmi","--output-summary","outputs/acmi/test_pa_summary.json"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    s=json.loads((ROOT/"outputs/acmi/test_pa_summary.json").read_text(encoding="utf-8"))
    assert s.get("mav_alive") is not None
