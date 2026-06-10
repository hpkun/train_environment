"""Test early-stop runner."""
from __future__ import annotations
import subprocess, sys, shutil, os
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
    r=subprocess.run([PY,"scripts/run_main_mappo_baseline_early_stop_fast.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_dry_run():
    r=subprocess.run([PY,"scripts/run_main_mappo_baseline_early_stop_fast.py","--dry-run"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
