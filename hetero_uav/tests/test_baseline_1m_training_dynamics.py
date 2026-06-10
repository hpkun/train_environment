"""Test training dynamics."""
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
    r=subprocess.run([PY,"scripts/summarize_baseline_1m_training_dynamics.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/summarize_baseline_1m_training_dynamics.py","--experiment-dir","outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0,f"{r.stderr[-300:]}"
    d=json.loads((ROOT/"outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim/baseline_1m_training_dynamics.json").read_text(encoding="utf-8"))
    for k in ["learning_window","collapse_detection","best_eval","recommendations"]:
        assert k in d, f"missing {k}"
