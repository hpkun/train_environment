"""Test 1M checkpoint eval."""
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
    r=subprocess.run([PY,"scripts/evaluate_baseline_1m_checkpoints.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_missing_dir():
    r=subprocess.run([PY,"scripts/evaluate_baseline_1m_checkpoints.py","--experiment-dir","outputs/_nonexistent_1m"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode!=0 or "error" in (r.stdout+r.stderr).lower() or "not found" in (r.stdout+r.stderr).lower()
def test_smoke():
    r=subprocess.run([PY,"scripts/evaluate_baseline_1m_checkpoints.py","--episodes","1","--experiment-dir","outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim/checkpoint_eval/baseline_1m_checkpoint_eval.json").read_text(encoding="utf-8"))
    assert "latest" in d and "best" in d
