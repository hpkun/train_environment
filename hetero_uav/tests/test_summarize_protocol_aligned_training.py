"""Test training summary."""
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
    r=subprocess.run([PY,"scripts/summarize_protocol_aligned_training.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_missing_dir_handled():
    r=subprocess.run([PY,"scripts/summarize_protocol_aligned_training.py","--experiment-dir","outputs/_nonexistent_200k"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0, f"should not crash: {r.stderr[-300:]}"
def test_smoke_dir_summarized():
    r=subprocess.run([PY,"scripts/summarize_protocol_aligned_training.py","--experiment-dir","outputs/test_main_mappo_protocol_aligned_experiment"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0,f"{r.stderr[-300:]}"
    d=json.loads((ROOT/"outputs/test_main_mappo_protocol_aligned_experiment/protocol_aligned_training_summary.json").read_text(encoding="utf-8"))
    assert "train" in d and "best_checkpoint" in d
