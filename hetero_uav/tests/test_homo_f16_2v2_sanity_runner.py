"""Test 2v2 sanity runner."""
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
def test_smoke():
    r=subprocess.run([PY,"scripts/smoke_main_mappo_homo_f16_2v2_sanity.py"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    out=ROOT/"outputs/test_main_mappo_homo_f16_2v2_sanity"
    meta=json.loads((out/"latest/meta.json").read_text(encoding="utf-8"))
    assert meta.get("actor_arch")=="mlp"
    assert meta.get("opponent_policy")=="brma_rule"
    assert (out/"eval_summary.json").exists()
