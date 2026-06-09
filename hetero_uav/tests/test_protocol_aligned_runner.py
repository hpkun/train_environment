"""Test protocol-aligned runner."""
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
def test_smoke_runs():
    r=subprocess.run([PY,"scripts/smoke_main_mappo_protocol_aligned_experiment.py"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"smoke failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    out=ROOT/"outputs/test_main_mappo_protocol_aligned_experiment"
    meta=json.loads((out/"latest/meta.json").read_text(encoding="utf-8"))
    assert meta.get("opponent_policy")=="brma_rule"
    assert meta.get("actor_arch")=="mlp"
    assert meta.get("ppo_epochs")==10, f"ppo_epochs={meta.get('ppo_epochs')}"
    assert meta.get("entropy_coef")==0.05, f"entropy={meta.get('entropy_coef')}"
    assert meta.get("rollout_length")==16
    assert (out/"eval_summary.json").exists()
