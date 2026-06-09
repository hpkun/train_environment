"""Test posture reward audit script."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _fp():
    cs = [sys.executable, shutil.which("python")]
    for py in [c for c in cs if c]:
        try:
            if subprocess.run([py,"-c","import gymnasium"],capture_output=True,timeout=15).returncode==0:
                return py
        except: pass
    return sys.executable
PY=_fp()
def _env(): e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; return e
def test_help():
    r=subprocess.run([PY,"scripts/audit_posture_reward_and_mav_actions.py","--help"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=60)
    assert r.returncode==0
def test_runs():
    r=subprocess.run([PY,"scripts/audit_posture_reward_and_mav_actions.py","--steps","20","--output-json","outputs/reward_audit/test_pr.json","--output-md","outputs/reward_audit/test_pr.md"],cwd=ROOT,env=_env(),text=True,capture_output=True,encoding="utf-8",errors="replace",timeout=600)
    assert r.returncode==0,f"{r.stdout[-500:]} {r.stderr[-500:]}"
    d=json.loads((ROOT/"outputs/reward_audit/test_pr.json").read_text(encoding="utf-8"))
    for k in ["reward_component_audit","red0_action_reward_trace","paper_original_project_alignment","conclusions"]:
        assert k in d, f"missing {k}"
