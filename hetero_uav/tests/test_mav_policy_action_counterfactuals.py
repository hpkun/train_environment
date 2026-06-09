"""Test counterfactual script."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _find_python():
    cs = [sys.executable, shutil.which("python")]
    for py in [c for c in cs if c]:
        try:
            if subprocess.run([py,"-c","import gymnasium"],capture_output=True,timeout=15).returncode==0:
                return py
        except: pass
    return sys.executable
PYTHON = _find_python()
def _env(): e = os.environ.copy(); e["PYTHONIOENCODING"] = "utf-8"; return e
def test_help():
    r = subprocess.run([PYTHON,"scripts/diagnose_mav_policy_action_counterfactuals.py","--help"], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode==0
def test_runs():
    r = subprocess.run([PYTHON,"scripts/diagnose_mav_policy_action_counterfactuals.py","--steps","10","--output-json","outputs/flight_audit/test_cf.json","--output-md","outputs/flight_audit/test_cf.md"], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode==0, f"failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    d = json.loads((ROOT/"outputs/flight_audit/test_cf.json").read_text(encoding="utf-8"))
    for k in ["current_policy","red0_zero_override","current_policy_no_trim","red0_clamped_policy"]:
        assert k in d, f"missing {k}"
