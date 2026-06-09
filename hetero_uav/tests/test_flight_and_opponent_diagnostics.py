"""Test flight/opponent diagnostic scripts. No training."""
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

def _env():
    e = os.environ.copy(); e["PYTHONIOENCODING"] = "utf-8"; return e

def test_diagnose_help():
    r = subprocess.run([PYTHON,"scripts/diagnose_flight_and_opponent_behavior.py","--help"], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr

def test_diagnose_runs():
    r = subprocess.run([PYTHON,"scripts/diagnose_flight_and_opponent_behavior.py","--steps","20",
        "--output-json","outputs/flight_audit/test_flight_diag.json",
        "--output-md","outputs/flight_audit/test_flight_diag.md"],
        cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, f"diag failed: stdout={r.stdout[-500:]} stderr={r.stderr[-500:]}"
    d = json.loads((ROOT / "outputs/flight_audit/test_flight_diag.json").read_text(encoding="utf-8"))
    for k in ["mav_zero_action","f16_zero_action","trained_policy_red0","blue_rule_nearest","high_priority_issues"]:
        assert k in d, f"missing {k}"

def test_debug_acmi():
    model = "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"
    if not (ROOT / model).exists(): return
    r = subprocess.run([PYTHON,"scripts/export_debug_acmi.py",
        "--output-acmi","outputs/acmi/test_debug.acmi",
        "--output-summary","outputs/acmi/test_debug_summary.json"],
        cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, f"debug acmi failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    t = (ROOT / "outputs/acmi/test_debug.acmi").read_text(encoding="utf-8")
    for token in ["FileType=text/acmi/tacview","FileVersion=2.2","T=","Type=Air+FixedWing"]:
        assert token in t, f"missing {token}"
    s = json.loads((ROOT / "outputs/acmi/test_debug_summary.json").read_text(encoding="utf-8"))
    assert "red0_first50" in s
