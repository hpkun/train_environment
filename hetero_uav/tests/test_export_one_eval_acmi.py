"""Test ACMI export script. No training."""
from __future__ import annotations
import os, shutil, subprocess, sys
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

def test_help():
    r = subprocess.run([PYTHON,"scripts/export_one_eval_acmi.py","--help"], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode==0, r.stdout+r.stderr

def test_acmi_export():
    model = "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"
    if not (ROOT / model).exists():
        return  # skip if model not found
    acmi = "outputs/acmi/test_alive_done_fix_3v2.acmi"
    smry = "outputs/acmi/test_alive_done_fix_3v2_summary.json"
    r = subprocess.run([
        PYTHON, "scripts/export_one_eval_acmi.py",
        "--model", model,
        "--output-acmi", acmi,
        "--output-summary", smry,
    ], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode==0, f"export failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    t = (ROOT / acmi).read_text(encoding="utf-8")
    for token in ["FileType=text/acmi/tacview","FileVersion=2.2","ReferenceTime","Type=Air+FixedWing","#0","T="]:
        assert token in t, f"missing {token}"
