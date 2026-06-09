"""Test granular win/loss metrics in eval output. No training."""
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

def test_eval_has_breakdown_fields():
    out = "outputs/acmi/eval_metric_breakdown_smoke.json"
    r = subprocess.run([
        PYTHON, "scripts/eval_mappo_zero_shot.py",
        "--model", "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt",
        "--obs-adapter-version", "v2", "--episodes", "1", "--device", "cpu",
        "--opponent-policy", "rule_nearest",
        "--configs", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
        "--summary-json", out,
    ], cwd=ROOT, env=_env(), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, f"eval failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    d = json.loads((ROOT / out).read_text(encoding="utf-8"))
    for k in ["red_elimination_win_rate","blue_elimination_win_rate","red_timeout_alive_advantage_rate","blue_timeout_alive_advantage_rate","timeout_draw_rate","kill_death_ratio","red_dead_final_mean","blue_dead_final_mean"]:
        assert k in d[0], f"missing {k}"
