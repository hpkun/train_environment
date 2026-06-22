"""Smoke test for action stability envelope diagnostic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_stability_script_smoke():
    """Script must run minimal grid without crashing."""
    import subprocess
    result = subprocess.run([
        sys.executable, "-u",
        str(ROOT / "scripts" / "diagnose_tam_action_stability_envelope.py"),
        "--config", "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "--output-dir", str(ROOT / ".tmp" / "test_action_stability"),
        "--max-steps", "200",
        "--seeds", "1",
    ], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"Script failed: {result.stderr[:500]}"

    out_dir = ROOT / ".tmp" / "test_action_stability"
    assert (out_dir / "action_stability_envelope.md").exists(), "md not written"
    assert (out_dir / "action_stability_envelope.json").exists(), "json not written"
    assert (out_dir / "action_stability_envelope.csv").exists(), "csv not written"
