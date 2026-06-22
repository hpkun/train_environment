"""Test advantage signal diagnostic is read-only (no loss/reward modification)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_advantage_signal_smoke():
    """Script must run and output md/json."""
    import subprocess
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/diagnose_tam_advantage_signal.py"),
        "--output-dir", str(ROOT / ".tmp/test_advantage_signal"),
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Diagnostic failed: {result.stderr[:500]}"
    out = ROOT / ".tmp/test_advantage_signal"
    assert (out / "advantage_signal.md").exists()
    assert (out / "advantage_signal.json").exists()
    report = json.loads((out / "advantage_signal.json").read_text(encoding="utf-8"))
    assert report["mode"] == "read_only_diagnostic"


def test_no_new_reward_loss_added():
    """Verify no reward/loss penalty added."""
    import ast
    paths = [
        ROOT / "scripts/diagnose_tam_advantage_signal.py",
        ROOT / "scripts/prepare_tam_paper_run_manifest.py",
        ROOT / "scripts/audit_tam_paper_readiness.py",
    ]
    forbidden = ["r_mav_survival", "action_penalty", "altitude_penalty",
                 "neutral_prior_loss", "imitation_loss", "bc_loss"]
    for pp in paths:
        src = pp.read_text(encoding="utf-8").lower()
        for term in forbidden:
            assert term not in src, f"{term} found in {pp.name}"
