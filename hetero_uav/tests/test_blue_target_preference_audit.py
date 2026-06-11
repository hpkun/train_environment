import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_blue_target_preference_audit_help_runs():
    result = _run(["scripts/audit_blue_target_preference_against_mav.py", "--help"])
    assert result.returncode == 0
    assert "target preference" in result.stdout.lower()


def test_blue_target_preference_missing_checkpoint_exits_cleanly(tmp_path):
    result = _run([
        "scripts/audit_blue_target_preference_against_mav.py",
        "--model", str(tmp_path / "missing.pt"),
        "--episodes", "1",
    ])
    assert result.returncode != 0
    assert "checkpoint not found" in (result.stderr + result.stdout).lower()


def test_blue_target_preference_fast_schema(tmp_path):
    model = ROOT / "outputs" / "happo_3v2_reference_200k" / "best" / "model.pt"
    if not model.exists():
        return
    out_json = tmp_path / "blue_target.json"
    result = _run([
        "scripts/audit_blue_target_preference_against_mav.py",
        "--episodes", "1",
        "--max-steps-override", "3",
        "--output-json", str(out_json),
        "--output-md", str(tmp_path / "blue_target.md"),
    ])
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ["mav_target_fraction", "mav_missile_target_fraction", "unavailable_fields", "conclusion"]:
        assert key in data
