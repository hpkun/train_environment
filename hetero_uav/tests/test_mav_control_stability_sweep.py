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


def test_mav_control_stability_sweep_help_runs():
    result = _run(["scripts/audit_mav_control_stability_sweep.py", "--help"])
    assert result.returncode == 0
    assert "control stability" in result.stdout.lower()


def test_mav_control_stability_sweep_fast_schema(tmp_path):
    out_json = tmp_path / "sweep.json"
    out_md = tmp_path / "sweep.md"
    result = _run([
        "scripts/audit_mav_control_stability_sweep.py",
        "--episodes", "1",
        "--steps", "3",
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert {"cases", "summary", "recommendations"}.issubset(data)
    assert data["cases"]
    assert out_md.exists()
