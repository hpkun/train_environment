import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def test_consistency_audit_help_runs():
    result = _run(["scripts/audit_happo_train_eval_consistency.py", "--help"])
    assert result.returncode == 0
    assert "train/eval consistency" in result.stdout


def test_consistency_audit_missing_dir_exits_cleanly(tmp_path):
    result = _run(
        [
            "scripts/audit_happo_train_eval_consistency.py",
            "--output-dir",
            str(tmp_path / "missing"),
        ]
    )
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)


def test_consistency_audit_generates_json_for_existing_outputs():
    output_dir = ROOT / "outputs" / "happo_3v2_reference_200k"
    if not output_dir.exists():
        return
    out_json = ROOT / "outputs" / "test_happo_audit" / "consistency.json"
    out_md = ROOT / "outputs" / "test_happo_audit" / "consistency.md"
    result = _run(
        [
            "scripts/audit_happo_train_eval_consistency.py",
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    assert result.returncode == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["consistency_status"] in {"consistent", "inconsistent", "unknown"}
    assert "likely_causes" in data
    assert out_md.exists()
