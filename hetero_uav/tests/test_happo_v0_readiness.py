import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_happo_v0_readiness_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/check_happo_reference_v0_readiness.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "HAPPO reference v0 readiness" in result.stdout


def test_happo_v0_readiness_outputs_schema():
    out_json = "outputs/test_happo_readiness/happo_v0_readiness.json"
    out_md = "outputs/test_happo_readiness/happo_v0_readiness.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_happo_reference_v0_readiness.py",
            "--output-json",
            out_json,
            "--output-md",
            out_md,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=180,
    )
    assert "ready_for_200k" in result.stdout
    data = json.loads((ROOT / out_json).read_text(encoding="utf-8"))
    for key in ["ready_for_200k", "blocking_issues", "warnings", "next_action"]:
        assert key in data
    assert (ROOT / out_md).exists()
