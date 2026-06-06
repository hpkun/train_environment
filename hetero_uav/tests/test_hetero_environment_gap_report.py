from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_gap_report_from_mainline_v2_readiness():
    readiness_json = "outputs/test_environment_audit/paper_aligned_v2_readiness.json"
    gap_json = "outputs/test_environment_audit/gap_report.json"
    gap_md = "outputs/test_environment_audit/gap_report.md"

    audit = subprocess.run(
        [
            "python",
            "scripts/audit_hetero_environment_readiness.py",
            "--steps",
            "1",
            "--output-json",
            readiness_json,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert audit.returncode == 0, audit.stdout + audit.stderr

    report = subprocess.run(
        [
            "python",
            "scripts/report_hetero_environment_gaps.py",
            "--input-json",
            readiness_json,
            "--output-json",
            gap_json,
            "--output-md",
            gap_md,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert report.returncode == 0, report.stdout + report.stderr

    gap_json_path = ROOT / gap_json
    gap_md_path = ROOT / gap_md
    assert gap_json_path.exists()
    assert gap_md_path.exists()

    data = json.loads(gap_json_path.read_text(encoding="utf-8"))
    md = gap_md_path.read_text(encoding="utf-8").lower()
    combined = json.dumps(data).lower() + "\n" + md

    for token in [
        "paper-aligned",
        "3v2",
        "5v4",
        "mav_shared_geo",
        "blue greedy finite-state",
        "reward/termination",
        "not ready for method module",
    ]:
        assert token in combined

    assert (
        data["next_environment_task"]
        == "reward_termination_audit_review"
    )
    assert data["ready_for_method_module"] is False
    assert len(data["protocol_status"]) == 2
    assert all(item["exists"] for item in data["protocol_status"])
