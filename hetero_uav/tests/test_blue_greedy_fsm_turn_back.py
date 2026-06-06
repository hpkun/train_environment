from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_controlled_turn_back_wrap_cases(tmp_path):
    out = tmp_path / "greedy_fsm_controlled_branches.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_greedy_fsm_controlled_branches.py",
            "--output-json",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["all_passed"], result.stdout + result.stderr
    by_case = {record["case"]: record for record in data["records"]}
    assert by_case["heading_wrap_positive_case"]["action"][1] < 0.0
    assert by_case["heading_wrap_negative_case"]["action"][1] > 0.0
    assert by_case["search_acquire_wrap_case"]["action"][1] < 0.0


def test_turn_back_diagnostic_doc_exists():
    doc = ROOT / "docs" / "blue_greedy_fsm_turn_back_diagnostic.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8").lower()
    assert "wrap" in text
    assert "not clip" in text
    assert "candidate maneuver" in text
