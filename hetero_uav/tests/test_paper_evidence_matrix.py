from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_paper_evidence_matrix_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_paper_evidence_matrix.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "paper evidence" in result.stdout.lower()


def test_build_paper_evidence_matrix_handles_missing_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_evidence_matrix"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_paper_evidence_matrix.py",
            "--outputs-root",
            str(tmp_path / "missing_outputs"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    matrix = json.loads((output_dir / "paper_evidence_matrix.json").read_text(encoding="utf-8"))
    transfer = json.loads((output_dir / "transfer_quality.json").read_text(encoding="utf-8"))
    assert matrix["records"]
    assert any(record["evidence_level"] == "missing" for record in matrix["records"])
    for value in transfer["geometry_curriculum_full_method"].values():
        assert value is None or isinstance(value, (int, float, str, bool))
