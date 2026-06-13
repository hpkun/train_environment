from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generate_progress_report_figures_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_progress_report_figures.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "progress report figures" in result.stdout.lower()


def test_generate_progress_report_figures_handles_missing_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_progress_report_figures.py",
            "--outputs-root",
            str(tmp_path / "missing_outputs"),
            "--output-dir",
            str(output_dir),
            "--no-show",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "generated" in result.stdout.lower()
    assert (output_dir / "figure_index.md").exists()
    assert (output_dir / "fig01_experiment_pipeline.png").exists()
    assert (output_dir / "fig01_experiment_pipeline.svg").exists()
