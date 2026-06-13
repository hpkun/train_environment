from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_representative_scripts_help() -> None:
    for script in [
        "scripts/select_representative_acmi_episodes.py",
        "scripts/generate_representative_trajectory_figures.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "representative" in result.stdout.lower()


def test_representative_scripts_handle_missing_acmi(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/select_representative_acmi_episodes.py",
            "--acmi-dir",
            str(tmp_path / "missing_acmi"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    selection_path = output_dir / "representative_episode_selection.json"
    assert selection_path.exists()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["status"] in {"ok", "limited", "missing"}

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_representative_trajectory_figures.py",
            "--selection-json",
            str(selection_path),
            "--output-dir",
            str(output_dir),
            "--no-show",
        ],
        cwd=ROOT,
        check=True,
    )
    assert (output_dir / "fig07_trajectory_3v2_representative.png").exists()
    assert (output_dir / "fig08_trajectory_5v4_representative.png").exists()
