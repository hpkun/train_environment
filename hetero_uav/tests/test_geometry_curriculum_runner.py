from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_curriculum_runner_dry_run_commands() -> None:
    medium_config = (
        ROOT
        / "uav_env"
        / "JSBSim"
        / "configs"
        / "hetero_mav_shared_geo_3v2_medium_combat_f16_mav_surrogate.yaml"
    )
    assert medium_config.exists()

    result = subprocess.run(
        [sys.executable, "scripts/run_happo_geometry_curriculum_100k.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = result.stdout
    assert "medium_50k" in stdout
    assert "normal_50k" in stdout
    assert "hetero_mav_shared_geo_3v2_medium_combat_f16_mav_surrogate.yaml" in stdout
    assert "hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml" in stdout
    assert "--eval-during-training" in stdout
    assert "--uav-imitation-coef" in stdout
