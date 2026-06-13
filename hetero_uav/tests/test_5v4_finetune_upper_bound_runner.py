from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_5v4_finetune_upper_bound_runner_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_5v4_finetune_upper_bound_50k.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = result.stdout
    assert "train_happo_reference.py" in stdout
    assert "evaluate_happo_3v2_reference_checkpoints.py" in stdout
    assert "hetero_mav_shared_geo_5v4_happo_ref_v0_f16_mav_surrogate.yaml" in stdout
    assert "happo_5v4_finetune_upper_bound_50k" in stdout
    assert "--total-env-steps 50000" in stdout
    assert "--configs 5v4_zero_shot" in stdout
