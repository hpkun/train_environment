import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_happo_reference_smoke_runner():
    out_dir = ROOT / "outputs" / "test_happo_3v2_reference"
    subprocess.run(
        [
            sys.executable,
            "scripts/smoke_happo_3v2_reference.py",
            "--output-dir",
            str(out_dir.relative_to(ROOT)),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=240,
    )
    meta_path = out_dir / "latest" / "meta.json"
    summary_path = out_dir / "eval_summary.json"
    assert meta_path.exists()
    assert summary_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["algorithm"] == "happo_reference_v0"
    assert meta["separate_actors"] is True
    assert meta["centralized_critic"] is True
    assert meta["sequential_update"] is True
    assert meta["reward_mode"] == "happo_ref_v0"
    assert meta["opponent_policy"] == "brma_rule"
