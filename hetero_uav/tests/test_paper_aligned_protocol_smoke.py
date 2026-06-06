"""Test paper-aligned protocol smoke runner. No training claim."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    """Find a Python executable that can import gymnasium."""
    candidates = [sys.executable]
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import gymnasium"],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


PYTHON = _find_python()


def test_smoke_script_help_lists_expected_args():
    result = subprocess.run(
        [PYTHON, "scripts/smoke_paper_aligned_protocol.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in [
        "--total-env-steps",
        "--opponent-policies",
        "--eval-episodes",
        "--output-dir",
    ]:
        assert token in result.stdout, f"missing {token} in --help"


def test_smoke_rule_nearest_short_run():
    output_dir = "outputs/test_paper_aligned_protocol_smoke"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/smoke_paper_aligned_protocol.py",
            "--total-env-steps", "64",
            "--rollout-length", "16",
            "--max-steps", "64",
            "--eval-episodes", "1",
            "--opponent-policies", "rule_nearest",
            "--device", "cpu",
            "--output-dir", output_dir,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, (
        f"smoke failed:\nstdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )

    # Check outputs
    json_path = ROOT / output_dir / "protocol_smoke_summary.json"
    csv_path = ROOT / output_dir / "protocol_smoke_summary.csv"
    assert json_path.exists(), f"missing {json_path}"
    assert csv_path.exists(), f"missing {csv_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) > 0, "empty summary"

    for rec in data:
        assert rec["status"] == "passed", f"record not passed: {rec}"
        assert rec["actor_dim"] == 96, f"actor_dim={rec['actor_dim']}"
        assert rec["critic_dim"] == 480, f"critic_dim={rec['critic_dim']}"
        assert rec["eval_nan_detected"] is False, f"eval nan: {rec['eval_nan_detected']}"
        for ck in [
            "red_win_rate", "blue_win_rate", "draw_rate",
            "timeout_rate", "mav_survival_rate",
        ]:
            assert ck in rec, f"missing combat metric {ck}"
        assert rec["actor_dim_ok"] is True
        assert rec["critic_dim_ok"] is True


def test_smoke_doc_exists_and_has_required_content():
    doc_path = ROOT / "docs" / "paper_aligned_protocol_smoke.md"
    assert doc_path.exists(), f"missing doc: {doc_path}"
    text = doc_path.read_text(encoding="utf-8")
    for token in [
        "paper-aligned",
        "brma_legacy",
        "mav_shared_geo",
        "not a zero-shot claim",
        "not a method module",
    ]:
        assert token in text, f"missing '{token}' in doc"
