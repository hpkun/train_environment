"""Test hetero environment protocol review script. No training."""
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


def test_review_script_help_lists_expected_args():
    result = subprocess.run(
        [PYTHON, "scripts/review_hetero_environment_protocol.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ["--output-json", "--output-md", "--skip-env-reset"]:
        assert token in result.stdout, f"missing {token} in --help"


def test_review_script_produces_json_and_markdown():
    output_json = "outputs/test_environment_audit/protocol_review.json"
    output_md = "outputs/test_environment_audit/protocol_review.md"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/review_hetero_environment_protocol.py",
            "--output-json",
            output_json,
            "--output-md",
            output_md,
            "--skip-env-reset",
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr[-500:]}"

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists(), f"JSON missing: {json_path}"
    assert md_path.exists(), f"MD missing: {md_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "records" in data
    assert "summary" in data

    summary = data["summary"]
    assert summary["main_protocol_ready"] is True, (
        f"main_protocol_ready=False, failures={summary['blocking_failures']}"
    )
    assert summary["reward_overlay_ready"] is True, (
        f"reward_overlay_ready=False, failures={summary['blocking_failures']}"
    )
    assert summary["blocking_failures"] == [], (
        f"unexpected blocking failures: {summary['blocking_failures']}"
    )

    # Check main paper-aligned configs have correct protocol fields
    main_records = [
        r for r in data["records"]
        if r.get("protocol_group") == "main_paper_aligned"
    ]
    for rec in main_records:
        assert rec["hetero_reward_mode"] == "brma_legacy", (
            f"{rec['config']}: reward_mode={rec['hetero_reward_mode']}"
        )
        assert rec["observation_mode"] == "mav_shared_geo", (
            f"{rec['config']}: obs_mode={rec['observation_mode']}"
        )
        # actor_dim / critic_dim may be None with --skip-env-reset
        assert rec["max_steps"] >= 1000, (
            f"{rec['config']}: max_steps={rec['max_steps']}"
        )

    # Check optional reward overlay configs
    overlay_records = [
        r for r in data["records"]
        if r.get("protocol_group") == "optional_reward_overlay"
    ]
    for rec in overlay_records:
        assert rec["hetero_reward_mode"] == "minimal_v1", (
            f"{rec['config']}: reward_mode={rec['hetero_reward_mode']}"
        )
        assert any(
            "optional" in w.lower() or "not default" in w.lower()
            for w in rec.get("warnings", [])
        ), f"{rec['config']}: missing overlay warning"

    # Check Markdown content
    md = md_path.read_text(encoding="utf-8")
    for token in [
        "paper-aligned",
        "balanced",
        "brma_legacy",
        "minimal_v1",
        "greedy_fsm",
        "not a method module",
    ]:
        assert token in md, f"missing '{token}' in markdown"
