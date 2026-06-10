from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_gru_mappo_alignment.py"


def test_gru_mappo_alignment_audit_help_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "GRU MAPPO" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout


def test_gru_mappo_alignment_audit_outputs_json_and_markdown():
    out_json = ROOT / "outputs" / "test_protocol_audit" / "gru_mappo_alignment.json"
    out_md = ROOT / "outputs" / "test_protocol_audit" / "gru_mappo_alignment.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(out_json.relative_to(ROOT)),
            "--output-md",
            str(out_md.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "vanilla_actor_has_gru" in result.stdout
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "original_vanilla_gru_protocol" in data
    assert "original_attention_gru_protocol" in data
    assert "current_hetero_feedforward_protocol" in data
    assert "required_code_changes" in data
    assert "minimal_gru_plan" in data
    assert data["minimal_gru_plan"]["decision"] == "plan_only_this_round"

    md = out_md.read_text(encoding="utf-8")
    for token in ("GRU", "shared MLP", "attention", "hetero_uav", "reward"):
        assert token in md
