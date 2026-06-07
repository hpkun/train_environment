"""Tests for the MAPPO baseline network architecture audit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_network_architecture_audit_outputs_expected_fields(tmp_path):
    out_json = tmp_path / "network_audit.json"
    out_md = tmp_path / "network_audit.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_mappo_network_architecture.py",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out_json.read_text(encoding="utf-8"))

    current = data["current_architecture"]
    for key in [
        "actor_mlp",
        "critic_mlp",
        "action_dim",
        "no_attention",
        "no_gru",
        "no_happo",
        "baseline_only",
    ]:
        assert key in current
    assert current["actor_mlp"] == [96, 256, 128, 3]
    assert current["critic_mlp"] == [480, 256, 128, 1]
    assert current["action_dim"] == 3
    assert current["no_attention"] is True
    assert current["no_gru"] is True
    assert current["no_happo"] is True
    assert current["baseline_only"] is True


def test_network_architecture_doc_describes_scope():
    doc = ROOT / "docs/mappo_network_architecture_audit.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for token in [
        "MAPPO baseline",
        "BRMA-MAPPO",
        "TAM-HAPPO",
        "not the final proposed method",
    ]:
        assert token in text
