"""Test that the algorithm path document exists and has required content."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "algorithm_path_mappo_vs_happo.md"


def test_doc_exists():
    assert DOC.exists(), f"Missing {DOC}"


def test_doc_contains_mappo():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "mappo" in text


def test_doc_contains_happo():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "happo" in text


def test_doc_explains_why_start_with_mappo():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "why" in text and "start" in text and "mappo" in text


def test_doc_explains_when_to_move_beyond_mappo():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "beyond" in text.lower() or "move beyond" in text.lower()


def test_doc_contains_staged_roadmap():
    text = DOC.read_text(encoding="utf-8")
    assert "Stage" in text or "stage" in text
    assert "MAPPO" in text
    assert "HAPPO" in text
