"""Test zero-shot experiment protocol doc exists and has required content."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "zero_shot_experiment_protocol.md"


def test_doc_exists():
    assert DOC.exists()


def test_doc_brma():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "brma" in text


def test_doc_tam_happo():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "tam-happo" in text or "tam happo" in text


def test_doc_smoke_not_zero_shot_success():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "not evidence" in text or "smoke" in text


def test_doc_metrics():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "return" in text or "metric" in text


def test_doc_stages():
    text = DOC.read_text(encoding="utf-8")
    assert "Stage" in text or "stage" in text or "E0" in text
