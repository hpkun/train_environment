from __future__ import annotations

from pathlib import Path


DOC = Path(__file__).resolve().parents[1] / "docs" / "mav_shared_geo_observation_design.md"


def test_design_doc_exists():
    assert DOC.exists()


def test_design_doc_contains_source_priority():
    text = DOC.read_text(encoding="utf-8")
    assert "direct observation > MAV shared observation > unavailable" in text


def test_design_doc_deemphasizes_brma_sensor_for_main_actor_obs():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "does not use brma" in text
    assert "radar/rcs/fov" in text


def test_design_doc_says_mechanisms_unchanged():
    text = DOC.read_text(encoding="utf-8").lower()
    for word in ["action", "missile", "reward", "termination"]:
        assert word in text
    assert "does not change" in text


def test_design_doc_contains_v2_dims():
    text = DOC.read_text(encoding="utf-8")
    assert "flat_actor_obs_dim = 12 + 4*9 + 4*7 + 20 = 96" in text
    assert "critic_state_dim = 96 * 5 = 480" in text
