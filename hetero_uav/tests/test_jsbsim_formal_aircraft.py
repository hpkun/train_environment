from __future__ import annotations

import importlib.util

import pytest

from scripts.diagnose_jsbsim_formal_aircraft import run_scenario


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def test_f16_level_10_seconds_stable():
    row = run_scenario("f16", "level", 10.0)
    assert not row["crashed"]
    assert not row["nan_detected"]


def test_a4_level_10_seconds_stable():
    row = run_scenario("A-4", "level", 10.0)
    assert not row["crashed"]
    assert not row["nan_detected"]


def test_f16_turn_right_heading_increases():
    row = run_scenario("f16", "turn_right", 10.0)
    assert row["heading_delta"] > 0.0


def test_a4_turn_right_heading_increases():
    row = run_scenario("A-4", "turn_right", 10.0)
    assert row["heading_delta"] > 0.0


def test_f16_climb_finishes_above_level():
    level = run_scenario("f16", "level", 10.0)
    climb = run_scenario("f16", "climb", 10.0)
    assert climb["final_altitude"] > level["final_altitude"]


def test_a4_climb_finishes_above_level():
    level = run_scenario("A-4", "level", 10.0)
    climb = run_scenario("A-4", "climb", 10.0)
    assert climb["final_altitude"] > level["final_altitude"]
