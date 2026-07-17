from __future__ import annotations

import pytest

from scripts.check_tam_paper_runtime_initial_states import check_scenario


@pytest.mark.parametrize("scenario", ("2v2", "3v2", "5v4"))
def test_real_jsbsim_runtime_initial_state_matches_table_configuration(scenario):
    result = check_scenario(scenario, seed=2026)
    assert result["within_tolerance"]
    assert len(result["agents"]) == sum(map(int, scenario.split("v")))
    for row in result["agents"]:
        assert row["alive"]
        assert row["missile_count"] == (0 if row["role"] == "mav" else 2)
        assert row["side"] in {"red", "blue"}
    assert all(value >= 0.0 for value in result["maximum_absolute_error"].values())
