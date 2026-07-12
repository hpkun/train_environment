from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.experiment_logging_schema import (
    EPISODE_REWARD_COMPONENTS_COLUMNS,
    FILE_SCHEMAS,
    REWARD_COMPONENT_COLUMNS,
    TRAIN_METRICS_COLUMNS,
)
from uav_env.JSBSim.envs.role_situation_v3 import (
    V3_EFFECTIVE_FIELDS,
    V3_EPISODE_FIELDS,
    V3_REWARD_COMPONENT_FIELDS,
    accumulate_v3_episode_step,
    aggregate_v3_effective_step,
)


def _component(total, *, mav=False, uav=False, identity=0.0, j=0.2):
    comp = {key: 0.0 for key in V3_REWARD_COMPONENT_FIELDS}
    comp.update({
        "reward_contract_revision": 5.0,
        "role_situation_v3_task_attrition": 1.0,
        "role_situation_v3_common": 1.0,
        "role_situation_v3_j_combat": j,
        "role_situation_v3_uav_local_offense_raw": 0.4 if uav else 0.0,
        "role_situation_v3_uav_local_threat_raw": 0.1 if uav else 0.0,
        "role_situation_v3_team_coverage_raw": 0.3 if uav else 0.0,
        "role_situation_v3_team_exposure_raw": 0.2 if uav else 0.0,
        "role_situation_v3_uav_situation_raw": 0.2 if uav else 0.0,
        "role_situation_v3_uav_situation_scaled": 0.01 if uav else 0.0,
        "role_situation_v3_uav_situation_encoded": 0.015 if uav else 0.0,
        "role_situation_v3_mav_marginal_information_raw": 0.5 if mav else 0.0,
        "role_situation_v3_mav_support_position_raw": 0.4 if mav else 0.0,
        "role_situation_v3_mav_threat_raw": 0.1 if mav else 0.0,
        "role_situation_v3_mav_role_raw": 0.2 if mav else 0.0,
        "role_situation_v3_mav_role_scaled": 0.01 if mav else 0.0,
        "role_situation_v3_mav_role_encoded": 0.03 if mav else 0.0,
        "role_situation_v3_role_encoded": 0.03 if mav else 0.015,
        "role_situation_v3_flight_encoded": -0.005,
        "role_situation_v3_is_mav": float(mav),
        "role_situation_v3_is_attack_uav": float(uav),
    })
    expected = comp["role_situation_v3_common"] + comp["role_situation_v3_role_encoded"] + comp["role_situation_v3_flight_encoded"]
    comp["role_situation_v3_total"] = expected
    comp["role_situation_v3_component_sum"] = expected
    comp["role_situation_v3_identity_error"] = identity
    return comp


def test_v3_effective_aggregation_uses_role_denominators_and_alive_before():
    components = {
        "red_0": _component(0, mav=True),
        "red_1": _component(0, uav=True),
        "red_2": _component(0, uav=True),
    }
    roles = {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"}
    values = aggregate_v3_effective_step(components, roles, np.array([1, 1, 0]), list(roles), step=7)
    assert values["effective_role_situation_v3_mav_marginal_information"] == 0.5
    assert values["effective_role_situation_v3_uav_local_offense"] == 0.4
    assert values["effective_role_situation_v3_common"] == 1.0
    assert all(np.isfinite(values[key]) for key in V3_EFFECTIVE_FIELDS)


def test_v3_effective_role_means_are_scale_consistent_3v2_to_5v4():
    roles3 = {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"}
    comp3 = {"red_0": _component(0, mav=True), "red_1": _component(0, uav=True), "red_2": _component(0, uav=True)}
    values3 = aggregate_v3_effective_step(comp3, roles3, np.ones(3), list(roles3), step=1)
    roles5 = {"red_0": "mav", **{f"red_{i}": "attack_uav" for i in range(1, 5)}}
    comp5 = {"red_0": _component(0, mav=True), **{f"red_{i}": _component(0, uav=True) for i in range(1, 5)}}
    values5 = aggregate_v3_effective_step(comp5, roles5, np.ones(5), list(roles5), step=1)
    for key in (
        "effective_role_situation_v3_uav_local_offense",
        "effective_role_situation_v3_uav_local_threat",
        "effective_role_situation_v3_team_coverage",
        "effective_role_situation_v3_team_exposure",
        "effective_role_situation_v3_mav_role_raw",
    ):
        assert values5[key] == values3[key]


def test_v3_episode_accumulator_last_and_max_semantics_and_dead_before():
    acc = {}
    first = _component(0, uav=True, j=0.2)
    second = _component(0, uav=True, j=0.7)
    accumulate_v3_episode_step(acc, first, agent_id="red_1", role="attack_uav", alive_before=True, step=1)
    accumulate_v3_episode_step(acc, second, agent_id="red_1", role="attack_uav", alive_before=True, step=2)
    snapshot = dict(acc)
    accumulate_v3_episode_step(acc, second, agent_id="red_1", role="attack_uav", alive_before=False, step=3)
    assert acc == snapshot
    assert acc["episode_role_situation_v3_task_attrition_sum"] == 2.0
    assert acc["episode_role_situation_v3_final_j_combat"] == 0.7
    assert acc["episode_role_situation_v3_max_abs_identity_error"] == 0.0
    assert set(V3_EPISODE_FIELDS) <= set(acc)


def test_logging_schemas_are_unique_and_contain_v3_contracts():
    for columns in FILE_SCHEMAS.values():
        assert len(columns) == len(set(columns))
    assert set(V3_EFFECTIVE_FIELDS) <= set(TRAIN_METRICS_COLUMNS)
    assert set(V3_REWARD_COMPONENT_FIELDS) <= set(REWARD_COMPONENT_COLUMNS)
    assert set(V3_EPISODE_FIELDS) <= set(EPISODE_REWARD_COMPONENTS_COLUMNS)


def validate_v3_csv(path: Path, required):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert set(required) <= set(reader.fieldnames)
        rows = list(reader)
    assert rows
    for row in rows:
        for key in required:
            assert row[key] != ""
            assert np.isfinite(float(row[key]))
    return rows
