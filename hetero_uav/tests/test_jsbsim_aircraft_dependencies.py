from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scripts.check_jsbsim_aircraft_dependencies import (
    AIRCRAFT_ROOT,
    check_aircraft,
    collect_dependencies,
)


A4_XML = AIRCRAFT_ROOT / "A-4" / "A-4.xml"
F16_XML = AIRCRAFT_ROOT / "f16" / "f16.xml"


def _by_kind(deps, kind: str):
    return [dep for dep in deps if dep.kind == kind]


def test_aircraft_xml_files_exist():
    assert A4_XML.exists()
    assert F16_XML.exists()


def test_a4_dependencies_are_complete_and_systems_not_required():
    deps = collect_dependencies(A4_XML)
    assert [dep for dep in deps if not dep.exists] == []
    assert _by_kind(deps, "system") == []
    assert not (A4_XML.parent / "Systems").exists()


def test_f16_dependencies_are_complete_and_systems_required():
    deps = collect_dependencies(F16_XML)
    systems = _by_kind(deps, "system")
    system_values = {dep.value for dep in systems}
    assert [dep for dep in deps if not dep.exists] == []
    assert {"pushback", "hook"}.issubset(system_values)
    assert (F16_XML.parent / "Systems" / "pushback.xml").exists()
    assert (F16_XML.parent / "Systems" / "hook.xml").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)
def test_a4_load_model_and_run_ic_success():
    report = check_aircraft("A-4", A4_XML)
    assert report["missing_dependencies"] == []
    assert report["load_model_success"]
    assert report["run_ic_success"]


@pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)
def test_f16_load_model_and_run_ic_success():
    report = check_aircraft("f16", F16_XML)
    assert report["missing_dependencies"] == []
    assert report["load_model_success"]
    assert report["run_ic_success"]
