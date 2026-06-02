from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_ROOT = ROOT / "uav_env" / "JSBSim" / "data"
AIRCRAFT_ROOT = DATA_ROOT / "aircraft"
ENGINE_ROOT = DATA_ROOT / "engine"


@dataclass(frozen=True)
class Dependency:
    kind: str
    value: str
    candidates: tuple[Path, ...]

    @property
    def exists(self) -> bool:
        return any(path.exists() for path in self.candidates)

    @property
    def resolved(self) -> Path | None:
        for path in self.candidates:
            if path.exists():
                return path
        return None


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _with_xml_suffix(path: Path) -> Path:
    return path if path.suffix.lower() == ".xml" else path.with_suffix(".xml")


def _file_attr(element: ET.Element) -> str | None:
    value = element.attrib.get("file")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _dependency_candidates(kind: str, value: str, aircraft_dir: Path) -> tuple[Path, ...]:
    raw = Path(value)
    if kind in {"engine", "thruster"}:
        return (_with_xml_suffix(ENGINE_ROOT / raw),)
    if kind == "system":
        return (
            _with_xml_suffix(aircraft_dir / "Systems" / raw),
            _with_xml_suffix(aircraft_dir / raw),
        )
    if kind == "include":
        return (
            _with_xml_suffix(aircraft_dir / raw),
            _with_xml_suffix(DATA_ROOT / raw),
        )
    return (_with_xml_suffix(aircraft_dir / raw), _with_xml_suffix(DATA_ROOT / raw))


def collect_dependencies(xml_path: Path) -> list[Dependency]:
    xml_path = Path(xml_path)
    aircraft_dir = xml_path.parent
    root = ET.parse(xml_path).getroot()
    dependencies: list[Dependency] = []
    for element in root.iter():
        tag = _tag_name(element)
        value = _file_attr(element)
        if value is None:
            continue
        if tag in {"engine", "thruster", "system", "include"}:
            dependencies.append(
                Dependency(tag, value, _dependency_candidates(tag, value, aircraft_dir))
            )
    return dependencies


def check_jsbsim_load_model(model_name: str) -> tuple[bool, bool, str]:
    try:
        import jsbsim

        fdm = jsbsim.FGFDMExec(str(DATA_ROOT))
        fdm.set_debug_level(0)
        load_ok = bool(fdm.load_model(model_name))
        run_ic_ok = bool(fdm.run_ic()) if load_ok else False
        return load_ok, run_ic_ok, ""
    except Exception as exc:  # pragma: no cover - exercised by diagnostic CLI
        return False, False, f"{type(exc).__name__}: {exc}"


def check_aircraft(model_name: str, xml_path: Path) -> dict:
    xml_path = Path(xml_path)
    aircraft_dir = xml_path.parent
    dependencies = collect_dependencies(xml_path)
    missing = [dep for dep in dependencies if not dep.exists]
    load_ok, run_ic_ok, error = check_jsbsim_load_model(model_name)
    return {
        "model_name": model_name,
        "xml_path": xml_path,
        "aircraft_dir": aircraft_dir,
        "systems_dir": aircraft_dir / "Systems",
        "has_systems_dir": (aircraft_dir / "Systems").is_dir(),
        "dependencies": dependencies,
        "missing_dependencies": missing,
        "load_model_success": load_ok,
        "run_ic_success": run_ic_ok,
        "jsbsim_error": error,
    }


def _print_dependency_group(label: str, deps: list[Dependency]) -> None:
    print(f"{label}:")
    if not deps:
        print("  []")
        return
    for dep in deps:
        resolved = dep.resolved
        status = "OK" if resolved is not None else "MISSING"
        path_text = str(resolved) if resolved is not None else " | ".join(str(p) for p in dep.candidates)
        print(f"  - {dep.value} -> {path_text} [{status}]")


def print_aircraft_report(report: dict) -> None:
    deps = report["dependencies"]
    by_kind = {
        kind: [dep for dep in deps if dep.kind == kind]
        for kind in ("engine", "thruster", "system", "include")
    }
    print(f"{report['model_name']}:")
    print(f"- xml path: {report['xml_path']}")
    print(f"- aircraft directory: {report['aircraft_dir']}")
    print(f"- has Systems directory: {str(report['has_systems_dir']).lower()}")
    _print_dependency_group("- engine references", by_kind["engine"])
    _print_dependency_group("- thruster references", by_kind["thruster"])
    _print_dependency_group("- system references", by_kind["system"])
    _print_dependency_group("- include references", by_kind["include"])
    if by_kind["system"]:
        print("- Systems directory is required by declared <system file=...> references.")
    else:
        print("- No system XML dependencies declared; Systems directory is not required.")
    _print_dependency_group("- missing dependencies", report["missing_dependencies"])
    print(f"- load_model success: {str(report['load_model_success']).lower()}")
    print(f"- run_ic success: {str(report['run_ic_success']).lower()}")
    if report["jsbsim_error"]:
        print(f"- JSBSim error: {report['jsbsim_error']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    reports = [
        check_aircraft("A-4", AIRCRAFT_ROOT / "A-4" / "A-4.xml"),
        check_aircraft("f16", AIRCRAFT_ROOT / "f16" / "f16.xml"),
    ]
    print(f"data_root: {DATA_ROOT}")
    print()
    for report in reports:
        print_aircraft_report(report)
    if any(report["missing_dependencies"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
