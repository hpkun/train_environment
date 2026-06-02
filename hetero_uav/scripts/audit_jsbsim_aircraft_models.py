from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_ROOT = ROOT / "uav_env" / "JSBSim" / "data"
AIRCRAFT = {
    "A-4": DATA_ROOT / "aircraft" / "A-4" / "A-4.xml",
    "f16": DATA_ROOT / "aircraft" / "f16" / "f16.xml",
}


def _text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _attr(root: ET.Element, path: str, attr: str) -> str:
    node = root.find(path)
    if node is None:
        return ""
    return node.attrib.get(attr, "").strip()


def _refs(root: ET.Element, tag: str) -> list[str]:
    return [node.attrib["file"].strip() for node in root.iter(tag) if node.attrib.get("file")]


def _contains_property(root: ET.Element, prop: str) -> bool:
    for node in root.iter():
        if node.text and prop in node.text:
            return True
        if any(prop in value for value in node.attrib.values()):
            return True
    return False


def audit_aircraft(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    systems_dir = xml_path.parent / "Systems"
    metrics = {
        "wingarea": _text(root, "metrics/wingarea"),
        "wingspan": _text(root, "metrics/wingspan"),
        "chord": _text(root, "metrics/chord"),
        "htailarea": _text(root, "metrics/htailarea"),
        "vtailarea": _text(root, "metrics/vtailarea"),
    }
    mass_balance = {
        "emptywt": _text(root, "mass_balance/emptywt"),
        "ixx": _text(root, "mass_balance/ixx"),
        "iyy": _text(root, "mass_balance/iyy"),
        "izz": _text(root, "mass_balance/izz"),
    }
    tanks = []
    for tank in root.findall(".//tank"):
        tanks.append({
            "capacity": _text(tank, "capacity"),
            "contents": _text(tank, "contents"),
        })
    command_props = {
        "elevator": _contains_property(root, "fcs/elevator-cmd-norm"),
        "aileron": _contains_property(root, "fcs/aileron-cmd-norm"),
        "rudder": _contains_property(root, "fcs/rudder-cmd-norm"),
        "throttle": _contains_property(root, "fcs/throttle-cmd-norm"),
    }
    flight_control = root.find("flight_control")
    return {
        "xml_path": xml_path,
        "aircraft_name": root.attrib.get("name", ""),
        "description": _text(root, "fileheader/description"),
        "metrics": metrics,
        "mass_balance": mass_balance,
        "engine_files": _refs(root, "engine"),
        "thruster_files": _refs(root, "thruster"),
        "fuel_tanks": tanks,
        "has_flight_control": flight_control is not None,
        "flight_control_channels": len(root.findall(".//flight_control/channel")),
        "function_count": len(root.findall(".//function")),
        "aerosurface_scale_count": len(root.findall(".//aerosurface_scale")),
        "command_props": command_props,
        "system_refs": _refs(root, "system"),
        "include_refs": _refs(root, "include"),
        "has_systems_dir": systems_dir.is_dir(),
        "line_count": len(xml_path.read_text(encoding="utf-8", errors="ignore").splitlines()),
    }


def summarize_simplification(a4: dict, f16: dict) -> list[str]:
    notes = []
    if not a4["system_refs"] and f16["system_refs"]:
        notes.append("A-4 has no system references, while f16 declares system XML dependencies.")
    if a4["function_count"] < f16["function_count"]:
        notes.append(
            f"A-4 has fewer aerodynamic/control functions ({a4['function_count']} vs {f16['function_count']})."
        )
    if a4["line_count"] < f16["line_count"]:
        notes.append(f"A-4 XML is shorter ({a4['line_count']} lines vs {f16['line_count']} lines).")
    if a4["flight_control_channels"] <= f16["flight_control_channels"]:
        notes.append(
            f"A-4 flight-control channel count is not more detailed than f16 "
            f"({a4['flight_control_channels']} vs {f16['flight_control_channels']})."
        )
    return notes


def _print_report(name: str, report: dict) -> None:
    print(f"{name}:")
    print(f"- xml_path: {report['xml_path']}")
    print(f"- aircraft_name: {report['aircraft_name']}")
    print(f"- description: {report['description']}")
    print(f"- metrics: {report['metrics']}")
    print(f"- mass_balance: {report['mass_balance']}")
    print(f"- engine_files: {report['engine_files']}")
    print(f"- thruster_files: {report['thruster_files']}")
    print(f"- fuel_tanks: {report['fuel_tanks']}")
    print(f"- has_flight_control: {report['has_flight_control']}")
    print(f"- flight_control_channels: {report['flight_control_channels']}")
    print(f"- function_count: {report['function_count']}")
    print(f"- aerosurface_scale_count: {report['aerosurface_scale_count']}")
    print(f"- command_props: {report['command_props']}")
    print(f"- system_refs: {report['system_refs']}")
    print(f"- include_refs: {report['include_refs']}")
    print(f"- has_systems_dir: {report['has_systems_dir']}")
    print(f"- line_count: {report['line_count']}")
    print()


def main() -> None:
    reports = {name: audit_aircraft(path) for name, path in AIRCRAFT.items()}
    for name, report in reports.items():
        _print_report(name, report)
    print("simplification_notes:")
    for note in summarize_simplification(reports["A-4"], reports["f16"]):
        print(f"- {note}")


if __name__ == "__main__":
    main()
