from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.core.utils import load_yaml


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _xml_dependencies(model_root: Path, model_name: str) -> list[tuple[str, Path, bool]]:
    model_xml = model_root / "aircraft" / model_name / f"{model_name}.xml"
    if not model_xml.exists():
        return [("aircraft", model_xml, False)]
    result = [("aircraft", model_xml, True)]
    try:
        root = ET.parse(model_xml).getroot()
    except ET.ParseError as exc:
        return result + [(f"xml_parse_error: {exc}", model_xml, False)]
    for elem in root.iter():
        ref = elem.attrib.get("file")
        if not ref:
            continue
        tag = elem.tag.lower()
        if tag in {"engine", "thruster"}:
            path = model_root / "engine" / f"{ref}.xml"
        elif tag == "system":
            path = model_root / "aircraft" / model_name / "Systems" / f"{ref}.xml"
        else:
            continue
        result.append((f"{tag}:{ref}", path, path.exists()))
    return result


def _configure_initial_conditions(fdm) -> None:
    props = {
        "ic/long-gc-deg": 120.0,
        "ic/lat-geod-deg": 60.0,
        "ic/h-sl-ft": 6000.0 / 0.3048,
        "ic/psi-true-deg": 0.0,
        "ic/theta-deg": 0.0,
        "ic/phi-deg": 0.0,
        "ic/u-fps": 250.0 / 0.3048,
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
        "ic/terrain-elevation-ft": 0.0,
    }
    for name, value in props.items():
        fdm.set_property_value(name, float(value))


def _try_load_model(model_root: Path, model_name: str) -> dict[str, object]:
    try:
        import jsbsim
    except Exception as exc:
        return {
            "load_success": False,
            "run_ic_success": False,
            "error": f"jsbsim import failed: {exc}",
            "jsbsim_output": "",
        }

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            fdm = jsbsim.FGFDMExec(str(model_root))
            fdm.set_debug_level(0)
            if hasattr(fdm, "set_aircraft_path"):
                fdm.set_aircraft_path(str(model_root / "aircraft"))
            if hasattr(fdm, "set_engine_path"):
                fdm.set_engine_path(str(model_root / "engine"))
            ok = bool(fdm.load_model(model_name))
            run_ic_ok = False
            if ok:
                _configure_initial_conditions(fdm)
                run_ic_ok = bool(fdm.run_ic())
        return {
            "load_success": ok,
            "run_ic_success": run_ic_ok,
            "error": "" if ok and run_ic_ok else "load_model or run_ic returned false",
            "jsbsim_output": output.getvalue().strip(),
        }
    except Exception as exc:
        return {
            "load_success": False,
            "run_ic_success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "jsbsim_output": output.getvalue().strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    args = parser.parse_args()

    config = load_yaml(str(_resolve(args.config)))
    model_root = _resolve(config.get("jsbsim_model_root", "uav_env/JSBSim/models"))
    aircraft_path = model_root / "aircraft"
    engine_path = model_root / "engine"
    spec = importlib.util.find_spec("jsbsim")
    print(f"config: {args.config}")
    print(f"jsbsim package installed: {'yes' if spec else 'no'}")
    if spec:
        import jsbsim
        print(f"jsbsim version: {getattr(jsbsim, '__version__', 'unknown')}")
    else:
        print("jsbsim version: unavailable")
        print("install hint: pip install -r requirements.txt")
        print("install hint: pip install jsbsim==1.1.6")
    print(f"model_root absolute path: {model_root}")
    print(f"aircraft path: {aircraft_path}")
    print(f"engine path: {engine_path}")
    print(f"aircraft_path_exists: {aircraft_path.exists()}")
    print(f"engine_path_exists: {engine_path.exists()}")

    for label, name in [
        ("J52 engine XML", "J52.xml"),
        ("F100-PW-229 engine XML", "F100-PW-229.xml"),
        ("direct thruster XML", "direct.xml"),
    ]:
        path = model_root / "engine" / name
        print(f"{label} exists: {path.exists()} path={path}")

    seen: set[str] = set()
    for type_name, spec in config.get("aircraft_type_params", {}).items():
        model_name = str(spec.get("aircraft_model", ""))
        if model_name in seen:
            continue
        seen.add(model_name)
        model_path = _resolve(str(spec.get("model_path", "")))
        print(f"\nmodel_name: {model_name}")
        print(f"declared_type: {type_name}")
        print(f"declared_model_path: {model_path}")
        print(f"{model_name} XML exists: {model_path.exists()}")
        missing = []
        for dep_label, dep_path, exists in _xml_dependencies(model_root, model_name):
            print(f"dependency {dep_label}: exists={exists} path={dep_path}")
            if not exists:
                missing.append(str(dep_path))
        result = _try_load_model(model_root, model_name)
        print(f"{model_name} load_model success: {result['load_success']}")
        print(f"{model_name} run_ic success: {result['run_ic_success']}")
        if missing:
            print(f"{model_name} missing file path: {missing}")
        else:
            print(f"{model_name} missing file path: none detected")
        print(f"{model_name} JSBSim error message: {result['error']}")
        if result["jsbsim_output"]:
            print(f"{model_name} JSBSim output:\n{result['jsbsim_output']}")
        else:
            print(f"{model_name} JSBSim output: <empty>")


if __name__ == "__main__":
    main()
