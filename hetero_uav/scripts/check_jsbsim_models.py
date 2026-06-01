from __future__ import annotations

import argparse
import importlib.util
import sys
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


def _try_load_model(model_root: Path, model_name: str) -> tuple[bool, str]:
    try:
        import jsbsim
    except Exception as exc:
        return False, f"jsbsim import failed: {exc}"

    try:
        fdm = jsbsim.FGFDMExec(str(model_root))
        fdm.set_debug_level(0)
        if hasattr(fdm, "set_aircraft_path"):
            fdm.set_aircraft_path(str(model_root / "aircraft"))
        if hasattr(fdm, "set_engine_path"):
            fdm.set_engine_path(str(model_root / "engine"))
        ok = bool(fdm.load_model(model_name))
        return ok, "loaded" if ok else "load_model returned false"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    args = parser.parse_args()

    config = load_yaml(str(_resolve(args.config)))
    model_root = _resolve(config.get("jsbsim_model_root", "uav_env/JSBSim/models"))
    print(f"config: {args.config}")
    print(f"model_root: {model_root}")
    print(f"jsbsim_package_installed: {importlib.util.find_spec('jsbsim') is not None}")

    engine_files = ["J52.xml", "F100-PW-229.xml", "direct.xml"]
    for name in engine_files:
        path = model_root / "engine" / name
        print(f"engine_file {name}: exists={path.exists()} path={path}")

    seen: set[str] = set()
    for type_name, spec in config.get("aircraft_type_params", {}).items():
        model_name = str(spec.get("aircraft_model", ""))
        if model_name in seen:
            continue
        seen.add(model_name)
        model_path = _resolve(str(spec.get("model_path", "")))
        file_ok = model_path.exists()
        load_ok, reason = _try_load_model(model_root, model_name)
        print(
            f"aircraft_type={type_name} model={model_name} "
            f"file_exists={file_ok} model_path={model_path} "
            f"load_success={load_ok} reason={reason}"
        )


if __name__ == "__main__":
    main()
