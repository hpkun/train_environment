from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.simulator import AircraftSimulator


PROPERTIES = [
    "fcs/throttle-cmd-norm",
    "fcs/throttle-pos-norm",
    "fcs/elevator-cmd-norm",
    "fcs/elevator-pos-rad",
    "fcs/aileron-cmd-norm",
    "fcs/rudder-cmd-norm",
    "propulsion/engine[0]/throttle-pos-norm",
    "propulsion/engine[0]/n1",
    "propulsion/engine[0]/n2",
    "propulsion/engine[0]/thrust-lbs",
]


def _catalog_contains(sim: AircraftSimulator, prop: str) -> bool:
    try:
        return prop in "\n".join(sim.jsbsim_exec.query_property_catalog(""))
    except Exception:
        return False


def _safe_get(sim: AircraftSimulator, prop: str) -> float | None:
    try:
        value = float(sim.get_property_value(prop))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _run_throttle_case(model: str, throttle_cmd: float, duration: float) -> dict:
    sim = AircraftSimulator(
        uid=f"{model}_{throttle_cmd}",
        color="Red",
        model=model,
        sim_freq=60,
        num_missiles=0,
        suppress_jsbsim_output=True,
    )
    try:
        steps = max(1, int(round(duration * 60)))
        speeds = []
        prop_values = {prop: [] for prop in PROPERTIES}
        for _ in range(steps):
            sim.set_property_value("fcs/aileron-cmd-norm", 0.0)
            sim.set_property_value("fcs/elevator-cmd-norm", 0.0)
            sim.set_property_value("fcs/rudder-cmd-norm", 0.0)
            sim.set_property_value("fcs/throttle-cmd-norm", float(throttle_cmd))
            sim.run()
            speeds.append(float(np.linalg.norm(sim.get_velocity())))
            for prop in PROPERTIES:
                value = _safe_get(sim, prop)
                if value is not None:
                    prop_values[prop].append(value)
        return {
            "model": model,
            "throttle_cmd": throttle_cmd,
            "final_speed": float(speeds[-1]),
            "mean_speed": float(np.mean(speeds)),
            "property_exists": {prop: _catalog_contains(sim, prop) for prop in PROPERTIES},
            "property_means": {
                prop: float(np.mean(values)) if values else None
                for prop, values in prop_values.items()
            },
            "property_finals": {
                prop: float(values[-1]) if values else None
                for prop, values in prop_values.items()
            },
        }
    finally:
        sim.close()


def diagnose_model(model: str, duration: float = 10.0) -> dict:
    cases = [_run_throttle_case(model, cmd, duration) for cmd in (0.0, 0.5, 1.0)]
    speeds = {case["throttle_cmd"]: case["final_speed"] for case in cases}
    delta = speeds[1.0] - speeds[0.0]
    if delta > 5.0:
        conclusion = "active"
    elif abs(delta) <= 5.0:
        conclusion = "inactive"
    else:
        conclusion = "inconclusive"
    return {"model": model, "cases": cases, "speed_delta_1_minus_0": delta, "conclusion": conclusion}


def print_report(report: dict) -> None:
    print(f"{report['model']}:")
    for case in report["cases"]:
        print(
            f"- throttle_cmd={case['throttle_cmd']:.1f} "
            f"final_speed={case['final_speed']:.3f} mean_speed={case['mean_speed']:.3f}"
        )
        print(f"  property_exists={case['property_exists']}")
        print(f"  property_finals={case['property_finals']}")
    print(f"- speed_delta_1_minus_0={report['speed_delta_1_minus_0']:.3f}")
    print(f"- throttle path appears: {report['conclusion']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    for model in ("A-4", "f16"):
        print_report(diagnose_model(model, args.duration))


if __name__ == "__main__":
    main()
