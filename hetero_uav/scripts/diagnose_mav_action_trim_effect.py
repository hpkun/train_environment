"""Compare MAV zero-action flight with config trim disabled and enabled."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_mav_flight_stability import run_case


def _zero_case() -> dict:
    return {
        "case": "zero_all",
        "mav_action": [0.0, 0.0, 0.0],
        "attack_action": [0.0, 0.0, 0.0],
    }


def _rename(record: dict, case_name: str) -> dict:
    out = dict(record)
    out["case"] = case_name
    return out


def _summary(records: list[dict]) -> dict:
    by_case = {record["case"]: record for record in records}
    disabled = by_case["trim_disabled_zero"]
    enabled = by_case["trim_enabled_zero"]
    trim_improves_altitude = (
        enabled["mav_final_altitude_m"] > disabled["mav_final_altitude_m"] + 500.0
        or (enabled["stable_level_like"] and not disabled["stable_level_like"])
    )
    trim_prevents_crash = bool(disabled["mav_crashed"] and not enabled["mav_crashed"])
    return {
        "trim_improves_altitude": bool(trim_improves_altitude),
        "trim_prevents_crash": trim_prevents_crash,
        "recommend_keep_trim": bool(trim_improves_altitude or trim_prevents_crash),
        "disabled_altitude_delta_m": disabled["mav_altitude_delta_m"],
        "enabled_altitude_delta_m": enabled["mav_altitude_delta_m"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/mav_action_trim_effect.json",
    )
    args = parser.parse_args()

    base_case = _zero_case()
    disabled = run_case(
        config=args.config,
        case=base_case,
        steps=args.steps,
        seed=args.seed,
        blue_policy_name="zero",
        export_acmi=False,
        output_acmi_dir=Path(""),
        disable_config_trim=True,
    )
    enabled = run_case(
        config=args.config,
        case=base_case,
        steps=args.steps,
        seed=args.seed,
        blue_policy_name="zero",
        export_acmi=False,
        output_acmi_dir=Path(""),
        disable_config_trim=False,
    )
    records = [
        _rename(disabled, "trim_disabled_zero"),
        _rename(enabled, "trim_enabled_zero"),
    ]
    data = {"summary": _summary(records), "records": records}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"output_json: {output_json}")
    for record in records:
        print(
            f"{record['case']}: alt_delta={record['mav_altitude_delta_m']:.3f} "
            f"final_alt={record['mav_final_altitude_m']:.3f} "
            f"min_alt={record['mav_min_altitude_m']:.3f} "
            f"crashed={record['mav_crashed']} stable={record['stable_level_like']}"
        )
    print(f"summary: {data['summary']}")


if __name__ == "__main__":
    main()
