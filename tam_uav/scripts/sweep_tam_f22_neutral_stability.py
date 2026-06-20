"""Coarse-to-fine fixed-action F22 neutral stability sweep."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


AILERON_BINS = (18, 19, 20, 21, 22)
RUDDER_BINS = (18, 19, 20, 21, 22)
ELEVATOR_BINS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)


def select_stable_candidate(candidates):
    if not candidates:
        raise ValueError("stability sweep produced no candidates")
    def score(candidate):
        alive_score = candidate["death_step"] if candidate["death_step"] > 0 else 1001
        action = candidate["action"]
        surface_extremity = sum(abs(value - 20) for value in action[1:])
        if candidate["primary_pass"]:
            return (
                1, -abs(candidate["mean_vertical_speed_mps"]),
                -candidate["max_abs_roll_rad"], -candidate["max_abs_pitch_rad"],
                candidate["min_altitude_m"], candidate["final_speed_mps"],
                alive_score, -surface_extremity,
            )
        return (
            0, alive_score, -abs(candidate["mean_vertical_speed_mps"]),
            -candidate["max_abs_roll_rad"], -candidate["max_abs_pitch_rad"],
            candidate["min_altitude_m"], candidate["final_speed_mps"],
            -surface_extremity,
        )
    return max(candidates, key=score)


def _disable_all_missiles(env):
    for sim in [*env.red_planes.values(), *env.blue_planes.values()]:
        sim.num_left_missiles = 0
        sim.num_missiles = 0


def _run_candidate(env, action, max_steps, seed):
    env.reset(seed=seed)
    _disable_all_missiles(env)
    rows = []
    for _ in range(max_steps):
        actions = {}
        for agent_id in [*env.red_ids, *env.blue_ids]:
            if agent_id == "red_0":
                actions[agent_id] = np.asarray(action, dtype=np.int64)
            else:
                actions[agent_id] = np.asarray([39, 20, 4, 20], dtype=np.int64)
        _obs, _rewards, terminated, truncated, _info = env.step(actions)
        mav = env.red_planes["red_0"]
        velocity = np.asarray(mav.get_velocity(), dtype=np.float64)
        roll, pitch, _yaw = (float(value) for value in mav.get_rpy())
        rows.append({
            "speed_mps": float(np.linalg.norm(velocity)),
            "altitude_m": float(mav.get_geodetic()[2]),
            "vertical_speed_mps": float(velocity[2]),
            "roll_rad": roll, "pitch_rad": pitch,
        })
        if not mav.is_alive:
            break
    mav = env.red_planes["red_0"]
    death_reason = env._death_reasons.get("red_0") or "alive"
    result = {
        "action": list(action),
        "steps": len(rows),
        "final_altitude_m": rows[-1]["altitude_m"],
        "min_altitude_m": min(row["altitude_m"] for row in rows),
        "final_speed_mps": rows[-1]["speed_mps"],
        "min_speed_mps": min(row["speed_mps"] for row in rows),
        "mean_vertical_speed_mps": float(np.mean([
            row["vertical_speed_mps"] for row in rows
        ])),
        "max_abs_roll_rad": max(abs(row["roll_rad"]) for row in rows),
        "max_abs_pitch_rad": max(abs(row["pitch_rad"]) for row in rows),
        "death_reason": death_reason,
        "death_step": -1 if mav.is_alive else int(env.current_step),
    }
    result["primary_pass"] = bool(
        result["death_reason"] != "Crash_LowAlt"
        and result["steps"] >= max_steps
        and result["final_speed_mps"] >= 180.0
        and result["min_altitude_m"] >= 4500.0
    )
    return result


def _fine_actions(selected):
    _, aileron, elevator, rudder = selected["action"]
    values = itertools.product(
        range(max(0, aileron - 1), min(39, aileron + 1) + 1),
        range(max(0, elevator - 2), min(39, elevator + 2) + 1),
        range(max(0, rudder - 1), min(39, rudder + 1) + 1),
    )
    return [(39, a, e, r) for a, e, r in values]


def run_sweep(config, output_dir, max_steps=1000, seed=0):
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    existing_path = out_dir / "tam_f22_neutral_stability_sweep.json"
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    base_profile = dict(
        env.tam_direct_fcs_calibration.get("by_model", {}).get("f22", {})
    )
    calibration_signature = json.dumps(base_profile, sort_keys=True)
    coarse_actions = [
        (39, aileron, elevator, rudder)
        for aileron, rudder, elevator in itertools.product(
            AILERON_BINS, RUDDER_BINS, ELEVATOR_BINS
        )
    ]
    resumed = False
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if (
            existing.get("config") == config
            and existing.get("max_steps") == max_steps
            and existing.get("calibration_signature") == calibration_signature
        ):
            coarse = [
                candidate for candidate in existing.get("candidates", [])
                if "calibration_profile" not in candidate
            ]
            resumed = bool(coarse)
        else:
            coarse = []
    else:
        coarse = []
    if not coarse:
        coarse = [
            _run_candidate(env, action, max_steps, seed) for action in coarse_actions
        ]
    coarse_selected = select_stable_candidate(coarse)
    seen = {tuple(candidate["action"]) for candidate in coarse}
    fine = [] if resumed else [
        _run_candidate(env, action, max_steps, seed)
        for action in _fine_actions(coarse_selected) if tuple(action) not in seen
    ]
    raw_middle = -1.0 + 2.0 * 20.0 / 39.0
    calibration_candidates = []
    if not any(candidate["primary_pass"] for candidate in coarse + fine):
        profile = env.tam_direct_fcs_calibration["by_model"]["f22"]
        profile["aileron_bias"] = -raw_middle
        profile["rudder_bias"] = -raw_middle
        for elevator_bias in np.arange(-0.045, -0.0049, 0.002):
            profile["elevator_bias"] = float(elevator_bias)
            candidate = _run_candidate(env, (39, 20, 20, 20), max_steps, seed)
            candidate["calibration_profile"] = {
                "aileron_bias": -raw_middle,
                "elevator_bias": float(elevator_bias),
                "rudder_bias": -raw_middle,
            }
            calibration_candidates.append(candidate)
    env.close()
    candidates = coarse + fine + calibration_candidates
    selected = select_stable_candidate(candidates)
    result = {
        "config": config, "max_steps": max_steps,
        "calibration_signature": calibration_signature,
        "coarse_candidate_count": len(coarse), "coarse_resumed": resumed,
        "fine_candidate_count": len(fine),
        "calibration_candidate_count": len(calibration_candidates),
        "primary_pass_count": sum(candidate["primary_pass"] for candidate in candidates),
        "selected": selected,
        "requires_elevator_calibration": "calibration_profile" in selected,
        "candidates": candidates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tam_f22_neutral_stability_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    columns = list(dict.fromkeys(
        key for candidate in candidates for key in candidate
    ))
    with (out_dir / "tam_f22_neutral_stability_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            row = dict(candidate)
            row["action"] = json.dumps(row["action"])
            if "calibration_profile" in row:
                row["calibration_profile"] = json.dumps(row["calibration_profile"])
            writer.writerow(row)
    lines = [
        "# TAM F22 Neutral Stability Sweep", "",
        f"- Coarse candidates: `{len(coarse)}`",
        f"- Fine candidates: `{len(fine)}`",
        f"- Primary passes: `{result['primary_pass_count']}`",
        f"- Selected action: `{selected['action']}`",
        f"- Selected primary pass: `{selected['primary_pass']}`",
        f"- Requires elevator calibration: `{result['requires_elevator_calibration']}`",
    ]
    (out_dir / "tam_f22_neutral_stability_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_sweep(
        args.config, args.output_dir, max_steps=args.max_steps, seed=args.seed
    )
    print(json.dumps({
        "primary_pass_count": result["primary_pass_count"],
        "selected": result["selected"],
    }, indent=2))


if __name__ == "__main__":
    main()
