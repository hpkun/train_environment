"""Rollout diagnostics for tam_env direct-FCS response.

This script does not train. It resets the TAM direct-FCS diagnostic
environment for fixed single-axis action scenarios and records the aircraft
response needed to check control signs and gross stability.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml"
)

RAW_DIRECT_FCS_SCENARIOS: dict[str, np.ndarray] = {
    "neutral": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "throttle_high": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "aileron_pos": np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32),
    "aileron_neg": np.array([0.0, -0.2, 0.0, 0.0], dtype=np.float32),
    "elevator_pos": np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32),
    "elevator_neg": np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32),
    "rudder_pos": np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32),
    "rudder_neg": np.array([0.0, 0.0, 0.0, -0.2], dtype=np.float32),
}

MANEUVER_FCS_SCENARIOS: dict[str, np.ndarray] = {
    "neutral": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "throttle_high": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "bank_left": np.array([0.0, -0.5, 0.0, 0.0], dtype=np.float32),
    "bank_right": np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32),
    "pull_up": np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32),
    "push_down": np.array([0.0, 0.0, -0.5, 0.0], dtype=np.float32),
    "yaw_left": np.array([0.0, 0.0, 0.0, -0.5], dtype=np.float32),
    "yaw_right": np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
}

SCENARIOS = RAW_DIRECT_FCS_SCENARIOS


FIELDNAMES = [
    "step",
    "scenario",
    "tam_control_mode",
    "tam_command_semantics",
    "agent_id",
    "alive",
    "altitude_m",
    "speed_mps",
    "roll_rad",
    "pitch_rad",
    "heading_rad",
    "throttle_cmd",
    "aileron_cmd",
    "elevator_cmd",
    "rudder_cmd",
    "bank_cmd_rad",
    "nz_cmd_g",
    "yaw_rate_cmd_rad_s",
    "bank_error_rad",
    "roll_rate_rad_s",
    "nz_current_g",
    "nz_error_g",
    "pitch_rate_rad_s",
    "yaw_rate_rad_s",
    "raw_aileron",
    "raw_elevator",
    "raw_rudder",
    "clipped_aileron",
    "clipped_elevator",
    "clipped_rudder",
    "g_load",
    "g_load_total",
    "g_load_x",
    "g_load_y",
    "g_load_z",
    "reward",
    "terminated",
    "truncated",
    "initial_frame",
    "crash_or_done",
    "nonfinite_state",
]


def _scenarios_for_mode(mode: str) -> dict[str, np.ndarray]:
    if mode == "maneuver_fcs":
        return MANEUVER_FCS_SCENARIOS
    return RAW_DIRECT_FCS_SCENARIOS


def _scenario_names(selected: str, mode: str) -> list[str]:
    scenarios = _scenarios_for_mode(mode)
    if selected == "all":
        return list(scenarios)
    if selected not in scenarios:
        raise ValueError(f"unknown scenario {selected!r}; choose one of {sorted(scenarios)} or all")
    return [selected]


def _wrapped_angle_delta(new_angle: float, old_angle: float) -> float:
    return float((float(new_angle) - float(old_angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _finite_or_empty(value):
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return val


def _make_actions(env, scenario_action: np.ndarray, agent_id: str | None) -> dict:
    actions = {}
    neutral = SCENARIOS["neutral"]
    for aid in env.agent_ids:
        actions[aid] = scenario_action.copy() if agent_id in (None, aid) else neutral.copy()
    return actions


def _nonfinite_state(row: dict) -> bool:
    for key in ("altitude_m", "speed_mps", "roll_rad", "pitch_rad", "heading_rad"):
        value = row.get(key)
        if value == "":
            continue
        if not np.isfinite(float(value)):
            return True
    return False


def _row_from_info(
    step: int,
    scenario: str,
    aid: str,
    rewards,
    terminated,
    truncated,
    info,
    *,
    initial_frame: bool = False,
) -> dict:
    diag = (info.get("tam_control_diagnostics", {}) or {}).get(aid, {}) or {}
    terms = (info.get("tam_controller_terms", {}) or {}).get(aid, {}) or {}
    fcs = (info.get("tam_applied_fcs", {}) or {}).get(aid)
    fcs = fcs if isinstance(fcs, (list, tuple)) and len(fcs) >= 4 else [None, None, None, None]
    command = (info.get("tam_control_commands", {}) or {}).get(aid)
    command = command if isinstance(command, (list, tuple)) and len(command) >= 4 else [None, None, None, None]
    row = {
        "step": step,
        "scenario": scenario,
        "tam_control_mode": str(info.get("tam_control_mode", "")),
        "tam_command_semantics": "|".join(info.get("tam_command_semantics", []) or []),
        "agent_id": aid,
        "alive": bool(diag.get("alive", False)),
        "altitude_m": _finite_or_empty(diag.get("altitude_m")),
        "speed_mps": _finite_or_empty(diag.get("speed_mps")),
        "roll_rad": _finite_or_empty(diag.get("roll_rad")),
        "pitch_rad": _finite_or_empty(diag.get("pitch_rad")),
        "heading_rad": _finite_or_empty(diag.get("heading_rad")),
        "throttle_cmd": _finite_or_empty(fcs[0]),
        "aileron_cmd": _finite_or_empty(fcs[1]),
        "elevator_cmd": _finite_or_empty(fcs[2]),
        "rudder_cmd": _finite_or_empty(fcs[3]),
        "bank_cmd_rad": _finite_or_empty(command[1] if info.get("tam_control_mode") == "maneuver_fcs" else ""),
        "nz_cmd_g": _finite_or_empty(command[2] if info.get("tam_control_mode") == "maneuver_fcs" else ""),
        "yaw_rate_cmd_rad_s": _finite_or_empty(command[3] if info.get("tam_control_mode") == "maneuver_fcs" else ""),
        "bank_error_rad": _finite_or_empty(terms.get("bank_error_rad")),
        "roll_rate_rad_s": _finite_or_empty(terms.get("roll_rate_rad_s")),
        "nz_current_g": _finite_or_empty(terms.get("nz_current_g")),
        "nz_error_g": _finite_or_empty(terms.get("nz_error_g")),
        "pitch_rate_rad_s": _finite_or_empty(terms.get("pitch_rate_rad_s")),
        "yaw_rate_rad_s": _finite_or_empty(terms.get("yaw_rate_rad_s")),
        "raw_aileron": _finite_or_empty(terms.get("raw_aileron")),
        "raw_elevator": _finite_or_empty(terms.get("raw_elevator")),
        "raw_rudder": _finite_or_empty(terms.get("raw_rudder")),
        "clipped_aileron": _finite_or_empty(terms.get("clipped_aileron")),
        "clipped_elevator": _finite_or_empty(terms.get("clipped_elevator")),
        "clipped_rudder": _finite_or_empty(terms.get("clipped_rudder")),
        "g_load": _finite_or_empty(diag.get("g_load")),
        "g_load_total": _finite_or_empty(diag.get("g_load_total")),
        "g_load_x": _finite_or_empty(diag.get("g_load_x")),
        "g_load_y": _finite_or_empty(diag.get("g_load_y")),
        "g_load_z": _finite_or_empty(diag.get("g_load_z")),
        "reward": "" if initial_frame else _finite_or_empty((rewards or {}).get(aid)),
        "terminated": bool((terminated or {}).get(aid, False)),
        "truncated": bool((truncated or {}).get(aid, False)),
        "initial_frame": bool(initial_frame),
    }
    row["crash_or_done"] = bool((not row["alive"]) or row["terminated"] or row["truncated"])
    row["nonfinite_state"] = _nonfinite_state(row)
    return row


def _print_summary(scenario: str, rows: list[dict], missiles_per_plane: int):
    by_agent: dict[str, list[dict]] = {}
    for row in rows:
        by_agent.setdefault(str(row["agent_id"]), []).append(row)

    crashed = any((not bool(row.get("alive"))) or bool(row.get("terminated")) for row in rows)
    nonfinite = any(bool(row.get("nonfinite_state")) for row in rows)
    print(
        f"[tam-direct-fcs] scenario={scenario} "
        f"missiles_per_plane={missiles_per_plane} crash={crashed} nonfinite={nonfinite}"
    )
    for aid, agent_rows in by_agent.items():
        first = agent_rows[0]
        last = agent_rows[-1]
        def delta(key: str):
            if first.get(key) == "" or last.get(key) == "":
                return ""
            if key == "heading_rad":
                return _wrapped_angle_delta(float(last[key]), float(first[key]))
            return float(last[key]) - float(first[key])
        print(
            "  "
            f"{aid}: final_alt={last.get('altitude_m')} "
            f"final_speed={last.get('speed_mps')} "
            f"d_roll={delta('roll_rad')} "
            f"d_pitch={delta('pitch_rad')} "
            f"d_heading={delta('heading_rad')}"
        )


def run(args: argparse.Namespace) -> int:
    scenario_rows: list[dict] = []
    scenario_names = _scenario_names(args.scenario, args.tam_control_mode)
    scenarios = _scenarios_for_mode(args.tam_control_mode)
    for scenario_index, name in enumerate(scenario_names):
        env = make_env(
            args.config,
            env_type="tam",
            max_steps=args.steps,
            suppress_jsbsim_output=True,
            num_missiles_per_plane=args.num_missiles_per_plane,
            strict_action_shape=args.strict_action_shape,
            tam_throttle_min=args.tam_throttle_min,
            tam_throttle_max=args.tam_throttle_max,
            tam_surface_limit=args.tam_surface_limit,
            tam_control_mode=args.tam_control_mode,
        )
        rows: list[dict] = []
        try:
            _obs, reset_info = env.reset(seed=args.seed + scenario_index)
            if args.record_initial_frame:
                for aid in env.agent_ids:
                    rows.append(_row_from_info(
                        -1, name, aid, {}, {}, {}, reset_info, initial_frame=True
                    ))
            action = scenarios[name]
            for step in range(args.steps):
                actions = _make_actions(env, action, args.agent_id)
                _obs, rewards, terminated, truncated, info = env.step(actions)
                for aid in env.agent_ids:
                    rows.append(_row_from_info(
                        step, name, aid, rewards, terminated, truncated, info
                    ))
                if (not args.continue_after_agent_done
                        and (all(terminated.values()) or all(truncated.values()))):
                    break
        finally:
            env.close()
        scenario_rows.extend(rows)
        _print_summary(name, rows, args.num_missiles_per_plane)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(scenario_rows)
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="outputs/tam_direct_fcs_response.csv")
    all_scenarios = sorted(set(RAW_DIRECT_FCS_SCENARIOS) | set(MANEUVER_FCS_SCENARIOS))
    parser.add_argument("--scenario", default="all", choices=["all", *all_scenarios])
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--tam-control-mode", default="raw_direct_fcs",
                        choices=["raw_direct_fcs", "maneuver_fcs"])
    parser.add_argument("--num-missiles-per-plane", type=int, default=0)
    parser.add_argument("--record-initial-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-after-agent-done", action="store_true")
    parser.add_argument("--strict-action-shape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tam-throttle-min", type=float, default=0.4)
    parser.add_argument("--tam-throttle-max", type=float, default=0.9)
    parser.add_argument("--tam-surface-limit", type=float, default=1.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
