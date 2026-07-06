"""Lightweight rollout audit for tam_brma_paper_aligned_v1 reward components."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


COMPONENT_KEYS = [
    "paper_v1_uav_flight",
    "paper_v1_uav_adv",
    "paper_v1_uav_end",
    "paper_v1_uav_total",
    "paper_v1_mav_flight",
    "paper_v1_mav_safety",
    "paper_v1_mav_support",
    "paper_v1_mav_event_raw",
    "paper_v1_mav_scaled_tam",
    "paper_v1_mav_total",
]


def _action_dict(env, rng: np.random.Generator, mode: str) -> dict[str, np.ndarray]:
    if mode == "random":
        return {
            aid: rng.uniform(-0.2, 0.2, size=3).astype(np.float32)
            for aid in env.agent_ids
        }
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": float(arr.size),
    }


def _read_reset_state(env) -> dict:
    return {
        "paper_v1_mav_team_credit_used": float(
            getattr(env, "_paper_aligned_v1_mav_team_credit_used", 0.0) or 0.0),
        "paper_v1_mav_death_penalized": bool(
            getattr(env, "_paper_aligned_v1_mav_death_penalized", False)),
    }


def run_audit(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    env = make_env(args.config, max_steps=args.max_steps)
    values = {key: [] for key in COMPONENT_KEYS}
    event_raw_samples: list[float] = []
    reset_records: list[dict] = []
    end_records: list[dict] = []
    try:
        for ep in range(args.episodes):
            _obs, _info = env.reset(seed=args.seed + ep)
            reset_records.append(_read_reset_state(env))
            for _step in range(args.max_steps):
                actions = _action_dict(env, rng, args.action_mode)
                _obs, _rewards, terminated, truncated, info = env.step(actions)
                rc = info.get("reward_components", {}) if isinstance(info, dict) else {}
                for comp in rc.values():
                    if not isinstance(comp, dict):
                        continue
                    for key in COMPONENT_KEYS:
                        if key in comp:
                            try:
                                val = float(comp[key])
                                values[key].append(val)
                                if key == "paper_v1_mav_event_raw":
                                    event_raw_samples.append(val)
                            except (TypeError, ValueError):
                                pass
                if any(bool(v) for v in terminated.values()) or any(bool(v) for v in truncated.values()):
                    break
            end_records.append(_read_reset_state(env))
    finally:
        env.close()
    summary = {key: _summary(vals) for key, vals in values.items()}
    return summary, reset_records, end_records, event_raw_samples


def _write_reset_status(reset_records: list[dict], out_dir: Path) -> None:
    credits = [r["paper_v1_mav_team_credit_used"] for r in reset_records]
    deaths = [r["paper_v1_mav_death_penalized"] for r in reset_records]
    payload = {
        "episodes": len(reset_records),
        "reset_team_credit_all_zero": all(abs(c) < 1e-9 for c in credits),
        "reset_death_penalized_all_false": all(not d for d in deaths),
        "max_team_credit_at_reset": float(max(credits)) if credits else 0.0,
        "any_death_penalized_at_reset": any(deaths),
        "episode_reset_records": [
            {"episode": i, **r} for i, r in enumerate(reset_records)
        ],
    }
    with (out_dir / "reset_status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if not payload["reset_team_credit_all_zero"]:
        print(f"WARNING: team_credit_used not zero at reset! max={payload['max_team_credit_at_reset']}")
    if payload["any_death_penalized_at_reset"]:
        print("WARNING: death_penalized is True at reset!")


def _write_reward_scale_summary(summary: dict, event_raw_samples: list[float],
                                out_dir: Path) -> None:
    def _abs_mean(key):
        return abs(float(summary.get(key, {}).get("mean", 0.0)))

    uav_total = _abs_mean("paper_v1_uav_total")
    mav_total = _abs_mean("paper_v1_mav_total")
    uav_adv = _abs_mean("paper_v1_uav_adv")
    mav_tam = _abs_mean("paper_v1_mav_scaled_tam")
    mav_safety = _abs_mean("paper_v1_mav_safety")
    mav_support = _abs_mean("paper_v1_mav_support")
    mav_event = _abs_mean("paper_v1_mav_event_raw")
    mav_flight = max(_abs_mean("paper_v1_mav_flight"), 1e-6)
    nonzero = int(sum(1 for v in event_raw_samples if abs(v) > 1e-9))

    payload = {
        "uav_total_abs_mean": uav_total,
        "mav_total_abs_mean": mav_total,
        "uav_adv_abs_mean": uav_adv,
        "mav_scaled_tam_abs_mean": mav_tam,
        "mav_safety_abs_mean": mav_safety,
        "mav_support_abs_mean": mav_support,
        "mav_event_raw_abs_mean": mav_event,
        "mav_scaled_tam_to_flight_ratio": mav_tam / mav_flight,
        "mav_event_raw_nonzero_count": nonzero,
        "note": (
            "This audit uses zero/random policy rollouts for reward-scale "
            "checking only; it is not a learned-policy performance evaluation."
        ),
    }
    with (out_dir / "reward_scale_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_outputs(summary: dict[str, dict[str, float]], out_dir: Path,
                  reset_records: list[dict], end_records: list[dict],
                  event_raw_samples: list[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Legacy outputs (unchanged)
    with (out_dir / "paper_aligned_reward_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "paper_aligned_reward_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "mean", "std", "min", "max", "count"])
        writer.writeheader()
        for key, stats in summary.items():
            row = {"component": key}
            row.update(stats)
            writer.writerow(row)
    # New outputs
    _write_reset_status(reset_records, out_dir)
    _write_reward_scale_summary(summary, event_raw_samples, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "uav_env/JSBSim/configs/"
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"
        ),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    parser.add_argument("--output-dir", default="outputs/audit_paper_aligned_reward")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, reset_records, end_records, event_raw = run_audit(args)
    write_outputs(summary, Path(args.output_dir), reset_records, end_records, event_raw)
    print(f"wrote {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
