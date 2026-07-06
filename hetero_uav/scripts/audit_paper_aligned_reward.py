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


def run_audit(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(args.seed)
    env = make_env(args.config, max_steps=args.max_steps)
    values = {key: [] for key in COMPONENT_KEYS}
    try:
        for _ep in range(args.episodes):
            _obs, _info = env.reset(seed=args.seed + _ep)
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
                                values[key].append(float(comp[key]))
                            except (TypeError, ValueError):
                                pass
                if any(bool(v) for v in terminated.values()) or any(bool(v) for v in truncated.values()):
                    break
    finally:
        env.close()
    return {key: _summary(vals) for key, vals in values.items()}


def write_outputs(summary: dict[str, dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "paper_aligned_reward_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "paper_aligned_reward_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "mean", "std", "min", "max", "count"])
        writer.writeheader()
        for key, stats in summary.items():
            row = {"component": key}
            row.update(stats)
            writer.writerow(row)


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
    summary = run_audit(args)
    write_outputs(summary, Path(args.output_dir))
    print(f"wrote {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
