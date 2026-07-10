"""Short rollout audit for brma_tam_scripted_composite_v1.

This script only rolls out fixed actions and summarizes reward diagnostics. It
does not train, tune weights, or alter environment dynamics.
"""
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

from uav_env import make_env  # noqa: E402


def _zero_actions(env) -> dict:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _sum_components(rows: list[dict], prefix: str) -> float:
    total = 0.0
    for row in rows:
        for key, value in row.items():
            if key.startswith(prefix):
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    pass
    return total


def run_audit(args) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.config, max_steps=args.max_steps)
    rows: list[dict] = []
    target_rows: list[dict] = []
    try:
        for ep in range(args.episodes):
            _obs, _info = env.reset(seed=args.seed + ep)
            for step in range(args.max_steps):
                obs, rewards, terminated, truncated, info = env.step(_zero_actions(env))
                comps = info.get("reward_components", {}) if isinstance(info, dict) else {}
                for aid in env.red_ids:
                    comp = comps.get(aid, {}) if isinstance(comps, dict) else {}
                    row = {
                        "episode_id": ep,
                        "step": step,
                        "agent_id": aid,
                        "role": env.agent_roles.get(aid, ""),
                        "reward": float(rewards.get(aid, 0.0)),
                    }
                    for key, value in comp.items():
                        try:
                            row[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    rows.append(row)
                for rec in info.get("__reward_target_diagnostics__", []) or []:
                    rec = dict(rec)
                    rec.setdefault("episode_id", ep)
                    rec.setdefault("step", step)
                    target_rows.append(rec)
                if all(terminated.values()) or all(truncated.values()):
                    break
    finally:
        env.close()

    component_csv = out_dir / "brma_tam_scripted_composite_v1_components.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with component_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    target_csv = out_dir / "reward_target_diagnostics.csv"
    if target_rows:
        fieldnames = sorted({key for row in target_rows for key in row})
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(target_rows)

    summary = {
        "episodes": int(args.episodes),
        "steps_recorded": len(rows),
        "reward_total_sum": sum(float(row.get("reward", 0.0)) for row in rows),
        "uav_total_sum": _sum_components(rows, "uav_total"),
        "mav_total_sum": _sum_components(rows, "mav_total"),
        "uav_event_kill_sum": _sum_components(rows, "uav_event_kill"),
        "uav_event_loss_sum": _sum_components(rows, "uav_event_loss"),
        "mav_team_credit_delta_sum": _sum_components(rows, "mav_team_credit_delta"),
        "target_diagnostic_rows": len(target_rows),
        "note": "Zero-action rollout audit only; this is not learned-policy performance.",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "audit_summary.md").write_text(
        "# BRMA/TAM scripted composite v1 audit\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/audit_brma_tam_scripted_composite_v1")
    args = parser.parse_args()
    summary = run_audit(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
