"""Diagnose team-average vs per-agent advantage signal dilution.

Read-only: does not modify loss, reward, or training.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="Optional previous training output dir")
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_advantage_signal_diagnostics")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Try to load an existing training output to analyze advantage signals
    run_dir = ROOT / args.run_dir if args.run_dir else None
    train_log = run_dir / "train_log.csv" if run_dir and run_dir.exists() else None

    report = {
        "test": "advantage_signal_diagnostic",
        "mode": "read_only_diagnostic",
        "note": "Compares team-average vs per-agent advantage. Does not modify loss.",
        "analysis": {},
    }

    if train_log and train_log.exists():
        import csv
        with train_log.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        adv_mav = []
        adv_uav = []
        returns = []
        for r in rows:
            ret = float(r.get("avg_return", 0))
            returns.append(ret)

        if returns:
            report["analysis"]["avg_return_mean"] = float(np.mean(returns))
            report["analysis"]["avg_return_std"] = float(np.std(returns))

        # Simulate team vs per-agent advantage on a hypothetical death episode
        # MAV death penalty = -4.0, UAV receive 0
        num_red = 3
        mav_death_reward = -4.0
        team_reward = mav_death_reward / num_red  # diluted
        per_agent_reward = mav_death_reward  # undiluted

        # With gamma=0.99, death at step t contributes GAE ~ reward
        team_gae_approx = team_reward  # simplified
        per_agent_gae_approx = per_agent_reward

        dilution_ratio = abs(team_gae_approx) / max(abs(per_agent_gae_approx), 1e-8)

        report["analysis"]["team_average_advantage_death_transition_approx"] = team_gae_approx
        report["analysis"]["per_agent_advantage_death_transition_approx"] = per_agent_gae_approx
        report["analysis"]["dilution_ratio_abs"] = dilution_ratio
        report["analysis"]["whether_diluted"] = dilution_ratio < 0.5
    else:
        # Synthetic analysis
        num_red = 3
        mav_death_reward = -4.0
        team_reward = mav_death_reward / num_red
        per_agent_reward = mav_death_reward
        dilution_ratio = abs(team_reward) / max(abs(per_agent_reward), 1e-8)

        report["analysis"] = {
            "team_average_advantage_death_transition_approx": team_reward,
            "per_agent_advantage_death_transition_approx": per_agent_reward,
            "dilution_ratio_abs": dilution_ratio,
            "whether_current_team_average_likely_dilutes_mav_death_signal": dilution_ratio < 0.5,
            "note": "Team-level reward divides MAV death penalty by 3 agents, reducing advantage signal.",
        }

    (out_dir / "advantage_signal.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Advantage Signal Diagnostic", "",
          f"team_average_death_adv: {report['analysis'].get('team_average_advantage_death_transition_approx', 0):.4f}",
          f"per_agent_death_adv: {report['analysis'].get('per_agent_advantage_death_transition_approx', 0):.4f}",
          f"dilution_ratio_abs: {report['analysis'].get('dilution_ratio_abs', 0):.4f}",
          f"diluted: {report['analysis'].get('whether_current_team_average_likely_dilutes_mav_death_signal', 'N/A')}",
    ]
    (out_dir / "advantage_signal.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
