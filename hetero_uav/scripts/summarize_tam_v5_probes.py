from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = []
    for run in sorted(Path(args.outputs_root).glob("tam_v5_*_20k_probe_s*")):
        log = run / "train_log.csv"
        if not log.exists():
            continue
        frame = pd.read_csv(log); last = frame.iloc[-1]
        name = run.name.split("_20k_probe_s", 1)[0].replace("tam_v5_", "")
        seed = int(run.name.rsplit("s", 1)[1])
        ep_path = run / "rich_logs" / "episode_reward_components.csv"
        episodes = pd.read_csv(ep_path) if ep_path.exists() else pd.DataFrame()
        rows.append({"candidate": name, "seed": seed, "total_steps": int(last.total_steps),
                     "avg_return": float(last.avg_return), "red_win": float(last.red_win),
                     "blue_win": float(last.blue_win), "timeout": float(last.timeout),
                     "mav_survival": float(last.mav_survival), "red_alive_final": float(last.red_alive_final),
                     "blue_alive_final": float(last.blue_alive_final),
                     "red_launch": float(last.red_missiles_fired), "red_hit": float(last.red_missile_hits),
                     "episode_rows": int(len(episodes)), "nan_detected": int(last.nan_detected),
                     "identity_max": float(last.get("v5_identity_max_abs", 0.0)),
                     "runner_status": "normal" if (run / "latest/model.pt").exists() else "incomplete"})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows); data.to_csv(output / "v5_probe_per_seed.csv", index=False)
    numeric = [column for column in data.columns if column not in {"candidate", "runner_status"}]
    summary = data.groupby("candidate")[numeric].agg(["mean", "std"]) if not data.empty else pd.DataFrame()
    summary.to_csv(output / "v5_probe_mean_std.csv")
    ready = bool(not data.empty and (data.nan_detected == 0).all() and (data.identity_max <= 1e-8).all()
                 and (data.runner_status == "normal").all()
                 and ((data.red_launch > 0).any() or (data.blue_alive_final < 2).any()))
    status = "TAM_HAPPO_PAPER_FORMULA_V5_READY_FOR_200K_PROBE" if ready else "TAM_HAPPO_PAPER_FORMULA_V5_NOT_READY"
    (output / "v5_probe_summary.json").write_text(json.dumps({"status": status, "runs": len(data)}, indent=2), encoding="utf-8")
    (output / "v5_probe_summary.md").write_text("# TAM v5 probe summary\n\n" + f"`{status}`\n\n" + summary.to_markdown(), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
