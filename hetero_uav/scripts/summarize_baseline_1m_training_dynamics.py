"""Summarize 1M baseline training dynamics. Read-only."""
from __future__ import annotations
import json, csv, sys
from pathlib import Path

DEFAULT_DIR = "outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=DEFAULT_DIR)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    if not (exp / "train_log.csv").exists():
        print(f"ERROR: {exp}/train_log.csv not found"); return

    with open(exp/"train_log.csv") as f: train_rows = list(csv.DictReader(f))
    eval_rows = []
    if (exp/"eval_log.csv").exists():
        with open(exp/"eval_log.csv") as f: eval_rows = list(csv.DictReader(f))

    # Training curve analysis
    rets = [float(r["average_team_return"]) for r in train_rows]
    red_alives = [float(r["average_red_alive"]) for r in train_rows]
    ents = [float(r.get("entropy",0)) for r in train_rows]
    sats = [float(r.get("action_saturation_rate",0)) for r in train_rows]

    half = len(train_rows)//2
    quarter3 = 3*len(train_rows)//4
    red_alive_early_zero = all(v == 0.0 for v in red_alives[:10] if float(train_rows[10].get("episodes_completed","0"))>0)
    red_alive_mid = red_alives[half] if half < len(red_alives) else 0
    red_alive_end = red_alives[-1]
    entropy_mid = ents[half] if half < len(ents) else 0
    entropy_end = ents[-1]

    # Learning window detection
    window_rows = [r for r in train_rows if float(r.get("average_red_alive",0)) > 1.0]
    if window_rows:
        window_start = int(window_rows[0]["total_steps"])
        window_end = int(window_rows[-1]["total_steps"])
        window_duration = window_end - window_start
    else:
        window_start = window_end = window_duration = None

    # Collapse detection
    collapse_start = None
    for i in range(len(red_alives)-2, 0, -1):
        if red_alives[i] > 0.5 and red_alives[i+1] < 0.1:
            collapse_start = int(train_rows[i+1]["total_steps"])
            break

    # Best eval
    best_eval = None
    for r in eval_rows:
        s = float(r.get("red_win_rate",0)) + 0.1*float(r.get("mav_survival_rate",0)) + 0.01*float(r.get("avg_return",0))
        if best_eval is None or s > best_eval["score"]:
            best_eval = dict(step=int(r["total_steps"]), iteration=int(r["iteration"]), score=round(s,4),
                             cfg=r.get("eval_config","").split("/")[-1],
                             red_win=float(r.get("red_win_rate",0)), blue_win=float(r.get("blue_win_rate",0)),
                             ret=float(r.get("avg_return",0)))

    learning_window_confirmed = window_start is not None
    catastrophic_forgetting = collapse_start is not None
    best_is_better = red_alives[half] > red_alives[-1] + 0.5

    data = dict(
        learning_window=dict(confirmed=learning_window_confirmed, start_step=window_start, end_step=window_end,
                             duration_steps=window_duration),
        collapse_detection=dict(confirmed=catastrophic_forgetting, collapse_step=collapse_start),
        best_eval=best_eval,
        train_curve=dict(early_red_alive_zero=red_alive_early_zero, mid_red_alive=round(red_alive_mid,2),
                         end_red_alive=round(red_alive_end,2), mid_entropy=round(entropy_mid,2),
                         end_entropy=round(entropy_end,2)),
        recommendations=["Re-evaluate best checkpoint with >=50 episodes",
                         "If best is valid, run multi-seed with fixed best-checkpoint selection",
                         "If best is still weak, add entropy schedule or early stopping"])
    md = [f"# 1M Training Dynamics", "", f"## Learning Window", f"- confirmed: {learning_window_confirmed}",
          f"- start: {window_start}, end: {window_end}, duration: {window_duration}",
          f"## Collapse", f"- confirmed: {catastrophic_forgetting}, step: {collapse_start}",
          f"## Train Curve", f"- mid red_alive: {red_alive_mid:.2f}, end: {red_alive_end:.2f}",
          f"- mid entropy: {entropy_mid:.2f}, end: {entropy_end:.2f}",
          f"## Best Eval", f"- step={best_eval['step'] if best_eval else 'N/A'} red_win={best_eval['red_win'] if best_eval else 'N/A'}",
          f"## Recommendations"]
    for r_ in data["recommendations"]: md.append(f"- {r_}")

    out_j = args.output_json or str(exp/"baseline_1m_training_dynamics.json")
    out_m = args.output_md or str(exp/"baseline_1m_training_dynamics.md")
    Path(out_j).parent.mkdir(parents=True, exist_ok=True)
    Path(out_j).write_text(json.dumps(data,indent=2))
    Path(out_m).write_text("\n".join(md))
    print(f"output_json: {out_j}"); print(f"output_md: {out_m}")
    print(f"learning_window: {learning_window_confirmed} collapse: {catastrophic_forgetting}")

if __name__ == "__main__": main()
