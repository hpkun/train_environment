"""Summarize protocol-aligned training results."""
from __future__ import annotations
import json, csv, sys
from pathlib import Path

def summarize(exp_dir):
    p = Path(exp_dir)
    if not p.exists(): return {"error": f"{exp_dir} not found"}
    train_csv = p / "train_log.csv"; eval_csv = p / "eval_log.csv"
    if not train_csv.exists(): return {"error": "train_log.csv not found"}

    with open(train_csv) as f: rows = list(csv.DictReader(f))
    first10 = [r for r in rows if float(r.get("average_episode_length",0)) > 0][:10]
    last10 = [r for r in rows if float(r.get("average_episode_length",0)) > 0][-10:]

    red_alive_first = [float(r["average_red_alive"]) for r in first10] if first10 else []
    red_alive_last = [float(r["average_red_alive"]) for r in last10] if last10 else []
    ret_first = [float(r["average_team_return"]) for r in first10] if first10 else []
    ret_last = [float(r["average_team_return"]) for r in last10] if last10 else []
    last_row = rows[-1]

    eval_recs = []
    if eval_csv.exists():
        with open(eval_csv) as f:
            for r in csv.DictReader(f): eval_recs.append(r)

    summary_json = p / "main_experiment_summary.json"
    final_eval = json.loads(summary_json.read_text()) if summary_json.exists() else []

    best_meta = p / "best" / "meta.json"
    best = json.loads(best_meta.read_text()) if best_meta.exists() else {}

    return dict(
        train=dict(first10_red_alive_mean=round(sum(red_alive_first)/len(red_alive_first),2) if red_alive_first else 0,
                   last10_red_alive_mean=round(sum(red_alive_last)/len(red_alive_last),2) if red_alive_last else 0,
                   first10_ret_mean=round(sum(ret_first)/len(ret_first),2) if ret_first else 0,
                   last10_ret_mean=round(sum(ret_last)/len(ret_last),2) if ret_last else 0,
                   final_red_alive=float(last_row.get("average_red_alive",0)),
                   final_blue_alive=float(last_row.get("average_blue_alive",0)),
                   final_ret=float(last_row.get("average_team_return",0)),
                   action_sat=float(last_row.get("action_saturation_rate",0)),
                   entropy=float(last_row.get("entropy",0))),
        eval=dict(records=[dict(steps=r.get("total_steps",""), cfg=r.get("eval_config","").split("/")[-1] if r.get("eval_config") else "", red_win=float(r.get("red_win_rate",0)), blue_elim=float(r.get("blue_elimination_win_rate",0)), mav_surv=float(r.get("mav_survival_rate",0))) for r in eval_recs]),
        final_eval=[dict(cfg=r["eval_config"].split("/")[-1] if r.get("eval_config") else "", red_win=r.get("red_win_rate",0), blue_elim=r.get("blue_elimination_win_rate",0), red_to_adv=r.get("red_timeout_alive_advantage_rate",0)) for r in final_eval],
        best_checkpoint=dict(step=best.get("total_steps"), score=best.get("best_score")),
        conclusion="protocol-aligned summary"
    )

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default="outputs/main_mappo_experiment_protocol_aligned_brma_rule_no_mav_trim_200k")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    args = p.parse_args()
    exp_dir = args.experiment_dir
    out_j = args.output_json or f"{exp_dir}/protocol_aligned_training_summary.json"
    out_m = args.output_md or f"{exp_dir}/protocol_aligned_training_summary.md"
    data = summarize(exp_dir)
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return
    Path(out_j).parent.mkdir(parents=True, exist_ok=True)
    Path(out_j).write_text(json.dumps(data, indent=2))
    t = data["train"]; e = data["eval"]
    md = ["# Protocol-Aligned Training Summary", "", "## Train", f"- first10 red_alive: {t['first10_red_alive_mean']}", f"- last10 red_alive: {t['last10_red_alive_mean']}", f"- first10 ret: {t['first10_ret_mean']}", f"- last10 ret: {t['last10_ret_mean']}", f"- final red_alive: {t['final_red_alive']}", f"- final ret: {t['final_ret']}", f"- action_sat: {t['action_sat']}", f"- entropy: {t['entropy']}", "", "## Best Checkpoint", f"- step={data['best_checkpoint']['step']} score={data['best_checkpoint']['score']}"]
    Path(out_m).write_text("\n".join(md))
    print(f"output_json: {out_j}"); print(f"output_md: {out_m}")

if __name__ == "__main__": main()
