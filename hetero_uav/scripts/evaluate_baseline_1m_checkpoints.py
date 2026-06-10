"""Strict re-evaluation of 1M baseline best/latest checkpoints."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import subprocess

DEFAULT_DIR = "outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim"
EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]

def eval_checkpoint(model_path, episodes, device="cpu"):
    out_json = str(ROOT / "outputs" / "_tmp_ckpt_eval.json")
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model", model_path, "--obs-adapter-version", "v2",
        "--episodes", str(episodes), "--device", device,
        "--opponent-policy", "brma_rule", "--configs", *EVAL_CONFIGS,
        "--summary-json", out_json,
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=7200)
    if r.returncode != 0:
        print(f"EVAL FAILED: {r.stderr[-500:]}", flush=True)
        return None
    return json.loads(Path(out_json).read_text(encoding="utf-8"))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=DEFAULT_DIR)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-mode", choices=["both", "best_only"], default="both")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    suffix = "_best_only" if args.checkpoint_mode == "best_only" else ""
    out_j = args.output_json or str(exp / f"checkpoint_eval/baseline_1m_checkpoint_eval{suffix}.json")
    out_m = args.output_md or str(exp / f"checkpoint_eval/baseline_1m_checkpoint_eval{suffix}.md")
    out_j_p = Path(out_j); out_j_p.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    if args.checkpoint_mode in ("both", "best_only"):
        checkpoints.append(("best", str(exp / "best/model.pt")))
    if args.checkpoint_mode == "both":
        checkpoints.append(("latest", str(exp / "latest/model.pt")))

    results = {}
    for name, pth in checkpoints:
        print(f"Evaluating {name} ({pth})...")
        data = eval_checkpoint(pth, args.episodes, args.device)
        results[name] = data if data else "EVAL_FAILED"

    md = [f"# 1M Baseline Checkpoint Re-Evaluation ({args.episodes} eps)", ""]
    for name in ["best", "latest"]:
        data = results.get(name, "EVAL_FAILED")
        md.append(f"## {name}")
        if isinstance(data, str):
            md.append(f"- ERROR: {data}")
        else:
            for rec in data:
                cfg = rec.get("config","").split("/")[-1]
                md.append(f"### {cfg}")
                for k in ["avg_return","avg_length","red_win_rate","blue_win_rate","draw_rate","timeout_rate",
                          "red_elimination_win_rate","blue_elimination_win_rate","mav_survival_rate",
                          "red_alive_final_mean","blue_alive_final_mean","red_dead_final_mean","blue_dead_final_mean","kill_death_ratio"]:
                    md.append(f"- {k}: {rec.get(k, '?')}")
        md.append("")

    out_j_p.write_text(json.dumps(results, indent=2))
    Path(out_m).write_text("\n".join(md))
    print(f"output_json: {out_j}"); print(f"output_md: {out_m}")

if __name__ == "__main__": main()
