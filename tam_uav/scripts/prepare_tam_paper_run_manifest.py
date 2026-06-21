"""Generate paper-mode run manifest: commands + preconditions."""
from __future__ import annotations
import json, argparse, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--eval-config", default="uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_paper_run_manifest")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    BASE = "python -u scripts/train_tam_happo_direct.py"
    OPTS = ("--rollout-length 256 --num-envs 1 --max-steps 1000 --device cuda "
            "--policy-arch tam_categorical_recurrent --opponent-policy tam_direct_fsm "
            "--reward-mode happo_ref_v0 --tam-paper-mode "
            "--eval-during-training --eval-interval-steps 50000 --train-eval-episodes 5 "
            "--save-eval-checkpoints --eval-checkpoint-metric combined --keep-eval-checkpoints 30 "
            "--enable-rich-logging "
            "--eval-configs {cfg3v2} {cfg5v4}")

    manifest = {
        "commands": {
            "50k_engineering_validation": f"{BASE} --config {args.config} --output-dir outputs/tam_papermode_3v2_50k_val --total-env-steps 50000 --rich-log-dir outputs/tam_papermode_3v2_50k_val/rich_logs --heartbeat-log outputs/tam_papermode_3v2_50k_val/heartbeat.log",
            "2M_probe": f"{BASE} --config {args.config} --output-dir outputs/tam_papermode_3v2_2M_probe --total-env-steps 2000000 --rich-log-dir outputs/tam_papermode_3v2_2M_probe/rich_logs --heartbeat-log outputs/tam_papermode_3v2_2M_probe/heartbeat.log",
            "10M_main": f"{BASE} --config {args.config} --output-dir outputs/tam_papermode_3v2_10M_main --total-env-steps 10000000 --rich-log-dir outputs/tam_papermode_3v2_10M_main/rich_logs --heartbeat-log outputs/tam_papermode_3v2_10M_main/heartbeat.log",
            "5v4_zero_shot_eval": f"python -u scripts/eval_tam_happo_direct.py --model outputs/tam_papermode_3v2_10M_main/best_combined/model.pt --configs {args.config} {args.eval_config} --episodes 50",
        },
        "preconditions": [
            "strict audit PASS",
            "2k smoke PASS",
            "8k sanity PASS",
            "fixed F22 stability PASS",
            "train_log has agent_metrics_json",
            "eval_log has red_uav_fired_mean/red_uav_hits_mean",
            "no paper-readiness blocker",
        ],
        "eval_interval_steps": 50000,
        "checkpoint_interval": "every eval (50k steps)",
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md = ["# TAM-HAPPO Paper Run Manifest", ""]
    for k, v in manifest["commands"].items():
        md.append(f"## {k}")
        md.append(f"```bash\n{v}\n```")
        md.append("")
    md.append("## Preconditions")
    for c in manifest["preconditions"]:
        md.append(f"- [ ] {c}")
    (out_dir / "manifest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir}/manifest.json, manifest.md")

if __name__ == "__main__":
    main()
