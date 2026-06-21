"""Generate paper-mode run manifest with complete parameters."""
from __future__ import annotations
import json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHARED = (
    "--rollout-length 256 --num-envs 1 --max-steps 1000 --device cuda "
    "--policy-arch tam_categorical_recurrent --opponent-policy tam_direct_fsm "
    "--reward-mode happo_ref_v0 --tam-paper-mode --happo-update-granularity agent "
    "--eval-during-training --eval-at-start --train-eval-episodes 5 "
    "--save-eval-checkpoints --eval-checkpoint-metric combined --keep-eval-checkpoints 30 "
    "--enable-rich-logging"
)

def _cmd(config, eval_cfg, output_dir, total_steps, eval_interval, rich_log, heartbeat):
    return (
        f"python -u scripts/train_tam_happo_direct.py "
        f"--config {config} --output-dir {output_dir} "
        f"--total-env-steps {total_steps} --eval-interval-steps {eval_interval} "
        f"{SHARED} "
        f"--eval-configs {config} {eval_cfg} "
        f"--rich-log-dir {rich_log} --heartbeat-log {heartbeat}"
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--eval-config", default="uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_paper_run_manifest")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = args.config; evcfg = args.eval_config
    manifest = {
        "preconditions": [
            "strict audit PASS", "2k smoke PASS", "8k sanity PASS",
            "fixed F22 stability PASS", "train_log has agent_metrics_json",
            "eval_log has red_uav_fired_mean/red_uav_hits_mean",
            "no paper-readiness blocker",
        ],
        "commands": {
            "50k_val": {
                "total_env_steps": 50000,
                "runnable_by_codex": True, "requires_user_run": False,
                "command": _cmd(cfg, evcfg, "outputs/tam_papermode_3v2_50k_val", 50000, 25000,
                                "outputs/tam_papermode_3v2_50k_val/rich_logs",
                                "outputs/tam_papermode_3v2_50k_val/heartbeat.log"),
            },
            "2M_probe": {
                "total_env_steps": 2000000,
                "runnable_by_codex": False, "requires_user_run": True,
                "command": _cmd(cfg, evcfg, "outputs/tam_papermode_3v2_2M_probe", 2000000, 50000,
                                "outputs/tam_papermode_3v2_2M_probe/rich_logs",
                                "outputs/tam_papermode_3v2_2M_probe/heartbeat.log"),
            },
            "10M_main": {
                "total_env_steps": 10000000,
                "runnable_by_codex": False, "requires_user_run": True,
                "command": _cmd(cfg, evcfg, "outputs/tam_papermode_3v2_10M_main", 10000000, 100000,
                                "outputs/tam_papermode_3v2_10M_main/rich_logs",
                                "outputs/tam_papermode_3v2_10M_main/heartbeat.log"),
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md = ["# TAM-HAPPO Paper Run Manifest", ""]
    for name, c in manifest["commands"].items():
        md.append(f"## {name} ({c['total_env_steps']} steps, requires_user={c['requires_user_run']})")
        md.append(f"```bash\n{c['command']}\n```")
        md.append("")
    md.append("## Preconditions")
    for pc in manifest["preconditions"]:
        md.append(f"- [ ] {pc}")
    (out_dir / "manifest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir}/manifest.json, manifest.md")

if __name__ == "__main__":
    main()
