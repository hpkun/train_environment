"""Generate paper-mode run manifest with complete paper-listed hyperparameters."""
from __future__ import annotations
import json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPER_HYPERPARAMS = (
    "--actor-lr 0.0005 --critic-lr 0.0005 --entropy-coef 0.01 "
    "--clip-param 0.2 --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 10.0 "
    "--ppo-epochs 2"
)

SHARED = (
    f"{PAPER_HYPERPARAMS} "
    "--rollout-length 256 --num-envs 1 --max-steps 1000 --device cuda "
    "--policy-arch tam_categorical_recurrent --opponent-policy tam_direct_fsm "
    "--reward-mode happo_ref_v0 --tam-paper-mode --happo-update-granularity agent "
    "--eval-during-training --eval-at-start --train-eval-episodes 5 "
    "--save-eval-checkpoints --eval-checkpoint-metric combined --keep-eval-checkpoints 30 "
    "--enable-rich-logging"
)

PAPER_HYPERPARAMS_META = {
    "actor_lr": 0.0005,
    "critic_lr": 0.0005,
    "entropy_coef": 0.01,
    "clip_param": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 10.0,
    "ppo_epochs": 2,
    "ppo_epochs_source": "implementation_default_not_paper_listed",
    "paper_hyperparams_explicit": True,
}


def _cmd(config, eval_cfg, output_dir, total_steps, eval_interval, rich_log, heartbeat,
         extra_flags=""):
    flags = (
        f"python -u scripts/train_tam_happo_direct.py "
        f"--config {config} --output-dir {output_dir} "
        f"--total-env-steps {total_steps} --eval-interval-steps {eval_interval} "
        f"{SHARED} "
        f"--eval-configs {config} {eval_cfg} "
        f"--rich-log-dir {rich_log} --heartbeat-log {heartbeat}"
    )
    if extra_flags:
        flags += " " + extra_flags
    return flags


def _entry(name, steps, eval_interval, out_dir, runnable, requires_user, extra=""):
    return {
        "name": name,
        "total_env_steps": steps,
        "eval_interval_steps": eval_interval,
        "output_dir": out_dir,
        "runnable_by_codex": runnable,
        "requires_user_run": requires_user,
        "command": _cmd(
            "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
            "uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml",
            out_dir, steps, eval_interval,
            f"{out_dir}/rich_logs", f"{out_dir}/heartbeat.log",
            extra_flags=extra,
        ),
        **PAPER_HYPERPARAMS_META,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--eval-config", default="uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_paper_run_manifest")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "preconditions": [
            "strict audit PASS", "2k smoke PASS", "8k sanity PASS",
            "fixed F22 stability PASS", "train_log has agent_metrics_json",
            "eval_log has red_uav_fired_mean/red_uav_hits_mean",
            "no paper-readiness blocker",
        ],
        "paper_hyperparams": PAPER_HYPERPARAMS_META,
        "commands": {
            "50k_val": _entry("50k_val", 50000, 25000,
                             "outputs/tam_papermode_3v2_50k_val",
                             True, False),
            "100k_val": _entry("100k_val", 100000, 25000,
                              "outputs/tam_papermode_paperhparams_3v2_100k_val",
                              True, False,
                              extra="--heartbeat-stall-timeout-sec 1800 --exit-on-heartbeat-stall"),
            "2M_probe": _entry("2M_probe", 2000000, 50000,
                              "outputs/tam_papermode_3v2_2M_probe",
                              False, True,
                              extra="--heartbeat-stall-timeout-sec 1800 --exit-on-heartbeat-stall"),
            "10M_main": _entry("10M_main", 10000000, 100000,
                              "outputs/tam_papermode_3v2_10M_main",
                              False, True,
                              extra="--heartbeat-stall-timeout-sec 1800 --exit-on-heartbeat-stall"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md = ["# TAM-HAPPO Paper Run Manifest", "",
          "Paper-listed hyperparameters are explicitly included in all commands:", ""]
    for k, v in PAPER_HYPERPARAMS_META.items():
        md.append(f"- `{k}`: {v}")
    md.append("")
    for name, c in manifest["commands"].items():
        md.append(f"## {name} ({c['total_env_steps']} steps, "
                  f"runnable_by_codex={c['runnable_by_codex']}, "
                  f"requires_user={c['requires_user_run']})")
        md.append(f"```bash\n{c['command']}\n```")
        md.append("")
    md.append("## Preconditions")
    for pc in manifest["preconditions"]:
        md.append(f"- [ ] {pc}")
    (out_dir / "manifest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir}/manifest.json, manifest.md")


if __name__ == "__main__":
    main()
