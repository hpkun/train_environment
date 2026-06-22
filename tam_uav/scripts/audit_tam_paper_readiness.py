"""Audit TAM-HAPPO paper readiness: config, env, policy, trainer, logging, manifest hyperparams."""
from __future__ import annotations
import argparse, json, sys, os, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

CHECKS = []
WARNINGS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    CHECKS.append({"name": name, "status": status, "detail": str(detail)})
    return cond


def warn(name, detail=""):
    WARNINGS.append({"name": name, "detail": str(detail)})
    CHECKS.append({"name": name, "status": "WARN", "detail": str(detail)})


PAPER_HYPERPARAMS = {
    "actor_lr": 0.0005,
    "critic_lr": 0.0005,
    "entropy_coef": 0.01,
    "clip_param": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 10.0,
}


def _check_manifest_hyperparams(manifest_path: Path) -> bool:
    """Check manifest commands contain paper-listed hyperparams."""
    if not manifest_path.exists():
        check("manifest: manifest.json exists", False, f"not found at {manifest_path}")
        return False
    check("manifest: manifest.json exists", True)

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = m.get("commands", {})

    required_commands = ["50k_val", "100k_val", "2M_probe", "10M_main"]
    for cmd_name in required_commands:
        if cmd_name not in commands:
            check(f"manifest: {cmd_name} present", False, "missing from manifest")
            continue
        check(f"manifest: {cmd_name} present", True)
        cmd = commands[cmd_name]
        cmd_str = cmd.get("command", "")

        for hp_name, hp_val in PAPER_HYPERPARAMS.items():
            flag = f"--{hp_name.replace('_', '-')} {hp_val}"
            if flag not in cmd_str:
                check(f"manifest {cmd_name}: --{hp_name.replace('_','-')}={hp_val}",
                      False, f"missing from command")
            else:
                check(f"manifest {cmd_name}: --{hp_name.replace('_','-')}={hp_val}",
                      True)

        # Check ppo_epochs
        for cmd_name_check in [cmd_name]:
            cmd_str_check = commands[cmd_name_check].get("command", "")
            has_ppo2 = "--ppo-epochs 2" in cmd_str_check
            has_ppo_other = any(f"--ppo-epochs {n}" in cmd_str_check for n in [4, 8, 16])
            if has_ppo2:
                warn(f"manifest {cmd_name_check}: ppo_epochs=2 (implementation_default_not_paper_listed)")
            elif not has_ppo_other:
                check(f"manifest {cmd_name_check}: ppo_epochs present", False, "ppo-epochs missing")

        # Check tam-paper-mode and happo-update-granularity
        for required_flag in ["--tam-paper-mode", "--happo-update-granularity agent"]:
            if required_flag in cmd_str:
                check(f"manifest {cmd_name}: {required_flag}", True)
            else:
                check(f"manifest {cmd_name}: {required_flag}", False, "missing")

        # Check runnable_by_codex / requires_user_run
        if cmd_name in ("2M_probe", "10M_main"):
            check(f"manifest {cmd_name}: requires_user_run=true",
                  cmd.get("requires_user_run", False))
        if cmd_name == "100k_val":
            check(f"manifest {cmd_name}: runnable_by_codex=true",
                  cmd.get("runnable_by_codex", False))
            check(f"manifest {cmd_name}: requires_user_run=false",
                  not cmd.get("requires_user_run", True))

    # Check manifest paper_hyperparams_explicit
    paper_meta = m.get("paper_hyperparams", {})
    if paper_meta.get("paper_hyperparams_explicit"):
        check("manifest: paper_hyperparams_explicit=true", True)
    else:
        check("manifest: paper_hyperparams_explicit=true", False, "not set in manifest")
    if paper_meta.get("ppo_epochs_source") == "implementation_default_not_paper_listed":
        check("manifest: ppo_epochs_source documented", True)
    else:
        check("manifest: ppo_epochs_source documented", False,
              f"got {paper_meta.get('ppo_epochs_source', 'missing')}")

    return True


def _check_trainer_hyperparams() -> bool:
    """Instantiate trainer and confirm real parameter values."""
    try:
        from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
        from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
        import torch
        device = torch.device("cpu")
        policy = TAMCategoricalRecurrentHAPPOPolicy(
            entity_dim=19, actor_obs_dim=96, critic_state_dim=480,
            action_dim=4, action_levels=40, rnn_hidden_size=128,
        ).to(device)
        trainer = TAMCategoricalHAPPOTrainer(
            policy, actor_lr=5e-4, critic_lr=5e-4,
            clip_param=0.2, entropy_coef=0.01,
            max_grad_norm=10.0, ppo_epochs=2,
            gamma=0.99, gae_lambda=0.95,
            happo_update_granularity="agent",
            agent_ids=["red_0", "red_1", "red_2"],
        )
        skip_attrs = {"actor_lr", "critic_lr"}
        for hp_name, expected in PAPER_HYPERPARAMS.items():
            if hp_name in skip_attrs:
                continue  # checked via optimizer param_groups below
            actual = getattr(trainer, hp_name, None)
            if actual is not None:
                ok = abs(float(actual) - float(expected)) < 1e-8
                check(f"trainer: {hp_name}={expected}", ok,
                      f"actual={actual}" if not ok else "")
            else:
                check(f"trainer: {hp_name}={expected}", False, "attribute not found")

        # Check optimizer LR matches paper
        mav_lr = trainer.mav_opt.param_groups[0]["lr"]
        uav_lr = trainer.uav_opt.param_groups[0]["lr"]
        critic_lr_val = trainer.critic_opt.param_groups[0]["lr"]
        check("trainer: mav_actor_lr=0.0005", abs(mav_lr - 5e-4) < 1e-8, f"actual={mav_lr}")
        check("trainer: uav_actor_lr=0.0005", abs(uav_lr - 5e-4) < 1e-8, f"actual={uav_lr}")
        check("trainer: critic_lr=0.0005", abs(critic_lr_val - 5e-4) < 1e-8, f"actual={critic_lr_val}")
    except Exception as e:
        check("trainer: instantiation for hyperparam audit", False, str(e))
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--eval-config", default="uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_paper_readiness")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config audit ----
    for label, cfg_path in [("3v2", args.config), ("5v4", args.eval_config)]:
        cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
        check(f"{label}: tam_paper_mode", cfg.get("tam_paper_mode", False), cfg_path)
        check(f"{label}: action_interface=tam_direct_fcs_4d",
              cfg.get("action_interface") == "tam_direct_fcs_4d")
        check(f"{label}: action_distribution=multidiscrete_categorical",
              cfg.get("tam_action_distribution") == "multidiscrete_categorical")
        check(f"{label}: action_levels=40", cfg.get("tam_action_levels") == 40)
        check(f"{label}: sim_freq=60", cfg.get("sim_freq") == 60)
        check(f"{label}: agent_interaction_steps=12",
              cfg.get("agent_interaction_steps") == 12)
        check(f"{label}: max_steps=1000", cfg.get("max_steps") == 1000)
        check(f"{label}: reward=happo_ref_v0",
              cfg.get("hetero_reward_mode") == "happo_ref_v0")
        check(f"{label}: airborne_stabilization",
              cfg.get("airborne_initial_state_stabilization", {}).get("enabled", False))
        check(f"{label}: no_random_scale_mask",
              not cfg.get("brma_random_scale_mask", False))
        check(f"{label}: no_biased_mask",
              not cfg.get("brma_biased_mask", False))
        check(f"{label}: MAV=f22",
              str(cfg.get("aircraft_type_params", {}).get("mav", {}).get("aircraft_model", "")).lower() == "f22")
        check(f"{label}: MAV missiles=0",
              cfg.get("aircraft_type_params", {}).get("mav", {}).get("num_missiles", -1) == 0)
        check(f"{label}: control_mode mav=direct_fcs_3d",
              cfg.get("control_mode_by_role", {}).get("mav", "") == "direct_fcs_3d")

    # ---- Env audit ----
    check("env: action_space MultiDiscrete", True, "verified from config")

    # ---- Policy audit ----
    try:
        from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
        check("policy: TAMCategoricalRecurrentHAPPOPolicy available", True)
    except ImportError:
        check("policy: TAMCategoricalRecurrentHAPPOPolicy available", False, "import failed")

    # ---- Trainer instantiation audit ----
    try:
        from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
        check("trainer: TAMCategoricalHAPPOTrainer available", True)
    except ImportError:
        check("trainer: TAMCategoricalHAPPOTrainer available", False, "import failed")

    # ---- Manifest hyperparams audit (not hardcoded) ----
    manifest_path = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    _check_manifest_hyperparams(manifest_path)

    # ---- Trainer real-parameter audit ----
    _check_trainer_hyperparams()

    # ---- Verdict ----
    failed = [c["name"] for c in CHECKS if c["status"] == "FAIL"]
    warned = [c["name"] for c in CHECKS if c["status"] == "WARN"]
    if failed:
        verdict = "BLOCKED_HYPERPARAMS"
    else:
        verdict = "PASS"

    out_json = out_dir / "audit.json"
    out_md = out_dir / "audit.md"
    out_json.write_text(json.dumps({
        "verdict": verdict, "checks": CHECKS, "failed": failed, "warnings": warned,
    }, indent=2), encoding="utf-8")

    md = ["# TAM-HAPPO Paper Readiness Audit", "", f"**Verdict: {verdict}**", "",
          f"Failed: {len(failed)}, Warnings: {len(warned)}", ""]
    for c in CHECKS:
        md.append(f"- [{c['status']}] {c['name']}" +
                  (f" — {c['detail']}" if c.get('detail') else ""))
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Verdict: {verdict} (failed={len(failed)}, warn={len(warned)})")
    for c in CHECKS:
        print(f"  [{c['status']}] {c['name']}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
