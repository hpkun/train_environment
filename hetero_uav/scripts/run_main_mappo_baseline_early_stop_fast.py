"""Early-stop 1M baseline runner."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-env-steps", type=int, default=1000000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out_dir = f"outputs/main_mappo_baseline_early_stop_fast/seed_{args.seed}"
    if args.dry_run:
        print(f"[dry-run] early-stop seed={args.seed} output={out_dir} steps={args.total_env_steps}")
        return
    spec = ExperimentSpec(
        train_config="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
        eval_configs=["uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
                      "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"],
        obs_adapter_version="v2", opponent_policy="brma_rule", actor_arch="mlp",
        total_env_steps=args.total_env_steps, rollout_length=512, max_steps=1000, eval_episodes=20,
        seed=args.seed, device="cuda", output_dir=out_dir,
        enable_eval_during_training=True, eval_interval_steps=50000, train_eval_episodes=5)
    run_experiment(spec)

if __name__ == "__main__": main()
