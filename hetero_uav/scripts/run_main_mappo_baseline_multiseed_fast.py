"""Multi-seed 1M baseline runner."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[0,1,2])
    p.add_argument("--total-env-steps", type=int, default=1000000)
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for seed in args.seeds:
        out_dir = f"outputs/main_mappo_baseline_multiseed_fast/seed_{seed}"
        if args.dry_run:
            print(f"[dry-run] seed={seed} output={out_dir} steps={args.total_env_steps}")
            continue
        print(f"Running seed={seed}...")
        spec = ExperimentSpec(
            train_config="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
            eval_configs=["uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
                          "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"],
            obs_adapter_version="v2", opponent_policy="brma_rule", actor_arch="mlp",
            total_env_steps=args.total_env_steps, rollout_length=512, max_steps=1000, eval_episodes=20,
            seed=seed, device="cuda", output_dir=out_dir,
            enable_eval_during_training=True, eval_interval_steps=100000, train_eval_episodes=5)
        run_experiment(spec)
if __name__ == "__main__": main()
