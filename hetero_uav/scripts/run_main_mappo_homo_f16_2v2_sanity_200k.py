"""Homogeneous 2v2 F-16 sanity baseline runner."""
from __future__ import annotations
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment
spec = ExperimentSpec(
    train_config="uav_env/JSBSim/configs/homo_f16_2v2_brma_rule.yaml",
    eval_configs=["uav_env/JSBSim/configs/homo_f16_2v2_brma_rule.yaml"],
    obs_adapter_version="v2", opponent_policy="brma_rule", actor_arch="mlp",
    total_env_steps=200000, rollout_length=256, max_steps=1000, eval_episodes=20, device="cpu",
    output_dir="outputs/main_mappo_homo_f16_2v2_sanity_200k",
    enable_eval_during_training=True, eval_interval_steps=25000, train_eval_episodes=5)
if __name__ == "__main__": run_experiment(spec)
