"""Homogeneous 2v2 smoke."""
from __future__ import annotations
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment
spec = ExperimentSpec(
    train_config="uav_env/JSBSim/configs/homo_f16_2v2_brma_rule.yaml",
    eval_configs=["uav_env/JSBSim/configs/homo_f16_2v2_brma_rule.yaml"],
    obs_adapter_version="v2", opponent_policy="brma_rule", actor_arch="mlp",
    total_env_steps=64, rollout_length=16, max_steps=64, eval_episodes=1, device="cpu",
    output_dir="outputs/test_main_mappo_homo_f16_2v2_sanity",
    enable_eval_during_training=True, eval_interval_steps=32, train_eval_episodes=1)
if __name__ == "__main__": run_experiment(spec)
