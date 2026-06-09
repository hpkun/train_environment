"""role_conditioned MAPPO — 50k pilot."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment
spec = ExperimentSpec(
    train_config="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    eval_configs=[
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    ],
    obs_adapter_version="v2", opponent_policy="rule_nearest",
    actor_arch="role_conditioned",
    total_env_steps=50000, rollout_length=128, max_steps=1000,
    eval_episodes=20, device="cpu",
    output_dir="outputs/main_mappo_experiment_f22_50k_role_conditioned",
    enable_eval_during_training=True,
    eval_interval_steps=25000, train_eval_episodes=5,
)
if __name__ == "__main__":
    run_experiment(spec)
