"""1M shared MLP MAPPO baseline runner."""
from __future__ import annotations
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_main_mappo_experiment import ExperimentSpec, run_experiment
spec = ExperimentSpec(
    train_config="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
    eval_configs=[
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    ],
    obs_adapter_version="v2", opponent_policy="brma_rule", actor_arch="mlp",
    total_env_steps=1000000, rollout_length=512, max_steps=1000, eval_episodes=20,
    device="cuda", output_dir="outputs/main_mappo_baseline_1m_fast_brma_rule_no_mav_trim",
    enable_eval_during_training=True, eval_interval_steps=100000, train_eval_episodes=5)
if __name__ == "__main__": run_experiment(spec)
