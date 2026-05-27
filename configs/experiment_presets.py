"""Experiment presets — short names for common training configurations.

Usage:
    python train_vanilla_mappo.py --preset vanilla_2v2_smoke
    python train_vanilla_mappo.py --preset vanilla_2v2_smoke --total-env-steps 2000

CLI arguments take precedence over preset values.
"""
from __future__ import annotations

EXPERIMENT_PRESETS = {
    # ---- vanilla smoke ----
    "vanilla_1v1_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "log_file": "logs/vanilla_1v1_smoke.csv",
        "results_file": "results/vanilla_1v1_smoke_results.csv",
        "checkpoint_dir": "checkpoints/vanilla_1v1_smoke",
    },
    "vanilla_2v2_smoke": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 2,
        "total_env_steps": 10000,
        "replay_buffer_size": 100,
        "max_episode_length": 300,
        "device": "cpu",
        "log_file": "logs/vanilla_2v2_smoke.csv",
        "results_file": "results/vanilla_2v2_smoke_results.csv",
        "checkpoint_dir": "checkpoints/vanilla_2v2_smoke",
    },
    # ---- vanilla main ----
    "vanilla_2v2_main": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 8,
        "total_env_steps": 10_000_000,
        "replay_buffer_size": 2000,
        "max_episode_length": 1400,
        "device": "auto",
        "log_file": "logs/vanilla_2v2_main.csv",
        "results_file": "results/vanilla_2v2_main_results.csv",
        "checkpoint_dir": "checkpoints/vanilla_2v2_main",
    },
    # ---- attention smoke ----
    "attention_1v1_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "current",
        "log_file": "logs/attention_1v1_smoke.csv",
        "results_file": "results/attention_1v1_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_smoke",
    },
    "attention_2v2_current_smoke": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 2,
        "total_env_steps": 10000,
        "replay_buffer_size": 100,
        "max_episode_length": 300,
        "device": "cpu",
        "obs_adapter": "current",
        "log_file": "logs/attention_2v2_current_smoke.csv",
        "results_file": "results/attention_2v2_current_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_2v2_current_smoke",
    },
    "attention_2v2_placeholder_smoke": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 2,
        "total_env_steps": 10000,
        "replay_buffer_size": 100,
        "max_episode_length": 300,
        "device": "cpu",
        "obs_adapter": "paper-placeholder",
        "log_file": "logs/attention_2v2_placeholder_smoke.csv",
        "results_file": "results/attention_2v2_placeholder_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_2v2_placeholder_smoke",
    },
    "attention_1v1_strict_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "strict",
        "log_file": "logs/attention_1v1_strict_smoke.csv",
        "results_file": "results/attention_1v1_strict_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_strict_smoke",
    },
    "attention_2v2_strict_smoke": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 2,
        "total_env_steps": 10000,
        "replay_buffer_size": 100,
        "max_episode_length": 300,
        "device": "cpu",
        "obs_adapter": "strict",
        "log_file": "logs/attention_2v2_strict_smoke.csv",
        "results_file": "results/attention_2v2_strict_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_2v2_strict_smoke",
    },
    # ---- attention strict + strict-global critic ----
    "attention_1v1_strict_critic_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "strict",
        "critic_state": "strict-global",
        "log_file": "logs/attention_1v1_strict_critic_smoke.csv",
        "results_file": "results/attention_1v1_strict_critic_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_strict_critic_smoke",
    },
    "attention_2v2_strict_critic_smoke": {
        "num_red": 2,
        "num_blue": 2,
        "num_envs": 2,
        "total_env_steps": 10000,
        "replay_buffer_size": 100,
        "max_episode_length": 300,
        "device": "cpu",
        "obs_adapter": "strict",
        "critic_state": "strict-global",
        "log_file": "logs/attention_2v2_strict_critic_smoke.csv",
        "results_file": "results/attention_2v2_strict_critic_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_2v2_strict_critic_smoke",
    },
    # ---- attention strict + paper eq.33 encoder ----
    "attention_1v1_strict_eq33_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "strict",
        "encoder_mode": "paper-eq33",
        "log_file": "logs/attention_1v1_strict_eq33_smoke.csv",
        "results_file": "results/attention_1v1_strict_eq33_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_strict_eq33_smoke",
    },
    "attention_1v1_strict_eq33_attncritic_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "strict",
        "encoder_mode": "paper-eq33",
        "critic_state": "attention-entities",
        "log_file": "logs/attention_1v1_strict_eq33_attncritic_smoke.csv",
        "results_file": "results/attention_1v1_strict_eq33_attncritic_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_strict_eq33_attncritic_smoke",
    },
    "attention_1v1_strict_eq33_attncritic_brma_dryrun_smoke": {
        "num_red": 1,
        "num_blue": 1,
        "num_envs": 1,
        "total_env_steps": 20,
        "replay_buffer_size": 10,
        "max_episode_length": 10,
        "device": "cpu",
        "obs_adapter": "strict",
        "encoder_mode": "paper-eq33",
        "critic_state": "attention-entities",
        "brma_mode": "dry-run",
        "log_file": "logs/attention_1v1_strict_eq33_attncritic_brma_dryrun_smoke.csv",
        "results_file": "results/attention_1v1_strict_eq33_attncritic_brma_dryrun_smoke_results.csv",
        "checkpoint_dir": "checkpoints/attention_1v1_strict_eq33_attncritic_brma_dryrun_smoke",
    },
}


def list_presets() -> list[str]:
    return sorted(EXPERIMENT_PRESETS.keys())


def get_preset(name: str) -> dict:
    if name not in EXPERIMENT_PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}.  Available: {', '.join(list_presets())}")
    return dict(EXPERIMENT_PRESETS[name])
