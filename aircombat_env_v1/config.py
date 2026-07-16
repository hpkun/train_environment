"""Configuration loading and focused updates for experiment scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "f16_pid_v1.yaml"
PARAMETER_STATUSES = {"initial_guess", "candidate", "validated"}


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    if config.get("parameter_status") not in PARAMETER_STATUSES:
        raise ValueError("parameter_status must be initial_guess, candidate, or validated")
    return config


def mark_candidate(config):
    config["parameter_status"] = "candidate"
    config["gain_source"] = "offline_tuned_on_local_jsbsim_f16"
    config["paper_reported"] = False
    return config


def save_config(config, path=DEFAULT_CONFIG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
