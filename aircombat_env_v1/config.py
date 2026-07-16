"""Configuration loading and focused updates for experiment scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "f16_pid_v1.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def save_config(config, path=DEFAULT_CONFIG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
