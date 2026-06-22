"""Combat-oriented checkpoint selection scoring.

v2: rewards elimination wins and kill fractions, penalises timeout
advantage.  This prevents timeout-survival policies from outscoring
policies that actually engage and eliminate enemies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eval_record_label(record: dict) -> str:
    config = str(record.get("config", "")).lower()
    name = Path(config).name
    if "7v6" in name or "7v6" in config:
        return "7v6"
    if "5v4" in name or "5v4" in config:
        return "5v4"
    if "3v2" in name or "3v2" in config:
        return "3v2"
    return name or "unknown"


def score_record(record: dict) -> float:
    """Combat-oriented score that heavily favours elimination wins.

    Timeout alive-advantage wins receive minimal credit.
    Blue timeout wins and red losses are penalised.
    """
    num_red = _as_float(record.get("num_red"), 3.0)
    num_blue = _as_float(record.get("num_blue"), 2.0)
    if num_red <= 0:
        num_red = 3.0
    if num_blue <= 0:
        num_blue = 2.0

    red_kill_fraction = _as_float(record.get("blue_dead_mean")) / num_blue
    red_loss_fraction = _as_float(record.get("red_dead_mean")) / num_red
    net_kill_fraction = red_kill_fraction - red_loss_fraction

    return (
        3.0 * _as_float(record.get("red_elimination_win_rate"))
        + 1.0 * red_kill_fraction
        + 0.5 * net_kill_fraction
        + 0.2 * _as_float(record.get("red_missile_hits_mean"))
        + 0.1 * _as_float(record.get("red_timeout_alive_advantage_rate"))
        - 0.5 * _as_float(record.get("blue_timeout_alive_advantage_rate"))
        - 0.5 * red_loss_fraction
    )


def compute_eval_scores(records: list[dict]) -> dict[str, float]:
    scores = {
        "score_3v2": 0.0,
        "score_5v4": 0.0,
        "score_7v6": 0.0,
        "score_combined": 0.0,
    }
    for record in records:
        label = eval_record_label(record)
        if label in {"3v2", "5v4", "7v6"}:
            scores[f"score_{label}"] = score_record(record)
    scores["score_combined"] = (
        0.4 * scores["score_3v2"]
        + 0.3 * scores["score_5v4"]
        + 0.3 * scores["score_7v6"]
    )
    return scores


def best_metric_name(best_name: str) -> str:
    mapping = {
        "best_3v2": "score_3v2",
        "best_5v4": "score_5v4",
        "best_7v6": "score_7v6",
        "best_combined": "score_combined",
    }
    if best_name not in mapping:
        raise ValueError(f"unknown best checkpoint kind: {best_name}")
    return mapping[best_name]


def selected_eval_metrics(record: dict) -> dict[str, Any]:
    """Extended eval metrics including combat-oriented fields."""
    keys = [
        "config",
        "num_red", "num_blue",
        "avg_return", "avg_length",
        "red_win_rate", "blue_win_rate", "draw_rate", "timeout_rate",
        "red_elimination_win_rate",
        "red_timeout_alive_advantage_rate",
        "blue_timeout_alive_advantage_rate",
        "red_kill_fraction", "red_loss_fraction", "net_kill_fraction",
        "mav_survival_rate",
        "red_missiles_fired_mean", "blue_missiles_fired_mean",
        "red_missile_hits_mean", "blue_missile_hits_mean",
        "blue_dead_mean", "red_dead_mean",
        "red_alive_final_mean", "blue_alive_final_mean",
        "nan_detected",
    ]
    return {key: record.get(key) for key in keys if key in record}


def build_eval_checkpoint_meta(
    *,
    step: int,
    iteration: int,
    policy_arch: str,
    records: list[dict],
    extra: dict | None = None,
) -> dict:
    metrics = {eval_record_label(record): selected_eval_metrics(record) for record in records}
    meta = {
        "step": int(step),
        "iteration": int(iteration),
        "policy_arch": policy_arch,
        "scores": compute_eval_scores(records),
        "metrics": metrics,
        "eval_records": records,
    }
    if extra:
        meta.update(extra)
    return meta
