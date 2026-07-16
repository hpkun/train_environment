#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / 1024 / 1024, 3)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}


def numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce").dropna()


def boolean_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(dtype=bool)

    series = df[name]
    if series.dtype == bool:
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .dropna()
        .astype(bool)
    )


def finite_number(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value


def summary_stats(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None

    return {
        "count": int(series.size),
        "mean": finite_number(series.mean()),
        "std": finite_number(series.std(ddof=1)) if series.size > 1 else 0.0,
        "min": finite_number(series.min()),
        "p01": finite_number(series.quantile(0.01)),
        "p05": finite_number(series.quantile(0.05)),
        "median": finite_number(series.median()),
        "p95": finite_number(series.quantile(0.95)),
        "p99": finite_number(series.quantile(0.99)),
        "max": finite_number(series.max()),
    }


def phase_means(df: pd.DataFrame, metric_names: list[str]):
    if "environment_steps" not in df.columns or df.empty:
        return None

    steps = pd.to_numeric(df["environment_steps"], errors="coerce")
    maximum = float(steps.max())
    if not math.isfinite(maximum) or maximum <= 0:
        return None

    phases = {
        "early_0_10pct": steps <= maximum * 0.10,
        "middle_45_55pct": (steps >= maximum * 0.45) & (steps <= maximum * 0.55),
        "late_90_100pct": steps >= maximum * 0.90,
    }

    result = {}
    for phase_name, mask in phases.items():
        phase = df.loc[mask]
        values = {"row_count": int(len(phase))}
        for metric in metric_names:
            if metric in phase.columns:
                series = pd.to_numeric(phase[metric], errors="coerce").dropna()
                values[metric] = finite_number(series.mean()) if not series.empty else None
        result[phase_name] = values
    return result


def early_late(df: pd.DataFrame, column: str):
    if column not in df.columns or "environment_steps" not in df.columns:
        return None

    steps = pd.to_numeric(df["environment_steps"], errors="coerce")
    maximum = float(steps.max())
    values = pd.to_numeric(df[column], errors="coerce")

    early = values[steps <= maximum * 0.10].dropna()
    late = values[steps >= maximum * 0.90].dropna()

    if early.empty and late.empty:
        return None

    early_mean = finite_number(early.mean()) if not early.empty else None
    late_mean = finite_number(late.mean()) if not late.empty else None

    return {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_minus_early": (
            finite_number(late_mean - early_mean)
            if early_mean is not None and late_mean is not None
            else None
        ),
    }


def metric_with_step(df: pd.DataFrame, column: str):
    if column not in df.columns or "environment_steps" not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None

    index = values[valid].idxmax()
    return {
        "max": finite_number(values.loc[index]),
        "environment_steps_at_max": int(
            pd.to_numeric(df.loc[index, "environment_steps"], errors="coerce")
        ),
        "count_gt_0.05": int((values > 0.05).sum()),
        "count_gt_0.10": int((values > 0.10).sum()),
        "count_gt_0.20": int((values > 0.20).sum()),
        "count_gt_0.50": int((values > 0.50).sum()),
    }


def select_keys(data, keys):
    if not isinstance(data, dict):
        return data
    return {key: data.get(key) for key in keys if key in data}


def scan_log(path: Path):
    if not path.exists():
        return {"exists": False, "path": str(path)}

    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "traceback": r"Traceback \(most recent call last\)",
        "runtime_error": r"\bRuntimeError\b",
        "cuda_oom": r"CUDA out of memory|OutOfMemoryError",
        "killed": r"(^|\n)Killed(\n|$)",
        "floating_point_error": r"\bFloatingPointError\b",
        "assertion_error": r"\bAssertionError\b",
        "explicit_error_tag": r"\[ERROR\]",
    }

    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    return {
        "exists": True,
        "path": str(path.resolve()),
        "size_mb": file_size_mb(path),
        "pattern_counts": {
            name: len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
            for name, pattern in patterns.items()
        },
        "last_20_nonempty_lines": nonempty_lines[-20:],
    }


def main():
    result_dir = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "outputs/happo_2v2_1m_seed2026"
    )

    log_path = Path(
        sys.argv[2] if len(sys.argv) > 2
        else "happo_2v2_1m_seed2026.log"
    )

    training_path = result_dir / "training.csv"
    summary_path = result_dir / "summary.json"
    config_path = result_dir / "config_snapshot.json"
    checkpoint_path = result_dir / "checkpoint_final.pt"

    if not training_path.exists():
        raise SystemExit(
            f"找不到训练文件：{training_path.resolve()}\n"
            "请检查结果目录参数。"
        )

    df = pd.read_csv(training_path, low_memory=False)
    if df.empty:
        raise SystemExit("training.csv为空。")

    report = {
        "report_version": "happo_1m_compact_report_v1",
        "result_directory": str(result_dir.resolve()),
        "files": {
            "training_csv": {
                "exists": training_path.exists(),
                "size_mb": file_size_mb(training_path),
            },
            "summary_json": {
                "exists": summary_path.exists(),
                "size_mb": file_size_mb(summary_path),
            },
            "config_snapshot_json": {
                "exists": config_path.exists(),
                "size_mb": file_size_mb(config_path),
            },
            "checkpoint_final": {
                "exists": checkpoint_path.exists(),
                "size_mb": file_size_mb(checkpoint_path),
            },
        },
    }

    steps = numeric(df, "environment_steps")
    episodes = numeric(df, "episodes")

    update_steps = pd.to_numeric(
        df["environment_steps"], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    rollout_sizes = np.diff(np.concatenate([[0.0], update_steps]))
    rollout_sizes = pd.Series(rollout_sizes)

    report["basic_training"] = {
        "training_rows_updates": int(len(df)),
        "first_environment_steps": int(steps.iloc[0]) if not steps.empty else None,
        "final_environment_steps": int(steps.iloc[-1]) if not steps.empty else None,
        "final_episodes": int(episodes.iloc[-1]) if not episodes.empty else None,
        "policy_version_first": (
            finite_number(numeric(df, "policy_version").iloc[0])
            if not numeric(df, "policy_version").empty else None
        ),
        "policy_version_last": (
            finite_number(numeric(df, "policy_version").iloc[-1])
            if not numeric(df, "policy_version").empty else None
        ),
        "reached_1m": bool(not steps.empty and steps.iloc[-1] >= 1_000_000),
    }

    report["rollout_size_from_step_deltas"] = {
        "statistics": summary_stats(rollout_sizes),
        "count_equal_1": int((rollout_sizes == 1).sum()),
        "count_lt_8": int((rollout_sizes < 8).sum()),
        "count_lt_32": int((rollout_sizes < 32).sum()),
        "count_lt_64": int((rollout_sizes < 64).sum()),
        "count_equal_256": int((rollout_sizes == 256).sum()),
        "fraction_equal_256": finite_number((rollout_sizes == 256).mean()),
    }

    agent_ids = sorted({
        column.split("/", 1)[1]
        for column in df.columns
        if column.startswith("active_sample_count/") and "/" in column
    })

    report["detected_agents"] = agent_ids
    report["active_samples"] = {}

    for aid in agent_ids:
        column = f"active_sample_count/{aid}"
        values = numeric(df, column)
        positive = values[values > 0]

        report["active_samples"][aid] = {
            "statistics": summary_stats(values),
            "zero_count": int((values == 0).sum()),
            "count_lt_8": int(((values > 0) & (values < 8)).sum()),
            "count_lt_32": int(((values > 0) & (values < 32)).sum()),
            "count_lt_64": int(((values > 0) & (values < 64)).sum()),
            "minimum_positive": (
                finite_number(positive.min()) if not positive.empty else None
            ),
        }

    optimization_contract = boolean_series(
        df, "optimization_update_contract_valid"
    )
    runtime_invariants = boolean_series(df, "runtime_invariants_valid")

    report["correctness_and_finiteness"] = {
        "optimization_contract_checked_rows": int(len(optimization_contract)),
        "optimization_contract_failure_count": (
            int((~optimization_contract).sum())
            if not optimization_contract.empty else None
        ),
        "runtime_invariant_checked_rows": int(len(runtime_invariants)),
        "runtime_invariant_failure_count": (
            int((~runtime_invariants).sum())
            if not runtime_invariants.empty else None
        ),
        "nan_inf_count_sum": (
            finite_number(numeric(df, "nan_inf_count").sum())
            if not numeric(df, "nan_inf_count").empty else None
        ),
        "target_consistency_violation_sum": (
            finite_number(numeric(df, "target_consistency_violation").sum())
            if not numeric(df, "target_consistency_violation").empty else None
        ),
    }

    report["approximate_kl"] = {}
    report["entropy_trend"] = {}
    report["explained_variance_trend"] = {}
    report["critic_loss_trend"] = {}
    report["actor_gradient_norm"] = {}

    for aid in agent_ids:
        report["approximate_kl"][aid] = metric_with_step(
            df, f"approx_kl/{aid}"
        )
        report["entropy_trend"][aid] = early_late(
            df, f"entropy/{aid}"
        )
        report["explained_variance_trend"][aid] = early_late(
            df, f"explained_variance/{aid}"
        )
        report["critic_loss_trend"][aid] = early_late(
            df, f"critic_loss/{aid}"
        )
        report["actor_gradient_norm"][aid] = summary_stats(
            numeric(df, f"gradient_norm/{aid}")
        )

    factor_min = numeric(df, "importance_factor_min")
    factor_max = numeric(df, "importance_factor_max")

    report["importance_factor"] = {
        "global_min": finite_number(factor_min.min()) if not factor_min.empty else None,
        "global_max": finite_number(factor_max.max()) if not factor_max.empty else None,
        "mean_column_statistics": summary_stats(
            numeric(df, "importance_factor_mean")
        ),
        "std_column_statistics": summary_stats(
            numeric(df, "importance_factor_std")
        ),
    }

    episodic_df = df.copy()
    if "episode_length" in episodic_df.columns:
        episode_length = pd.to_numeric(
            episodic_df["episode_length"], errors="coerce"
        )
        episodic_df = episodic_df.loc[episode_length > 0]

    report["completed_episode_update_rows"] = int(len(episodic_df))
    report["training_combat_metric_phases"] = phase_means(
        episodic_df,
        [
            "mean_episode_return",
            "mav_return",
            "uav_return",
            "win_rate",
            "draw_rate",
            "episode_length",
            "launch_rate",
            "hit_rate",
            "survival_rate",
            "structural_failure_rate",
            "boundary_rate",
        ],
    )

    report["training_metric_overall"] = {}
    for name in [
        "mean_episode_return",
        "win_rate",
        "draw_rate",
        "episode_length",
        "launch_rate",
        "hit_rate",
        "survival_rate",
        "structural_failure_rate",
        "boundary_rate",
        "critic_gradient_norm",
    ]:
        report["training_metric_overall"][name] = summary_stats(
            numeric(episodic_df if name != "critic_gradient_norm" else df, name)
        )

    summary_json = read_json(summary_path)
    config_json = read_json(config_path)

    report["summary_json_selected"] = select_keys(
        summary_json,
        [
            "environment_steps",
            "episodes",
            "checkpoint",
            "checkpoint_type",
            "resume_semantics",
            "requested_seed",
            "restored_seed",
            "OPTIMIZATION_PIPELINE_ACTIVE",
            "HAPPO_RUNTIME_INVARIANTS_VALID",
            "EARLY_PERFORMANCE_SIGNAL_OBSERVED",
            "early_performance_signal_reason",
            "LEARNING_CONVERGENCE_NOT_VALIDATED",
        ],
    )

    report["config_selected"] = select_keys(
        config_json,
        [
            "scenario",
            "seed",
            "requested_seed",
            "restored_seed",
            "total_environment_steps",
            "rollout_length",
            "evaluation_interval",
            "evaluation_episodes",
            "checkpoint_interval",
            "actor_lr",
            "critic_lr",
            "ppo_epochs",
            "minibatch_size",
            "clip_param",
            "entropy_coef",
            "gamma",
            "gae_lambda",
            "hidden_dim",
            "actual_device",
            "torch_version",
            "jsbsim_version",
            "uses_tam",
            "uses_recurrence",
            "uses_attention",
            "algorithm_mode",
        ],
    )

    last_row_keys = [
        "environment_steps",
        "episodes",
        "mean_episode_return",
        "win_rate",
        "episode_length",
        "launch_rate",
        "hit_rate",
        "survival_rate",
        "structural_failure_rate",
        "boundary_rate",
        "nan_inf_count",
        "target_consistency_violation",
        "optimization_update_contract_valid",
        "runtime_invariants_valid",
        "importance_factor_min",
        "importance_factor_max",
    ]

    for aid in agent_ids:
        last_row_keys.extend([
            f"active_sample_count/{aid}",
            f"approx_kl/{aid}",
            f"entropy/{aid}",
            f"gradient_norm/{aid}",
            f"critic_loss/{aid}",
            f"explained_variance/{aid}",
            f"actor_changed/{aid}",
            f"critic_head_changed/{aid}",
            f"update_contract_valid/{aid}",
        ])

    last_row = df.iloc[-1]
    report["last_training_row_selected"] = {
        key: finite_number(last_row[key])
        for key in last_row_keys
        if key in df.columns and not pd.isna(last_row[key])
    }

    report["log_scan"] = scan_log(log_path)

    print("===== HAPPO 1M COMPACT REPORT BEGIN =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("===== HAPPO 1M COMPACT REPORT END =====")


if __name__ == "__main__":
    main()
