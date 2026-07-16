from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANALYSIS_FIELDS = [
    "ActorLoss",
    "PolicyLoss",
    "EntropyBonus",
    "BaseNormalEntropy",
    "CriticLoss",
    "ActionStdMean",
    "ActionStdMin",
    "ActionStdMax",
    "ActionLogStdMean",
    "ActionStdDeltaFromInit",
    "ActionStdGrowthRatio",
    "StateDependentStdMean",
    "StateDependentStdMin",
    "StateDependentStdMax",
    "StateDependentStdLowerBoundFrac",
    "StateDependentStdUpperBoundFrac",
    "ExecutedActionAbsMean",
    "ExecutedActionNearBoundFrac",
    "ExecutedActionNearBoundFracPitch",
    "ExecutedActionNearBoundFracHeading",
    "ExecutedActionNearBoundFracVelocity",
    "PolicyMeanNearBoundFrac",
    "ActorUpdatesSkipped",
    "ActorUpdateAttempts",
    "ActorUpdatesApplied",
    "CriticUpdatesSkipped",
    "CriticUpdateAttempts",
    "CriticUpdatesApplied",
    "InvalidNumericalEpisodes",
    "InvalidTransitionsDropped",
    "RedMeanReward",
    "WinRateRecent",
    "RedWinRate",
    "WinRateCumul",
    "Episodes",
    "RedWins",
    "BlueWins",
    "Draws",
    "RedMissiles",
    "BlueMissiles",
    "LaunchDiagRedGeometryOk",
    "LaunchDiagBlueGeometryOk",
    "LaunchDiagRedLockMature",
    "LaunchDiagBlueLockMature",
    "LaunchDiagRedLaunches",
    "LaunchDiagBlueLaunches",
    "RedMissileHitRate",
    "BlueMissileHitRate",
    "RedDeathsMissile",
    "RedDeathsCrash",
    "BlueDeathsMissile",
    "BlueDeathsCrash",
    "RedAliveMean",
    "BlueAliveMean",
    "MaximumSpeedAfterLimiterMps",
    "SpeedLimiterActivationRatePer1000PhysicsSteps",
    "MaximumLoadG",
    "LoadLimiterActivations",
]

TREND_FIELDS = [
    "ActionStdMean",
    "BaseNormalEntropy",
    "ExecutedActionNearBoundFrac",
    "PolicyMeanNearBoundFrac",
    "RedMeanReward",
    "CriticLoss",
]

CUMULATIVE_FIELDS = {
    "Episodes",
    "RedWins",
    "BlueWins",
    "Draws",
    "RedDeathsMissile",
    "BlueDeathsMissile",
}

STDOUT_PATTERNS = [
    "Traceback",
    "RuntimeError",
    "NaN",
    "Inf",
    "worker died",
    "timed out",
    "skipped",
]


def _to_float(value) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_values(rows: list[dict], field: str) -> list[float]:
    values = [_to_float(row.get(field)) for row in rows]
    return [v for v in values if math.isfinite(v)]


def _slope(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    x0 = pairs[0][0]
    xs0 = [x - x0 for x, _ in pairs]
    ys0 = [y for _, y in pairs]
    x_mean = sum(xs0) / len(xs0)
    y_mean = sum(ys0) / len(ys0)
    denom = sum((x - x_mean) ** 2 for x in xs0)
    if denom <= 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs0, ys0)) / denom


def load_training_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: _to_float(row.get("Step")))
    return rows


def summarize_metric(rows: list[dict], field: str) -> dict:
    values = _finite_values(rows, field)
    steps = _finite_values(rows, "Step")
    recent_n = max(1, math.ceil(len(rows) * 0.1)) if rows else 0
    recent_values = _finite_values(rows[-recent_n:], field) if recent_n else []
    if not values:
        return {
            "first": math.nan,
            "last": math.nan,
            "recent_mean": math.nan,
            "min": math.nan,
            "max": math.nan,
            "slope": math.nan,
        }
    return {
        "first": values[0],
        "last": values[-1],
        "recent_mean": sum(recent_values) / len(recent_values) if recent_values else math.nan,
        "min": min(values),
        "max": max(values),
        "slope": _slope(steps, [_to_float(row.get(field)) for row in rows]),
    }


def scan_stdout_log(path: str | Path | None) -> dict:
    if not path:
        return {}
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    result = {}
    for pattern in STDOUT_PATTERNS:
        hits = [line for line in lines if pattern.lower() in line.lower()]
        result[pattern] = {
            "count": len(hits),
            "tail": hits[-5:],
        }
    return result


def load_eval_summary(path: str | Path | None) -> dict:
    if not path:
        return {}
    rows = load_training_csv(path)
    return rows[-1] if rows else {}


def _last(rows: list[dict], field: str) -> float:
    return _to_float(rows[-1].get(field)) if rows else math.nan


def _recent_mean(rows: list[dict], field: str) -> float:
    recent_n = max(1, math.ceil(len(rows) * 0.1)) if rows else 0
    values = _finite_values(rows[-recent_n:], field) if recent_n else []
    return sum(values) / len(values) if values else math.nan


def _sum_field(rows: list[dict], field: str) -> float:
    return sum(v for v in (_to_float(row.get(field)) for row in rows) if math.isfinite(v))


def _mean(rows: list[dict], field: str) -> float:
    values = _finite_values(rows, field)
    return sum(values) / len(values) if values else math.nan


def _cumulative_delta(rows: list[dict], field: str) -> float:
    values = _finite_values(rows, field)
    return max(0.0, values[-1] - values[0]) if len(values) >= 2 else 0.0


def _window_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    step_values = _finite_values(rows, "Step")
    step_span = max(step_values[-1] - step_values[0], 1.0) if step_values else 1.0
    episode_delta = _cumulative_delta(rows, "Episodes")
    launches = _sum_field(rows, "LaunchDiagRedLaunches")
    hit_delta = _cumulative_delta(rows, "RedMissileHits")
    return {
        "RedMeanReward": _mean(rows, "RedMeanReward"),
        "geometry_rate": _sum_field(rows, "LaunchDiagRedGeometryOk") / step_span,
        "lock_mature_rate": _sum_field(rows, "LaunchDiagRedLockMature") / step_span,
        "launches_per_episode": launches / max(episode_delta, 1.0),
        "hits_per_episode": hit_delta / max(episode_delta, 1.0),
        "RedWinRate": _mean(rows, "WinRateRecent") if any(
            row.get("WinRateRecent", "") != "" for row in rows) else _mean(rows, "RedWinRate"),
        "RedAliveMean": _mean(rows, "RedAliveMean"),
        "BlueAliveMean": _mean(rows, "BlueAliveMean"),
        "PolicyLoss": _mean(rows, "PolicyLoss"),
        "BaseNormalEntropy": _mean(rows, "BaseNormalEntropy"),
        "EntropyBonus": _mean(rows, "EntropyBonus"),
        "StateDependentStdMean": _mean(rows, "StateDependentStdMean"),
        "StateDependentStdUpperBoundFrac": _mean(
            rows, "StateDependentStdUpperBoundFrac"),
        "ExecutedActionNearBoundFrac": _mean(
            rows, "ExecutedActionNearBoundFrac"),
    }


def _improved(middle: float, late: float, relative: float = 0.05) -> bool:
    if not math.isfinite(middle) or not math.isfinite(late):
        return False
    return late > middle + max(abs(middle) * relative, 1e-8)


def analyze_diagnostics(rows: list[dict], expected_steps: int,
                        eval_summary: dict | None = None,
                        stdout_hits: dict | None = None,
                        rule_audit: dict | None = None) -> dict:
    summaries = {
        field: summarize_metric(rows, field)
        for field in ANALYSIS_FIELDS
        if field in (rows[0].keys() if rows else [])
    }
    health_failures = []
    learning_reasons = []
    if not rows:
        health_failures.append("训练日志为空")
    key_fields = [field for field in ANALYSIS_FIELDS if rows and field in rows[0]]
    bad_numeric = []
    for field in key_fields:
        for row in rows:
            value = row.get(field, "")
            if value == "":
                continue
            parsed = _to_float(value)
            if not math.isfinite(parsed):
                bad_numeric.append(field)
                break
    if bad_numeric:
        health_failures.append(
            "关键列存在 NaN 或 Inf: " + ", ".join(sorted(set(bad_numeric))))
    if _sum_field(rows, "ActorUpdatesSkipped") > 0:
        health_failures.append("Actor 梯度异常导致更新跳过")
    if _sum_field(rows, "CriticUpdatesSkipped") > 0:
        health_failures.append("Critic 梯度异常导致更新跳过")
    attempts = _sum_field(rows, "ActorUpdateAttempts") + _sum_field(
        rows, "CriticUpdateAttempts")
    applied = _sum_field(rows, "ActorUpdatesApplied") + _sum_field(
        rows, "CriticUpdatesApplied")
    if attempts > 0 and applied == 0:
        health_failures.append("PPO 更新全部跳过")
    if max(_finite_values(rows, "InvalidNumericalEpisodes") or [0.0]) > 0:
        health_failures.append("存在 invalid numerical episode")
    std_min = min(_finite_values(rows, "StateDependentStdMin") or [0.05])
    std_max = max(_finite_values(rows, "StateDependentStdMax") or [0.6])
    if std_min < 0.05 - 1e-6 or std_max > 0.6 + 1e-6:
        health_failures.append("state-dependent std 越过 [0.05, 0.6] 硬边界")
    if max(_finite_values(rows, "MaximumSpeedAfterLimiterMps") or [0.0]) > 600.01:
        health_failures.append("速度投影后仍超过 600.01 m/s")
    stdout_hits = stdout_hits or {}
    if stdout_hits.get("Traceback", {}).get("count", 0) > 0:
        health_failures.append("stdout 中存在 Traceback")
    if stdout_hits.get("worker died", {}).get("count", 0) > 0:
        health_failures.append("stdout 中存在 worker died")
    if stdout_hits.get("timed out", {}).get("count", 0) > 0:
        health_failures.append("stdout 中存在 timed out")
    if _last(rows, "Step") < expected_steps:
        health_failures.append("实际最终 Step 小于 expected_steps")
    if rule_audit is not None and rule_audit.get("status") != "PASS":
        health_failures.append("规则环境审计 FAIL")
    eval_summary = eval_summary or {}
    metadata_fields = (
        "CheckpointSchema", "EnvironmentProfile",
        "EnvironmentConfigFingerprint", "ObsNormalization",
        "RewardVersion", "RewardMode", "PIDProfile",
        "MissileGuidanceMode", "ActionDistribution", "BluePolicyProfile",
        "RedMWSMode", "BlueMWSMode", "NumRed", "NumBlue", "MaxSteps",
    )
    if eval_summary:
        for field in metadata_fields:
            train_value = rows[-1].get(field, "") if rows else ""
            eval_value = eval_summary.get(field, "")
            if train_value in (None, "") or eval_value in (None, ""):
                health_failures.append(
                    f"checkpoint/eval metadata field missing: {field}")
            elif str(train_value) != str(eval_value):
                health_failures.append(
                    f"checkpoint/eval metadata mismatch: {field}")

    final_step = _last(rows, "Step")
    windows = {"burn_in": [], "middle": [], "late": []}
    if math.isfinite(final_step) and final_step >= 10_000:
        windows["burn_in"] = [
            row for row in rows if _to_float(row.get("Step")) <= 0.30 * final_step]
        windows["middle"] = [
            row for row in rows
            if 0.30 * final_step < _to_float(row.get("Step")) <= 0.65 * final_step]
        windows["late"] = [
            row for row in rows if _to_float(row.get("Step")) > 0.65 * final_step]
        middle = _window_metrics(windows["middle"])
        late = _window_metrics(windows["late"])
        improvements = []
        for field in ("geometry_rate", "lock_mature_rate", "launches_per_episode",
                      "hits_per_episode", "RedMeanReward", "RedAliveMean"):
            if _improved(middle.get(field, math.nan), late.get(field, math.nan)):
                improvements.append(field)
        if (_improved(late.get("BlueAliveMean", math.nan),
                      middle.get("BlueAliveMean", math.nan))):
            improvements.append("BlueAliveMean下降")
        if (middle.get("RedWinRate", 0.0) <= 0.0
                and late.get("RedWinRate", 0.0) > 0.0):
            improvements.append("出现非零胜率")
        warnings = []
        if late.get("StateDependentStdUpperBoundFrac", 0.0) > 0.20:
            warnings.append("std 大量贴近上界")
        middle_oob = middle.get("ExecutedActionNearBoundFrac", math.nan)
        late_oob = late.get("ExecutedActionNearBoundFrac", math.nan)
        if (math.isfinite(late_oob) and (late_oob > 0.25
                or (math.isfinite(middle_oob) and late_oob > middle_oob + 0.05))):
            warnings.append("raw action 越界率偏高或持续上升")
        if (math.isfinite(late.get("EntropyBonus", math.nan))
                and math.isfinite(late.get("PolicyLoss", math.nan))
                and abs(late["EntropyBonus"]) > 1.5 * max(
                    abs(late["PolicyLoss"]), 1e-8)):
            warnings.append("entropy bonus 长期明显支配 policy surrogate")
        worsening = sum(
            _improved(late.get(field, math.nan), middle.get(field, math.nan), 0.10)
            for field in ("BlueAliveMean",))
        worsening += sum(
            _improved(late.get(field, math.nan), middle.get(field, math.nan), 0.10)
            for field in ("ExecutedActionNearBoundFrac",))
        if worsening >= 2:
            warnings.append("战术或优化指标持续恶化")
        if warnings:
            learning_status = "OPTIMIZATION_WARNING"
            learning_reasons.extend(warnings)
        elif len(improvements) >= 2:
            learning_status = "POSITIVE_TREND"
            learning_reasons.extend(improvements)
        else:
            learning_status = "NO_CLEAR_TREND_YET"
            learning_reasons.append("middle 与 late 之间不足两个战术指标改善")
        window_metrics = {"middle": middle, "late": late}
    else:
        learning_status = "INSUFFICIENT_DATA"
        learning_reasons.append("少于 10k steps，只进行数值健康检查")
        window_metrics = {}

    return {
        "EnvironmentHealthStatus": "FAIL" if health_failures else "PASS",
        "LearningEvidenceStatus": learning_status,
        "EnvironmentHealthReasons": health_failures,
        "LearningEvidenceReasons": learning_reasons,
        "summaries": summaries,
        "burn_in_fraction": 0.30,
        "window_boundaries": {"middle": [0.30, 0.65], "late": [0.65, 1.00]},
        "window_metrics": window_metrics,
        "window_row_counts": {key: len(value) for key, value in windows.items()},
    }


def analyze(rows: list[dict], expected_steps: int,
            eval_summary: dict | None = None,
            stdout_hits: dict | None = None) -> tuple[str, list[str], list[str], dict]:
    """Compatibility wrapper; new callers should use analyze_diagnostics."""
    result = analyze_diagnostics(
        rows, expected_steps, eval_summary=eval_summary,
        stdout_hits=stdout_hits)
    return (
        result["EnvironmentHealthStatus"],
        result["EnvironmentHealthReasons"],
        result["LearningEvidenceReasons"],
        result["summaries"],
    )


def _plot_series(rows: list[dict], fields: list[str], output: Path,
                 title: str, ylabel: str) -> None:
    steps = [_to_float(row.get("Step")) for row in rows]
    plt.figure(figsize=(7, 4))
    plotted = False
    for field in fields:
        if not rows or field not in rows[0]:
            continue
        values = [_to_float(row.get(field)) for row in rows]
        plt.plot(steps, values, label=field)
        plotted = True
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def write_plots(rows: list[dict], plot_dir: str | Path) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    specs = {
        "action_std.png": (["ActionStdMean", "ActionStdMin", "ActionStdMax"], "Action std", "std"),
        "policy_entropy.png": (["BaseNormalEntropy"], "Base Normal entropy", "entropy"),
        "actor_loss_components.png": (["ActorLoss", "PolicyLoss", "EntropyBonus"], "Actor loss components", "loss"),
        "critic_loss.png": (["CriticLoss"], "Critic loss", "loss"),
        "action_bounds.png": (["ExecutedActionNearBoundFrac", "PolicyMeanNearBoundFrac"], "Action saturation", "fraction"),
        "reward.png": (["RedMeanReward"], "Red reward", "reward"),
        "win_rate.png": (["WinRateRecent", "RedWinRate", "WinRateCumul"], "Win rate", "rate"),
        "launch_diagnostics.png": ([
            "LaunchDiagRedGeometryOk",
            "LaunchDiagRedLockMature",
            "LaunchDiagRedLaunches",
            "LaunchDiagBlueGeometryOk",
            "LaunchDiagBlueLockMature",
            "LaunchDiagBlueLaunches",
        ], "Launch diagnostics", "count"),
    }
    for filename, (fields, title, ylabel) in specs.items():
        _plot_series(rows, fields, plot_path / filename, title, ylabel)


def write_report(output: str | Path, status: str, fail: list[str], review: list[str],
                 summaries: dict, rows: list[dict], eval_summary: dict,
                 stdout_hits: dict) -> None:
    lines = [
        f"# {status}",
        "",
        "PASS 只代表训练链路和数值行为允许继续长训练，不代表算法已经学会空战。",
        "",
        "## 判定原因",
    ]
    if fail:
        lines.extend(f"- FAIL: {item}" for item in fail)
    if review:
        lines.extend(f"- REVIEW: {item}" for item in review)
    if not fail and not review:
        lines.append("- 未触发 FAIL 或 REVIEW 条件。")
    lines.extend(["", "## 指标摘要", ""])
    lines.append("|Metric|First|Last|Last10Mean|Min|Max|Slope|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for field in ANALYSIS_FIELDS:
        if field not in summaries:
            continue
        s = summaries[field]
        lines.append(
            f"|{field}|{s['first']:.6g}|{s['last']:.6g}|"
            f"{s['recent_mean']:.6g}|{s['min']:.6g}|{s['max']:.6g}|"
            f"{s['slope']:.6g}|")
    if rows:
        lines.extend(["", "## 累计字段末行值", ""])
        for field in sorted(CUMULATIVE_FIELDS):
            if field in rows[-1]:
                lines.append(f"- {field}: {rows[-1].get(field)}")
    if eval_summary:
        lines.extend(["", "## Eval Summary", ""])
        for key in sorted(eval_summary):
            lines.append(f"- {key}: {eval_summary[key]}")
    if stdout_hits:
        lines.extend(["", "## Stdout Scan", ""])
        for pattern in STDOUT_PATTERNS:
            hit = stdout_hits.get(pattern, {"count": 0, "tail": []})
            lines.append(f"- {pattern}: {hit['count']}")
            for item in hit["tail"]:
                lines.append(f"  - {item}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostic_report(output: str | Path, result: dict,
                            rows: list[dict], eval_summary: dict,
                            stdout_hits: dict) -> None:
    lines = [
        f"# EnvironmentHealthStatus: {result['EnvironmentHealthStatus']}",
        "",
        f"LearningEvidenceStatus: {result['LearningEvidenceStatus']}",
        "",
        "环境健康只依据数值、执行链路、硬边界和规则审计；"
        "短期胜率、发射率、命中率和回报不参与环境 PASS/FAIL。",
        "",
        "## Environment Health",
    ]
    reasons = result["EnvironmentHealthReasons"]
    lines.extend(
        [f"- {item}" for item in reasons]
        if reasons else ["- 未触发环境健康硬失败条件。"])
    lines.extend(["", "## Learning Evidence"])
    lines.extend(f"- {item}" for item in result["LearningEvidenceReasons"])
    lines.extend([
        "",
        "## Burn-in And Windows",
        "",
        "- burn-in: 0%–30%，只检查数值健康性",
        "- middle: 30%–65%",
        "- late: 65%–100%",
        f"- rows: {result['window_row_counts']}",
    ])
    for window, metrics in result.get("window_metrics", {}).items():
        lines.extend(["", f"### {window}"])
        lines.extend(f"- {key}: {value}" for key, value in metrics.items())
    lines.extend(["", "## Metric Summary", ""])
    lines.append("|Metric|First|Last|Last10Mean|Min|Max|Slope|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for field in ANALYSIS_FIELDS:
        if field not in result["summaries"]:
            continue
        summary = result["summaries"][field]
        lines.append(
            f"|{field}|{summary['first']:.6g}|{summary['last']:.6g}|"
            f"{summary['recent_mean']:.6g}|{summary['min']:.6g}|"
            f"{summary['max']:.6g}|{summary['slope']:.6g}|")
    if eval_summary:
        lines.extend(["", "## Eval Summary"])
        lines.extend(f"- {key}: {eval_summary[key]}" for key in sorted(eval_summary))
    if stdout_hits:
        lines.extend(["", "## Stdout Scan"])
        lines.extend(
            f"- {pattern}: {stdout_hits.get(pattern, {}).get('count', 0)}"
            for pattern in STDOUT_PATTERNS)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--eval-summary")
    parser.add_argument("--stdout-log")
    parser.add_argument("--rule-audit")
    args = parser.parse_args()

    rows = load_training_csv(args.log)
    stdout_hits = scan_stdout_log(args.stdout_log)
    eval_summary = load_eval_summary(args.eval_summary)
    rule_audit = None
    if args.rule_audit:
        rule_audit = json.loads(Path(args.rule_audit).read_text(encoding="utf-8"))
    result = analyze_diagnostics(
        rows, args.expected_steps, eval_summary, stdout_hits, rule_audit)
    write_plots(rows, args.plot_dir)
    write_diagnostic_report(
        args.output, result, rows, eval_summary, stdout_hits)
    print(
        f"EnvironmentHealthStatus={result['EnvironmentHealthStatus']} "
        f"LearningEvidenceStatus={result['LearningEvidenceStatus']}")


if __name__ == "__main__":
    main()
