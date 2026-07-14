from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANALYSIS_FIELDS = [
    "ActorLoss",
    "PolicyLoss",
    "EntropyBonus",
    "PolicyEntropy",
    "CriticLoss",
    "ActionStdMean",
    "ActionStdMin",
    "ActionStdMax",
    "ActionLogStdMean",
    "ActionStdDeltaFromInit",
    "ActionStdGrowthRatio",
    "RawActionOutOfBoundsFrac",
    "RawActionOutOfBoundsFracPitch",
    "RawActionOutOfBoundsFracHeading",
    "RawActionOutOfBoundsFracVelocity",
    "EnvActionNearBoundFrac",
    "EnvActionNearBoundFracPitch",
    "EnvActionNearBoundFracHeading",
    "EnvActionNearBoundFracVelocity",
    "ActorUpdatesSkipped",
    "CriticUpdatesSkipped",
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
]

TREND_FIELDS = [
    "ActionStdMean",
    "PolicyEntropy",
    "RawActionOutOfBoundsFrac",
    "EnvActionNearBoundFrac",
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


def analyze(rows: list[dict], expected_steps: int,
            eval_summary: dict | None = None,
            stdout_hits: dict | None = None) -> tuple[str, list[str], list[str], dict]:
    summaries = {
        field: summarize_metric(rows, field)
        for field in ANALYSIS_FIELDS
        if field in (rows[0].keys() if rows else [])
    }
    fail = []
    review = []
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
        fail.append("关键列存在 NaN 或 Inf: " + ", ".join(sorted(set(bad_numeric))))
    if _sum_field(rows, "ActorUpdatesSkipped") > 0:
        fail.append("ActorUpdatesSkipped 总数大于 0")
    if _sum_field(rows, "CriticUpdatesSkipped") > 0:
        fail.append("CriticUpdatesSkipped 总数大于 0")
    if _recent_mean(rows, "ActionStdMean") > 0.6:
        fail.append("最后 10% 的 ActionStdMean 大于 0.6")
    if _recent_mean(rows, "RawActionOutOfBoundsFrac") > 0.05:
        fail.append("最后 10% 的 RawActionOutOfBoundsFrac 大于 0.05")
    if _recent_mean(rows, "EnvActionNearBoundFrac") > 0.05:
        fail.append("最后 10% 的 EnvActionNearBoundFrac 大于 0.05")
    stdout_hits = stdout_hits or {}
    if stdout_hits.get("Traceback", {}).get("count", 0) > 0:
        fail.append("stdout 中存在 Traceback")
    if stdout_hits.get("worker died", {}).get("count", 0) > 0:
        fail.append("stdout 中存在 worker died")
    if stdout_hits.get("timed out", {}).get("count", 0) > 0:
        fail.append("stdout 中存在 timed out")
    if _last(rows, "Step") < expected_steps:
        fail.append("实际最终 Step 小于 expected_steps")

    if _last(rows, "ActionStdGrowthRatio") > 1.25:
        review.append("ActionStdGrowthRatio 末值大于 1.25")
    raw_recent = _recent_mean(rows, "RawActionOutOfBoundsFrac")
    env_recent = _recent_mean(rows, "EnvActionNearBoundFrac")
    if 0.02 < raw_recent <= 0.05:
        review.append("最后 10% 的 RawActionOutOfBoundsFrac 位于 REVIEW 区间")
    if 0.02 < env_recent <= 0.05:
        review.append("最后 10% 的 EnvActionNearBoundFrac 位于 REVIEW 区间")
    trend = {field: summaries[field]["slope"] for field in TREND_FIELDS if field in summaries}
    if all(trend.get(field, 0.0) > 0.0 for field in (
            "ActionStdMean", "PolicyEntropy", "RawActionOutOfBoundsFrac",
            "EnvActionNearBoundFrac")):
        review.append("ActionStdMean、PolicyEntropy 和越界率均为正斜率")
    if "CriticLoss" in summaries:
        values = _finite_values(rows, "CriticLoss")
        if len(values) >= 5:
            n20 = max(1, math.ceil(len(values) * 0.2))
            last20 = values[-n20:]
            mid = values[len(values) // 3: max(len(values) // 3 + 1, 2 * len(values) // 3)]
            if last20[-1] >= last20[0] and mid and (sum(last20) / len(last20)) > (sum(mid) / len(mid)) * 1.2:
                review.append("CriticLoss 最后 20% 未下降且明显高于中段")
    eval_summary = eval_summary or {}
    red_launches = _to_float(eval_summary.get("RedMissilesFired", eval_summary.get("RedMissiles")))
    red_hits = _to_float(eval_summary.get("RedMissileHits", eval_summary.get("BlueDeathsMissile")))
    if eval_summary and (not math.isfinite(red_launches) or red_launches <= 0) and (
            not math.isfinite(red_hits) or red_hits <= 0):
        review.append("正式评估中红方没有任何击杀或发射")

    status = "FAIL" if fail else ("REVIEW" if review else "PASS")
    return status, fail, review, summaries


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
        "policy_entropy.png": (["PolicyEntropy"], "Policy entropy", "entropy"),
        "actor_loss_components.png": (["ActorLoss", "PolicyLoss", "EntropyBonus"], "Actor loss components", "loss"),
        "critic_loss.png": (["CriticLoss"], "Critic loss", "loss"),
        "action_bounds.png": (["RawActionOutOfBoundsFrac", "EnvActionNearBoundFrac"], "Action bounds", "fraction"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--eval-summary")
    parser.add_argument("--stdout-log")
    args = parser.parse_args()

    rows = load_training_csv(args.log)
    stdout_hits = scan_stdout_log(args.stdout_log)
    eval_summary = load_eval_summary(args.eval_summary)
    status, fail, review, summaries = analyze(
        rows, args.expected_steps, eval_summary, stdout_hits)
    write_plots(rows, args.plot_dir)
    write_report(args.output, status, fail, review, summaries,
                 rows, eval_summary, stdout_hits)
    print(status)


if __name__ == "__main__":
    main()
