"""Audit rich logging outputs and generated paper-style figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_logging_schema import FILE_SCHEMAS


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _rate_ok(value: Any) -> bool:
    x = _num(value)
    return x is not None and 0.0 <= x <= 1.0


def _result(status: str = "pass", reason: str = "") -> dict[str, Any]:
    return {"status": status, "reason": reason, "warnings": [], "errors": []}


def _downgrade(item: dict[str, Any], status: str, message: str) -> None:
    if status == "fail":
        item["status"] = "fail"
        item["errors"].append(message)
    elif item["status"] != "fail":
        item["status"] = "warning"
        item["warnings"].append(message)


def _check_header(item: dict[str, Any], header: list[str], expected: list[str]) -> None:
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        _downgrade(item, "fail", f"header mismatch missing={missing} extra={extra}")


def _monotonic(rows: list[dict[str, str]], key: str) -> bool:
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return all(b >= a for a, b in zip(vals, vals[1:]))


def audit_train(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "train_metrics.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing train_metrics.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["train_metrics.csv"])
    if not rows:
        _downgrade(item, "fail", "no metric rows")
        return item
    if not _monotonic(rows, "total_env_steps_actual"):
        _downgrade(item, "fail", "total_env_steps_actual is not monotonic")
    for key in ("wall_time_sec", "steps_per_second"):
        if any((_num(r.get(key)) is None or _num(r.get(key)) < 0) for r in rows):
            _downgrade(item, "fail", f"{key} has invalid values")
    for key in ("red_win_rate", "blue_win_rate", "draw_rate", "timeout_rate", "mav_survival_rate"):
        if any(not _rate_ok(r.get(key)) for r in rows):
            _downgrade(item, "fail", f"{key} outside [0,1] or empty")
    if any(_num(r.get("kill_death_ratio")) is not None and _num(r.get("kill_death_ratio")) > 1e5 for r in rows):
        _downgrade(item, "warning", "kill_death_ratio is very large because denominator is near zero")
    for key in ("actor_loss", "critic_loss", "entropy", "policy_gradient_norm", "value_gradient_norm", "nan_detected"):
        if key not in header:
            _downgrade(item, "fail", f"{key} column missing")
    if all(not r.get("policy_gradient_norm") for r in rows):
        _downgrade(item, "warning", "policy_gradient_norm column exists but current trainer does not expose values")
    if all(not r.get("value_gradient_norm") for r in rows):
        _downgrade(item, "warning", "value_gradient_norm column exists but current trainer does not expose values")
    critical = ["avg_episode_return", "red_win_rate", "mav_survival_rate", "critic_loss", "entropy"]
    empty_critical = [key for key in critical if all(r.get(key, "") == "" for r in rows)]
    if empty_critical:
        _downgrade(item, "fail", f"critical columns all empty: {empty_critical}")
    return item


def audit_eval_episode(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "eval_episode_metrics.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing eval_episode_metrics.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["eval_episode_metrics.csv"])
    if not rows:
        _downgrade(item, "fail", "no eval episode rows")
        return item
    if any(not r.get("outcome") for r in rows):
        _downgrade(item, "fail", "outcome is empty")
    if any(all(r.get(k, "") == "" for k in ("red_win", "blue_win", "draw", "timeout")) for r in rows):
        _downgrade(item, "fail", "episode outcome flags all empty")
    for key in ("red_alive_final", "blue_alive_final"):
        if any(_num(r.get(key)) is None or _num(r.get(key)) < 0 for r in rows):
            _downgrade(item, "fail", f"{key} invalid")
    for key in ("red_missiles_fired", "blue_missiles_fired", "red_missile_hits", "blue_missile_hits", "red_dead", "blue_dead"):
        if key not in header:
            _downgrade(item, "fail", f"{key} missing")
    return item


def audit_summary(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "eval_summary_metrics.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing eval_summary_metrics.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["eval_summary_metrics.csv"])
    if not rows:
        _downgrade(item, "fail", "no eval summary rows")
        return item
    for key in ("checkpoint_name", "eval_scenario", "episodes"):
        if any(not r.get(key) for r in rows):
            _downgrade(item, "fail", f"{key} empty")
    if any((_num(r.get("episodes")) or 0) <= 0 for r in rows):
        _downgrade(item, "fail", "episodes must be > 0")
    return item


def audit_aircraft(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "aircraft_timeseries.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing aircraft_timeseries.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["aircraft_timeseries.csv"])
    if not rows:
        _downgrade(item, "fail", "no aircraft timeseries rows")
        return item
    required = ["episode_id", "agent_id", "role", "team", "alive", "lon", "lat", "altitude", "roll", "pitch", "yaw", "heading", "is_mav", "is_uav"]
    for key in required:
        if key not in header:
            _downgrade(item, "fail", f"{key} missing")
    if "speed" not in header and "velocity" not in header:
        _downgrade(item, "fail", "speed/velocity missing")
    for key in ("action_pitch", "action_heading", "action_speed"):
        if key not in header:
            _downgrade(item, "fail", f"{key} missing")
    grouped: dict[str, int] = {}
    for row in rows:
        grouped[row.get("agent_id", "")] = grouped.get(row.get("agent_id", ""), 0) + 1
    if any(count < 2 for aid, count in grouped.items() if aid):
        _downgrade(item, "warning", "some agents have fewer than 2 steps")
    if all((_num(r.get("altitude")) or 0.0) == 0.0 for r in rows):
        _downgrade(item, "fail", "altitude all zero")
    if all((_num(r.get("speed")) or _num(r.get("velocity")) or 0.0) == 0.0 for r in rows):
        _downgrade(item, "fail", "speed/velocity all zero")
    _downgrade(item, "warning", "yaw/heading unit is degrees in smoke placeholder; verify source units for full experiment")
    return item


def audit_missile_events(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "missile_events.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing missile_events.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["missile_events.csv"])
    if not rows:
        _downgrade(item, "warning", "no missile events in smoke; header is present")
        return item
    if "event_type" not in header:
        _downgrade(item, "fail", "event_type missing")
    launch_rows = [r for r in rows if r.get("event_type") == "launch"]
    if launch_rows:
        for key in ("owner_id", "target_id", "sim_time", "lon", "lat", "altitude"):
            if all(not r.get(key) for r in launch_rows):
                _downgrade(item, "fail", f"launch rows have empty {key}")
    return item


def audit_missile_timeseries(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "missile_timeseries.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing missile_timeseries.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["missile_timeseries.csv"])
    if not rows:
        _downgrade(item, "warning", "no missile timeseries rows in smoke")
        return item
    for key in ("missile_id", "owner_id", "target_id", "alive", "lon", "lat", "altitude"):
        if key not in header:
            _downgrade(item, "fail", f"{key} missing")
    counts: dict[str, int] = {}
    for row in rows:
        mid = row.get("missile_id", "")
        counts[mid] = counts.get(mid, 0) + 1
    if any(v < 2 for k, v in counts.items() if k):
        _downgrade(item, "warning", "some missiles have fewer than 2 time points")
    return item


def audit_reward(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "reward_components.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing reward_components.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["reward_components.csv"])
    if not rows:
        _downgrade(item, "fail", "no reward component rows")
        return item
    roles = {r.get("role") for r in rows}
    if "mav" not in roles or not any(role and role != "mav" for role in roles):
        _downgrade(item, "warning", "smoke should include both MAV and UAV reward rows")
    component_cols = [c for c in header if c.endswith("_reward") and c != "total_reward"]
    if all(all((r.get(c, "") in ("", "0", "0.0")) for r in rows) for c in component_cols):
        _downgrade(item, "warning", "all explicit reward components are zero/empty in smoke")
    return item


def audit_efficiency(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "training_efficiency.json"
    item = _result()
    item["exists"] = path.exists()
    data = _read_json(path)
    if not isinstance(data, dict):
        _downgrade(item, "fail", "missing or invalid training_efficiency.json")
        return item
    required = ["device", "num_envs", "rollout_length_per_env", "transitions_per_rollout", "total_wall_time_sec", "steps_per_second_mean", "train_start_time", "train_end_time"]
    for key in required:
        if key not in data:
            _downgrade(item, "fail", f"{key} missing")
    for key in ("peak_gpu_memory_gb", "peak_cpu_memory_gb"):
        if key not in data:
            _downgrade(item, "warning", f"{key} missing")
        elif data.get(key) is None:
            _downgrade(item, "warning", f"{key} not_available")
    return item


def audit_perturbation(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "perturbation_eval_summary.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing perturbation_eval_summary.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["perturbation_eval_summary.csv"])
    if rows and any(r.get("availability") == "schema_only" for r in rows):
        _downgrade(item, "warning", "perturbation smoke data is schema_only, not a real perturbation result")
    return item


def audit_attention(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "attention_metrics.csv"
    item = _result()
    item["exists"] = path.exists()
    header, rows = _read_csv(path)
    item["rows"] = len(rows)
    if not path.exists():
        _downgrade(item, "fail", "missing attention_metrics.csv")
        return item
    _check_header(item, header, FILE_SCHEMAS["attention_metrics.csv"])
    if not rows or not any(r.get("availability") == "not_available" for r in rows):
        _downgrade(item, "fail", "attention must be marked not_available for current algorithm")
    if any(r.get("attention_entropy") or r.get("attention_top1_weight") for r in rows):
        _downgrade(item, "fail", "attention values should not be fabricated")
    return item


FIGURES = {
    "reward_curve": ("train_metrics.csv", "avg_episode_return"),
    "win_rate_curve": ("train_metrics.csv", "red_win_rate/blue_win_rate"),
    "rwr_kd_bar": ("train_metrics.csv", "relative_win_ratio/kill_death_ratio"),
    "zero_shot_transfer_bar": ("eval_summary_metrics.csv", "schema/example only for smoke"),
    "ablation_reward_win_curve": ("train_metrics.csv", "requires multiple runs for real ablation"),
    "trajectory_2d": ("aircraft_timeseries.csv", "lon/lat"),
    "aircraft_attitude_curves": ("aircraft_timeseries.csv", "altitude/speed/yaw/pitch"),
    "reward_component_curves": ("reward_components.csv", "role reward columns"),
    "perturbation_generalization_bar": ("perturbation_eval_summary.csv", "schema_only unless perturbation eval exists"),
    "loss_entropy_gradient_curves": ("train_metrics.csv", "loss/entropy/gradient columns"),
}


def audit_figures(figures_dir: Path) -> dict[str, Any]:
    out = {}
    for stem, (source, note) in FIGURES.items():
        png = figures_dir / f"{stem}.png"
        svg = figures_dir / f"{stem}.svg"
        item = _result()
        item.update({
            "exists": png.exists() and svg.exists(),
            "png_size": png.stat().st_size if png.exists() else 0,
            "svg_size": svg.stat().st_size if svg.exists() else 0,
            "data_source": source,
        })
        if not png.exists() or not svg.exists():
            _downgrade(item, "fail", "png/svg missing")
        elif item["png_size"] <= 0 or item["svg_size"] <= 0:
            _downgrade(item, "fail", "empty figure file")
        if "schema" in note or "requires" in note:
            _downgrade(item, "warning", note)
        out[stem] = item
    return out


def audit_coverage(input_dir: Path) -> dict[str, Any]:
    report = _read_json(input_dir / "plot_coverage_report.json")
    item = _result()
    item["exists"] = isinstance(report, dict)
    item["report"] = report
    if not isinstance(report, dict):
        _downgrade(item, "fail", "missing coverage report")
        return item
    brma = report.get("BRMA-MAPPO", {})
    tam = report.get("TAM-HAPPO", {})
    if brma.get("scale_transfer") == "available":
        _downgrade(item, "fail", "smoke scale_transfer must not be marked available")
    if brma.get("ablation_reward_win_curves") == "available":
        _downgrade(item, "fail", "smoke ablation must not be marked available")
    if brma.get("attention_heatmap_metrics") != "not_implemented_by_current_algorithm":
        _downgrade(item, "fail", "attention coverage must be not_implemented_by_current_algorithm")
    if tam.get("perturbation_generalization") == "available":
        _downgrade(item, "fail", "smoke perturbation must not be marked available")
    if tam.get("ablation_curves") == "available":
        _downgrade(item, "fail", "smoke TAM ablation must not be marked available")
    return item


def summarize_status(items: list[dict[str, Any]]) -> str:
    if any(item.get("status") == "fail" for item in items):
        return "fail"
    if any(item.get("status") == "warning" for item in items):
        return "pass_with_warnings"
    return "pass"


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Rich Logging Audit Report", "", f"- overall_status: `{report['overall_status']}`", ""]
    for section in ("files", "figures"):
        lines.append(f"## {section.title()}")
        for name, item in report[section].items():
            lines.append(f"- {name}: `{item['status']}`")
            for warning in item.get("warnings", []):
                lines.append(f"  - warning: {warning}")
            for error in item.get("errors", []):
                lines.append(f"  - error: {error}")
        lines.append("")
    lines.append("## Coverage")
    lines.append(f"- status: `{report['coverage']['status']}`")
    for warning in report["coverage"].get("warnings", []):
        lines.append(f"  - warning: {warning}")
    for error in report["coverage"].get("errors", []):
        lines.append(f"  - error: {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rich logging outputs and paper-style figures")
    parser.add_argument("--input-dir", default="outputs/rich_logging_smoke")
    parser.add_argument("--figures-dir", default="outputs/rich_logging_smoke/paper_style_figures")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    input_dir = _rel(args.input_dir)
    figures_dir = _rel(args.figures_dir)
    files = {
        "train_metrics.csv": audit_train(input_dir),
        "eval_episode_metrics.csv": audit_eval_episode(input_dir),
        "eval_summary_metrics.csv": audit_summary(input_dir),
        "aircraft_timeseries.csv": audit_aircraft(input_dir),
        "missile_events.csv": audit_missile_events(input_dir),
        "missile_timeseries.csv": audit_missile_timeseries(input_dir),
        "reward_components.csv": audit_reward(input_dir),
        "training_efficiency.json": audit_efficiency(input_dir),
        "perturbation_eval_summary.csv": audit_perturbation(input_dir),
        "attention_metrics.csv": audit_attention(input_dir),
    }
    figures = audit_figures(figures_dir)
    coverage = audit_coverage(input_dir)
    overall = summarize_status(list(files.values()) + list(figures.values()) + [coverage])
    report = {
        "overall_status": overall,
        "files": files,
        "figures": figures,
        "coverage": coverage,
    }
    out_json = _rel(args.output_json) if args.output_json else input_dir / "rich_logging_audit_report.json"
    out_md = _rel(args.output_md) if args.output_md else input_dir / "rich_logging_audit_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(out_md, report)
    print(f"audit_json: {out_json}")
    print(f"overall_status: {overall}")
    return 0 if overall != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
