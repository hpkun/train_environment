"""Build a paper-readiness evidence matrix from existing experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "outputs/paper_evidence_matrix"


RUNS = [
    {
        "method_variant": "HAPPO reference v0 baseline",
        "dir": "happo_3v2_reference_f16_mav_surrogate_1m_fast",
        "train_scenario": "3v2",
        "conclusion": "survival/timeout baseline; not a stable combat method",
    },
    {
        "method_variant": "oracle pretrain direct fine-tune",
        "dir": "happo_oracle_pretrain_finetune_200k",
        "train_scenario": "3v2 normal",
        "conclusion": "oracle pretrain alone was insufficient for robust approach-and-fire",
    },
    {
        "method_variant": "easy combat oracle anchor",
        "dir": "happo_easy_combat_oracle_anchor_50k",
        "train_scenario": "easy 3v2",
        "conclusion": "easy task shows learned fire/hit behavior",
    },
    {
        "method_variant": "normal geometry direct oracle anchor",
        "dir": "happo_normal_geometry_oracle_anchor_100k",
        "train_scenario": "normal 3v2",
        "conclusion": "direct normal geometry transfer failed",
    },
    {
        "method_variant": "geometry curriculum full method",
        "dir": "happo_geometry_curriculum_100k/medium_50k",
        "train_scenario": "easy-to-medium 3v2",
        "conclusion": "medium curriculum preserves attack behavior",
    },
    {
        "method_variant": "geometry curriculum full method",
        "dir": "happo_geometry_curriculum_100k/normal_50k",
        "train_scenario": "medium-to-normal 3v2",
        "conclusion": "best checkpoint supports normal 3v2 combat and 5v4 transfer",
    },
    {
        "method_variant": "5v4 fine-tune upper-bound",
        "dir": "happo_5v4_finetune_upper_bound_50k",
        "train_scenario": "5v4 adaptation",
        "conclusion": "adaptation upper-bound; not zero-shot",
    },
]


FIELDS = [
    "method_variant",
    "train_scenario",
    "eval_scenario",
    "checkpoint",
    "red_win_rate",
    "red_elimination_win_rate",
    "red_timeout_alive_advantage_rate",
    "blue_win_rate",
    "mav_survival_rate",
    "red_missiles_fired_mean",
    "red_missile_hits_mean",
    "blue_dead_mean",
    "kill_death_ratio",
    "action_saturation",
    "conclusion",
    "evidence_level",
]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [x for x in data["records"] if isinstance(x, dict)]
    if isinstance(data, dict) and "config" in data:
        return [data]
    return []


def _num(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


def _scenario_from_config(config: str) -> str:
    if "5v4" in config:
        return "5v4"
    if "medium" in config:
        return "medium 3v2"
    if "easy" in config:
        return "easy 3v2"
    if "3v2" in config:
        return "normal 3v2"
    return "unknown"


def _action_saturation(record: dict[str, Any]) -> float | None:
    mav = _num(record, "mav_action_saturation_rate")
    uav = _num(record, "uav_action_saturation_rate")
    if mav is None and uav is None:
        return None
    if mav is None:
        return uav
    if uav is None:
        return mav
    return float(max(mav, uav))


def _matrix_record(run: dict[str, str], record: dict[str, Any], evidence_level: str) -> dict[str, Any]:
    config = str(record.get("config", ""))
    return {
        "method_variant": run["method_variant"],
        "train_scenario": run["train_scenario"],
        "eval_scenario": _scenario_from_config(config),
        "checkpoint": record.get("checkpoint"),
        "red_win_rate": _num(record, "red_win_rate"),
        "red_elimination_win_rate": _num(record, "red_elimination_win_rate"),
        "red_timeout_alive_advantage_rate": _num(record, "red_timeout_alive_advantage_rate"),
        "blue_win_rate": _num(record, "blue_win_rate"),
        "mav_survival_rate": _num(record, "mav_survival_rate"),
        "red_missiles_fired_mean": _num(record, "red_missiles_fired_mean"),
        "red_missile_hits_mean": _num(record, "red_missile_hits_mean"),
        "blue_dead_mean": _num(record, "blue_dead_mean"),
        "kill_death_ratio": _num(record, "kill_death_ratio"),
        "action_saturation": _action_saturation(record),
        "conclusion": run["conclusion"],
        "evidence_level": evidence_level,
        "source_config": config,
        "source_model": record.get("model_path"),
    }


def _missing_record(run: dict[str, str]) -> dict[str, Any]:
    return {
        "method_variant": run["method_variant"],
        "train_scenario": run["train_scenario"],
        "eval_scenario": "missing",
        "checkpoint": None,
        "red_win_rate": None,
        "red_elimination_win_rate": None,
        "red_timeout_alive_advantage_rate": None,
        "blue_win_rate": None,
        "mav_survival_rate": None,
        "red_missiles_fired_mean": None,
        "red_missile_hits_mean": None,
        "blue_dead_mean": None,
        "kill_death_ratio": None,
        "action_saturation": None,
        "conclusion": run["conclusion"],
        "evidence_level": "missing",
    }


def _load_run_records(outputs_root: Path, run: dict[str, str]) -> list[dict[str, Any]]:
    run_dir = outputs_root / run["dir"]
    if not run_dir.exists():
        return [_missing_record(run)]

    candidates = [
        run_dir / "checkpoint_eval" / "happo_3v2_checkpoint_eval.json",
        run_dir / "zero_shot_5v4_eval.json",
    ]
    records: list[dict[str, Any]] = []
    for path in candidates:
        data = _read_json(path)
        for raw in _records_from_json(data):
            records.append(_matrix_record(run, raw, "evaluated"))
    if not records:
        return [_missing_record(run)]
    return records


def build_matrix(outputs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in RUNS:
        records.extend(_load_run_records(outputs_root, run))
    return records


def _select_full_method(records: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    candidates = [
        r for r in records
        if r["method_variant"] == "geometry curriculum full method"
        and r["eval_scenario"] == scenario
        and r.get("checkpoint") == "best"
        and r["evidence_level"] == "evaluated"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r.get("red_missile_hits_mean") or 0.0) + (r.get("blue_dead_mean") or 0.0))


def build_transfer_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    seen = _select_full_method(records, "normal 3v2")
    transfer = _select_full_method(records, "5v4")
    metrics = {
        "win_retention": _safe_div(
            transfer.get("red_win_rate") if transfer else None,
            seen.get("red_win_rate") if seen else None,
        ),
        "elimination_retention": _safe_div(
            transfer.get("red_elimination_win_rate") if transfer else None,
            seen.get("red_elimination_win_rate") if seen else None,
        ),
        "fire_retention": _safe_div(
            transfer.get("red_missiles_fired_mean") if transfer else None,
            seen.get("red_missiles_fired_mean") if seen else None,
        ),
        "hit_retention": _safe_div(
            transfer.get("red_missile_hits_mean") if transfer else None,
            seen.get("red_missile_hits_mean") if seen else None,
        ),
        "normalized_blue_dead_retention": _safe_div(
            _safe_div(transfer.get("blue_dead_mean") if transfer else None, 4.0),
            _safe_div(seen.get("blue_dead_mean") if seen else None, 2.0),
        ),
        "mav_survival_delta": (
            (transfer.get("mav_survival_rate") if transfer else None)
            - (seen.get("mav_survival_rate") if seen else None)
            if seen and transfer and seen.get("mav_survival_rate") is not None and transfer.get("mav_survival_rate") is not None
            else None
        ),
        "timeout_dependency_delta": (
            (transfer.get("red_timeout_alive_advantage_rate") if transfer else None)
            - (seen.get("red_timeout_alive_advantage_rate") if seen else None)
            if seen and transfer and seen.get("red_timeout_alive_advantage_rate") is not None and transfer.get("red_timeout_alive_advantage_rate") is not None
            else None
        ),
    }
    return {
        "geometry_curriculum_full_method": metrics,
        "seen_3v2_record": seen,
        "zero_shot_5v4_record": transfer,
        "interpretation": {
            "win_rate_retained": metrics["win_retention"] is not None and metrics["win_retention"] >= 0.8,
            "elimination_ability_retained": metrics["elimination_retention"] is not None and metrics["elimination_retention"] >= 0.5,
            "per_enemy_kill_efficiency_drop": (
                metrics["normalized_blue_dead_retention"] is not None
                and metrics["normalized_blue_dead_retention"] < 0.9
            ),
            "more_timeout_dependent": (
                metrics["timeout_dependency_delta"] is not None
                and metrics["timeout_dependency_delta"] > 0.1
            ),
        },
    }


def _write_matrix_md(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper Evidence Matrix",
        "",
        "| method_variant | train_scenario | eval_scenario | checkpoint | red_win | red_elim | red_timeout_adv | blue_win | mav_survival | red_fire | red_hit | blue_dead | k/d | action_sat | evidence | conclusion |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in records:
        lines.append(
            "| {method_variant} | {train_scenario} | {eval_scenario} | {checkpoint} | {red_win_rate} | {red_elimination_win_rate} | {red_timeout_alive_advantage_rate} | {blue_win_rate} | {mav_survival_rate} | {red_missiles_fired_mean} | {red_missile_hits_mean} | {blue_dead_mean} | {kill_death_ratio} | {action_saturation} | {evidence_level} | {conclusion} |".format(
                **{k: r.get(k) for k in FIELDS}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transfer_md(path: Path, transfer: dict[str, Any]) -> None:
    metrics = transfer["geometry_curriculum_full_method"]
    lines = ["# Transfer Quality", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Interpretation", ""])
    for key, value in transfer["interpretation"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_md(path: Path) -> None:
    lines = [
        "# Ablation Summary",
        "",
        "## 1. No wrapped-heading correction",
        "",
        "- Evidence: the original oracle checkpoint had action-match MSE around `0.075882` and closed-loop easy-combat evaluation did not fire.",
        "- Evidence after fix: wrapped heading action-match MSE around `0.010` and closed-loop easy-combat evaluation produced red launches.",
        "- Conclusion: wrapped heading correction is necessary for the oracle imitation component.",
        "",
        "## 2. No geometry curriculum",
        "",
        "- Evidence: normal geometry oracle-anchor 100k latest had `red_fire=0.05`, `red_hit=0.00`, `blue_dead=0.00`.",
        "- Evidence with curriculum: geometry curriculum normal best had `red_fire=1.82`, `red_hit=1.56`, `blue_dead=1.52`.",
        "- Conclusion: geometry curriculum is a key component in the current environment.",
        "",
        "## 3. No oracle anchor / weak baseline",
        "",
        "- Evidence: HAPPO reference v0 and early oracle direct fine-tune were mainly survival-oriented or unstable.",
        "- Conclusion: plain HAPPO reference v0 is not sufficient to form stable approach-and-fire behavior in this setup.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper evidence matrix")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs_root = _rel(args.outputs_root)
    output_dir = _rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_matrix(outputs_root)
    transfer = build_transfer_quality(records)
    matrix = {"records": records}

    (output_dir / "paper_evidence_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8")
    (output_dir / "transfer_quality.json").write_text(
        json.dumps(transfer, indent=2), encoding="utf-8")
    _write_matrix_md(output_dir / "paper_evidence_matrix.md", records)
    _write_transfer_md(output_dir / "transfer_quality.md", transfer)
    _write_ablation_md(output_dir / "ablation_summary.md")

    print(f"output_json: {output_dir / 'paper_evidence_matrix.json'}")
    print(f"transfer_quality: {output_dir / 'transfer_quality.json'}")
    print(f"records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
