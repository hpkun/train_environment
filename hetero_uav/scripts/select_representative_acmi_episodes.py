"""Select representative ACMI episodes for progress-report trajectory figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACMI_DIR = "outputs/happo_geometry_curriculum_100k/normal_50k/acmi"
DEFAULT_OUTPUT_DIR = "outputs/progress_report_figures"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_num(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _blue_dead(summary: dict[str, Any]) -> float:
    if "blue_dead" in summary:
        return _as_num(summary, "blue_dead")
    blue_alive = summary.get("blue_alive_final")
    config = str(summary.get("config", ""))
    total = 4 if "5v4" in config else 2
    try:
        return max(0.0, total - float(blue_alive))
    except (TypeError, ValueError):
        return _as_num(summary, "red_missile_hits")


def _scenario(summary_path: Path, summary: dict[str, Any]) -> str:
    text = f"{summary_path.name} {summary.get('config', '')}".lower()
    if "5v4" in text:
        return "5v4"
    if "3v2" in text:
        return "3v2"
    return "unknown"


def _score(summary: dict[str, Any], scenario: str) -> float:
    red_hits = _as_num(summary, "red_missile_hits")
    red_fire = _as_num(summary, "red_missiles_fired")
    blue_dead = _blue_dead(summary)
    mav_alive = 1.0 if bool(summary.get("mav_alive", False)) else 0.0
    score = red_hits * 3.0 + blue_dead * 3.0 + red_fire + mav_alive * 2.0
    outcome = str(summary.get("outcome", ""))
    if scenario == "3v2" and outcome == "red_win_elimination":
        score += 10.0
    if scenario == "5v4" and outcome == "red_win_elimination":
        score += 8.0
    if "timeout" in outcome and scenario == "5v4":
        score += 1.0
    return score


def _candidate_from_summary(path: Path) -> dict[str, Any] | None:
    summary = _read_json(path)
    if not summary:
        return None
    acmi = Path(str(summary.get("output_acmi", "")))
    if not acmi.is_absolute():
        acmi = path.with_name(path.stem.replace("_summary", "") + ".acmi")
    scenario = _scenario(path, summary)
    return {
        "scenario": scenario,
        "summary_json": str(path),
        "acmi": str(acmi),
        "outcome": summary.get("outcome", "unknown"),
        "red_missile_hits": _as_num(summary, "red_missile_hits"),
        "red_missiles_fired": _as_num(summary, "red_missiles_fired"),
        "blue_dead": _blue_dead(summary),
        "red_alive_final": _as_num(summary, "red_alive_final"),
        "mav_alive": bool(summary.get("mav_alive", False)),
        "steps": _as_num(summary, "steps"),
        "attack_score": _score(summary, scenario),
    }


def _select(candidates: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    selected = [c for c in candidates if c["scenario"] == scenario and Path(c["acmi"]).exists()]
    if not selected:
        return None
    if scenario == "3v2":
        preferred = [
            c for c in selected
            if c["outcome"] == "red_win_elimination"
            and c["red_missile_hits"] >= 1
            and c["blue_dead"] >= 1
            and c["mav_alive"]
        ]
    else:
        preferred = [
            c for c in selected
            if c["red_missile_hits"] >= 1
            and c["blue_dead"] >= 1
            and c["mav_alive"]
            and c["red_alive_final"] >= 1
        ]
    pool = preferred or selected
    return max(pool, key=lambda c: c["attack_score"])


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Representative Episode Selection", ""]
    lines.append(f"- status: `{result['status']}`")
    lines.append(f"- note: {result['note']}")
    for scenario in ("3v2", "5v4"):
        item = result["selected"].get(scenario)
        lines.append("")
        lines.append(f"## {scenario}")
        if not item:
            lines.append("- selected: none")
            continue
        lines.extend([
            f"- acmi: `{item['acmi']}`",
            f"- summary: `{item['summary_json']}`",
            f"- outcome: `{item['outcome']}`",
            f"- red hits: `{item['red_missile_hits']}`",
            f"- blue dead: `{item['blue_dead']}`",
            f"- mav alive: `{item['mav_alive']}`",
            f"- attack score: `{item['attack_score']}`",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select representative ACMI episodes")
    parser.add_argument("--acmi-dir", default=DEFAULT_ACMI_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    acmi_dir = _rel(args.acmi_dir)
    output_dir = _rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        c for c in (_candidate_from_summary(path) for path in acmi_dir.glob("*.json"))
        if c is not None
    ] if acmi_dir.exists() else []
    selected = {"3v2": _select(candidates, "3v2"), "5v4": _select(candidates, "5v4")}
    if not candidates:
        status, note = "missing", "No ACMI summaries were found. No training or rollout was run."
    elif not selected["3v2"] or not selected["5v4"]:
        status, note = "limited", "Existing ACMI candidates are incomplete; selected what is available."
    else:
        status, note = "limited" if len(candidates) <= 3 else "ok", (
            "Selection uses existing exported candidates only; 5v4 representative may be a timeout attack-transfer episode."
        )

    result = {
        "status": status,
        "note": note,
        "candidate_count": len(candidates),
        "selected": selected,
        "selection_rule": {
            "3v2": "prefer red_win_elimination, red hits, blue deaths, MAV alive",
            "5v4": "prefer red hits, blue deaths, MAV alive, high red_alive_final; timeout allowed if it best shows attack transfer",
        },
    }
    json_path = output_dir / "representative_episode_selection.json"
    md_path = output_dir / "representative_episode_selection.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(md_path, result)
    print(f"selection_json: {json_path}")
    print(f"selection_md: {md_path}")
    print(f"status: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
