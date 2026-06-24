from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


LEGACY_REASONS = {"low_speed", "overshoot"}


def _find_missile_event_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "missile_events.csv":
        return [root]
    files = list(root.glob("missile_events.csv"))
    files.extend(root.glob("rich_logs/**/missile_events.csv"))
    files.extend(root.glob("**/rich_logs/**/missile_events.csv"))
    return sorted(set(files))


def _read_termination_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            event_type = str(row.get("event_type", ""))
            reason = str(row.get("raw_termination_reason", "") or row.get("termination_reason", ""))
            if event_type and event_type != "launch":
                rows.append(row)
            elif reason:
                rows.append(row)
    return rows


def audit(input_dir: Path) -> dict:
    files = _find_missile_event_files(input_dir)
    by_team: dict[str, Counter] = defaultdict(Counter)
    by_file: dict[str, dict] = {}
    legacy_hits: list[dict] = []

    for path in files:
        file_counter = Counter()
        for row in _read_termination_rows(path):
            team = str(row.get("owner_team", "") or row.get("team", "") or "unknown").lower()
            reason = str(row.get("raw_termination_reason", "") or row.get("termination_reason", "") or "unknown")
            by_team[team][reason] += 1
            file_counter[reason] += 1
            if reason in LEGACY_REASONS:
                legacy_hits.append({
                    "file": str(path),
                    "missile_id": row.get("missile_id", ""),
                    "owner_id": row.get("owner_id", ""),
                    "target_id": row.get("target_id", ""),
                    "reason": reason,
                })
        by_file[str(path)] = dict(file_counter)

    return {
        "input_dir": str(input_dir),
        "files": [str(path) for path in files],
        "termination_reasons_by_team": {team: dict(counter) for team, counter in by_team.items()},
        "termination_reasons_by_file": by_file,
        "legacy_reasons": sorted(LEGACY_REASONS),
        "legacy_reason_count": len(legacy_hits),
        "legacy_reason_records": legacy_hits[:50],
        "warning": (
            "legacy missile termination reasons found"
            if legacy_hits else ""
        ),
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Missile Termination Reason Audit",
        "",
        f"input: `{payload['input_dir']}`",
        f"files: {len(payload['files'])}",
        f"legacy reason count: {payload['legacy_reason_count']}",
        "",
        "| team | reason | count |",
        "|---|---|---:|",
    ]
    for team, reasons in sorted(payload["termination_reasons_by_team"].items()):
        for reason, count in sorted(reasons.items()):
            marker = " (legacy)" if reason in LEGACY_REASONS else ""
            lines.append(f"| {team} | {reason}{marker} | {count} |")
    if payload["legacy_reason_count"]:
        lines.extend([
            "",
            "Warning: legacy scripted-AAM-incompatible reasons were found in the input logs.",
            "New scripted AAM runs should not emit `low_speed` or `overshoot`.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit missile termination reasons in rich logs.")
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output-json", default="outputs/environment_audit/missile_termination_reasons.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/missile_termination_reasons.md")
    args = parser.parse_args()

    payload = audit(Path(args.input_dir))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, out_md)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
