"""Checkpoint sweep for Pure-HAPPO low-level behavior diagnostics."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pure_happo_low_level_diagnostics import _find_run_dir, _write_md
from scripts.full_review_audit_utils import read_csv_rows, safe_float, select_checkpoints_from_train_log, write_csv_rows

DEFAULT_OUT = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_low_level"


def _candidate_steps(train_rows: list[dict]) -> dict[str, int]:
    def peak(metric: str) -> int:
        if not train_rows:
            return -1
        best = max(train_rows, key=lambda r: safe_float(r.get(metric)))
        return int(safe_float(best.get("total_steps", best.get("steps", -1))))
    return {
        "red_fire_peak": peak("red_missiles_fired"),
        "red_hit_peak": peak("missile_hits"),
        "red_win_peak": peak("red_win"),
        "mav_survival_peak": peak("mav_survival"),
    }


def _nearest_checkpoint(run_dir: Path, step: int) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists() or step < 0:
        return None
    candidates = []
    for p in ckpt_dir.glob("*/model.pt"):
        try:
            s = int("".join(ch for ch in p.parent.name if ch.isdigit()))
        except Exception:
            continue
        candidates.append((abs(s - step), p))
    return sorted(candidates, key=lambda x: x[0])[0][1] if candidates else None


def _eval_checkpoint(model: Path, episodes: int, max_steps: int, device: str, label: str, mode: str, out_dir: Path) -> dict:
    summary_json = out_dir / f"{label}_{mode}_summary.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "eval_happo_reference.py"),
        "--model", str(model),
        "--episodes", str(episodes),
        "--device", device,
        "--opponent-policy", "brma_rule",
        "--max-steps-override", str(max_steps),
        "--summary-json", str(summary_json),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    row = {"candidate": label, "mode": mode, "checkpoint": str(model), "returncode": proc.returncode}
    if proc.returncode != 0:
        row["status"] = "eval_failed"
        row["stderr_tail"] = proc.stderr[-1000:]
        return row
    records = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else []
    if records:
        rec = records[0]
        row.update({
            "status": "ok",
            "red_win": rec.get("red_win_rate", 0.0),
            "blue_win": rec.get("blue_win_rate", 0.0),
            "timeout": rec.get("timeout_rate", 0.0),
            "mav_survival": rec.get("mav_survival_rate", 0.0),
            "red_hit": rec.get("red_missile_hits_mean", 0.0),
            "blue_hit": rec.get("blue_missile_hits_mean", 0.0),
            "red_fire": rec.get("red_missiles_fired_mean", 0.0),
            "track_ok": "",
            "range_ok": "",
            "ata_ok": "",
            "ta_ok": "",
            "boresight_ok": "",
            "geometry_ok": "",
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv_rows(run_dir / "train_log.csv")
    candidates = {
        "latest": run_dir / "latest" / "model.pt",
        "best": run_dir / "best" / "model.pt",
    }
    for label, step in _candidate_steps(train_rows).items():
        candidates[label] = _nearest_checkpoint(run_dir, step) or (run_dir / "missing" / f"{label}_{step}.pt")

    rows = []
    for label, ckpt in candidates.items():
        if not ckpt.exists():
            rows.append({"candidate": label, "mode": "deterministic", "checkpoint": str(ckpt), "status": "missing"})
            rows.append({"candidate": label, "mode": "stochastic", "checkpoint": str(ckpt), "status": "missing"})
            continue
        rows.append(_eval_checkpoint(ckpt, args.episodes, args.max_steps, args.device, label, "deterministic", out_dir))
        # eval_happo_reference currently uses deterministic policy actions; keep the row explicit.
        stoch = dict(rows[-1])
        stoch["mode"] = "stochastic_not_supported_by_eval_loader"
        rows.append(stoch)
    write_csv_rows(out_dir / "checkpoint_low_level_sweep.csv", rows)
    _write_md(out_dir / "checkpoint_low_level_sweep.md", "Checkpoint Low-Level Sweep", "\n".join([
        f"- run_dir: `{run_dir}`",
        f"- candidates: {len(candidates)}",
        f"- ok eval rows: {sum(1 for r in rows if r.get('status') == 'ok')}",
        f"- missing rows: {sum(1 for r in rows if r.get('status') == 'missing')}",
        "",
        "Note: stochastic rows are marked unsupported unless the eval loader exposes stochastic action sampling.",
    ]))


if __name__ == "__main__":
    main()
