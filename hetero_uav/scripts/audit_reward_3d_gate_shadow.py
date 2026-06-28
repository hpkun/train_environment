"""Offline 3D launch-gate shadow audit for tam_brma_scripted_reward_v1."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pure_happo_low_level_diagnostics import _find_run_dir, _write_md
from scripts.full_review_audit_utils import pearson_corr, read_csv_rows, safe_float, write_csv_rows

DEFAULT_OUT = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_low_level"


def _score_near_threshold(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / limit))


def _shadow_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        range_m = safe_float(r.get("range_3d_m"))
        ata = abs(safe_float(r.get("ATA_3d_rad")))
        ta = safe_float(r.get("TA_3d_rad"))
        bore = abs(safe_float(r.get("boresight_3d_rad")))
        range_score = max(0.0, min(1.0, 1.0 - abs(range_m - 12000.0) / 12000.0))
        ata_score = _score_near_threshold(ata, 0.35)
        ta_score = max(0.0, min(1.0, ta / 0.25))
        boresight_score = _score_near_threshold(bore, 0.35)
        g3 = range_score * ata_score * ta_score * boresight_score
        g2 = safe_float(r.get("reward_g_own"))
        launch_ok = int(safe_float(r.get("launch_geometry_ok_3d")) > 0.5)
        out.append({
            **r,
            "a_own_2d": safe_float(r.get("reward_a_own")),
            "t_rear_2d": safe_float(r.get("reward_t_rear")),
            "d_gate_2d": safe_float(r.get("reward_d_gate")),
            "g_own_2d": g2,
            "uav_gate_sit_2d": g2,
            "range_score_3d": range_score,
            "ata_score_3d": ata_score,
            "ta_score_3d": ta_score,
            "boresight_score_3d": boresight_score,
            "g_own_3d_shadow": g3,
            "gate_sit_3d_shadow": g3,
            "launch_geometry_ok_3d": launch_ok,
        })
    return out


def _top_rate(rows: list[dict], score_key: str, ok_key: str = "launch_geometry_ok_3d") -> float:
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda x: safe_float(x.get(score_key)), reverse=True)
    top_n = max(1, int(math.ceil(0.10 * len(ordered))))
    top = ordered[:top_n]
    return sum(1 for r in top if safe_float(r.get(ok_key)) > 0.5) / max(len(top), 1)


def _summary(rows: list[dict]) -> list[dict]:
    g2 = [safe_float(r.get("g_own_2d")) for r in rows]
    g3 = [safe_float(r.get("g_own_3d_shadow")) for r in rows]
    ok = [safe_float(r.get("launch_geometry_ok_3d")) for r in rows]
    out = [{
        "samples": len(rows),
        "corr_g_own_2d_launch_geometry_ok_3d": pearson_corr(g2, ok),
        "corr_g_own_3d_shadow_launch_geometry_ok_3d": pearson_corr(g3, ok),
        "p_geometry_ok_top10_g_own_2d": _top_rate(rows, "g_own_2d"),
        "p_geometry_ok_top10_g_own_3d_shadow": _top_rate(rows, "g_own_3d_shadow"),
        "positive_rate_2d": sum(1 for v in g2 if v > 0.0) / max(len(g2), 1),
        "positive_rate_3d_shadow": sum(1 for v in g3 if v > 0.0) / max(len(g3), 1),
        "launch_geometry_ok_rate": sum(1 for v in ok if v > 0.5) / max(len(ok), 1),
    }]
    s = out[0]
    s["classification"] = (
        "REWARD_LAUNCH_GATE_MISMATCH_HIGH"
        if s["corr_g_own_3d_shadow_launch_geometry_ok_3d"] > s["corr_g_own_2d_launch_geometry_ok_3d"] + 0.05
        else "INCONCLUSIVE_OR_LOW_MISMATCH"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir()
    out_dir = Path(args.output_dir)
    rows = read_csv_rows(run_dir / "audit_trace" / "launch_gate_step.csv")
    shadow = _shadow_rows(rows)
    summary = _summary(shadow)
    write_csv_rows(out_dir / "reward_3d_gate_shadow_step.csv", shadow)
    write_csv_rows(out_dir / "reward_3d_gate_shadow_summary.csv", summary)
    s = summary[0] if summary else {}
    _write_md(out_dir / "reward_3d_gate_shadow_audit.md", "Reward 3D Gate Shadow Audit", "\n".join([
        f"- samples: {s.get('samples', 0)}",
        f"- corr 2D gate vs 3D geometry: {s.get('corr_g_own_2d_launch_geometry_ok_3d', 0):.6g}",
        f"- corr 3D shadow vs 3D geometry: {s.get('corr_g_own_3d_shadow_launch_geometry_ok_3d', 0):.6g}",
        f"- P(geometry_ok | top10 2D): {s.get('p_geometry_ok_top10_g_own_2d', 0):.6g}",
        f"- P(geometry_ok | top10 3D shadow): {s.get('p_geometry_ok_top10_g_own_3d_shadow', 0):.6g}",
        f"- classification: {s.get('classification', '')}",
        "",
        "This is a shadow diagnostic only; tam_brma_scripted_reward_v1 is unchanged.",
    ]))


if __name__ == "__main__":
    main()
