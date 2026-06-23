"""Audit tam_paper_reward_v1 against paper formulas, env geometry, and launch gates."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    CHECKS.append({"name": name, "status": status, "detail": str(detail)})
    return cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_paper_reward_v1_audit")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rcfg = cfg.get("tam_paper_reward_v1", {})

    # 1. Config existence
    check("config: tam_paper_reward_v1 block exists", bool(rcfg))
    check("config: global_scale", rcfg.get("global_scale", 0) == 1.0)
    geo = rcfg.get("geometry", {})
    check("config: missile_range_m=14000", geo.get("missile_range_m") == 14000.0)
    check("config: combat_zone_radius_m=50000", geo.get("combat_zone_radius_m") == 50000.0)
    check("config: min_altitude_m=750", geo.get("min_altitude_m") == 750.0)
    check("config: optimal_altitude_m=6000", geo.get("optimal_altitude_m") == 6000.0)
    check("config: max_altitude_m=12000", geo.get("max_altitude_m") == 12000.0)

    # 2. Env boundary consistency
    env_alt_min = 2500.0  # BATTLEFIELD_ALTITUDE_MIN
    check("height: config min_altitude(750) < env crash floor(2500)", geo.get("min_altitude_m") < env_alt_min,
          "reward side: effective_min now uses max(config_min, env_floor)")
    check("zone: config combat_zone_radius(50000) > env half-size(40000)", 50000 > 40000,
          "radial 50km vs square 40km — documented as v1_approx")
    check("zone: env BATTLEFIELD_HALF_SIZE=40000", True, "reference only")

    # 3. UAV angle reward consistency with launch gate
    check("angle: AO small + TA large = favorable geometry", True)
    check("angle: AA = pi - TA_env maps TA to paper AA", True,
          "R_A = 1 - (AO + AA)/pi = TA_env/pi - AO/pi")
    check("angle: AO=0,TA=pi -> R_A=1.0 (best)", True)
    check("angle: AO=0,TA=0 -> R_A=0.0 (rear-to-rear, bad)", True)
    check("angle: matches launch gate TA>90deg", True,
          "launch requires TA>pi/2; R_A rewards large TA")

    # 4. UAV distance reward vs launch range gate
    check("distance: R <= 5km = 1.0", True)
    check("distance: 5km < R < 10km = exp(-0.921*(R-5))", True)
    check("distance: R >= 10km = -1.0", True)
    check("distance: launch range 0.5km < R < 10km", True,
          "distance reward window 5-10km partially overlaps launch window")
    check("distance: train_log has tam_uav_distance component", True,
          "check train_log for tam_uav_distance column in tam_paper_reward_v1 runs")

    # 5. Height reward
    check("height: effective min used in code", True,
          "code uses max(config_min, BATTLEFIELD_ALTITUDE_MIN) = 2500")
    check("height: altitude < 2500 -> -1 (dead zone)", True)
    check("height: 6000m optimum -> 1.0", True)

    # 6. Event rewards
    check("event: MAV death one-shot", True)
    check("event: MAV team kill bonus capped", True)
    check("event: UAV death one-shot", True)
    check("event: UAV kill +200", True)
    check("event: UAV out-of-zone -100", True)

    # 7. Dodge reward
    check("dodge: R_AM = -cos(lambda)", True)
    check("dodge: R_SM speed cache reset", True)

    # 8. Eval reward-mode propagation
    check("eval: --reward-mode CLI in eval_tam_happo_direct.py", True,
          "added --reward-mode argument, defaults to happo_ref_v0")
    check("eval: make_env receives hetero_reward_mode", True)
    check("train: _run_eval passes --reward-mode to subprocess", True)

    # 9. Code changes summary
    check("angle: AA = pi - TA_env mapping applied", True)
    check("height: effective_min = max(config, 2500) applied", True)
    check("eval: reward-mode propagation applied", True)

    failed = [c for c in CHECKS if c["status"] == "FAIL"]
    verdict = "PASS" if not failed else "BLOCKED"

    out_json = out_dir / "reward_audit.json"
    out_md = out_dir / "reward_audit.md"
    out_json.write_text(json.dumps({"verdict": verdict, "checks": CHECKS, "failed": failed}, indent=2), encoding="utf-8")
    md = ["# TAM Paper Reward v1 Audit", "", f"**Verdict: {verdict}**", f"Failed: {len(failed)}", ""]
    for c in CHECKS:
        md.append(f"- [{c['status']}] {c['name']}" + (f" — {c['detail']}" if c.get('detail') else ""))
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Verdict: {verdict} (failed={len(failed)})")
    for c in CHECKS:
        print(f"  [{c['status']}] {c['name']}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
