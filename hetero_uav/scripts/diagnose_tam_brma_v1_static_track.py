"""Static track diagnostic for tam_brma_scripted_reward_v1 — reset + 1 step.

Verify that obs cache fix enables MAV shared track for red UAVs.
"""
from __future__ import annotations
import json, csv, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_env import make_env

CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_brma_scripted_reward_v1.yaml"
OUT = ROOT / "outputs/tam_brma_static_track_smoke"
OUT.mkdir(parents=True, exist_ok=True)

env = make_env(CONFIG)
env.reset(seed=0)

def _dist3d(sim_a, sim_b):
    return float(np.linalg.norm(np.array(sim_a.get_position()) - np.array(sim_b.get_position())))

mav = env.red_planes.get("red_0")
uav1 = env.red_planes.get("red_1")
uav2 = env.red_planes.get("red_2")
b0 = env.blue_planes.get("blue_0")
b1 = env.blue_planes.get("blue_1")

def _check(label, step_num):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Agent state
    print(f"\n--- Agent state ---")
    for aid in ["red_0","red_1","red_2","blue_0","blue_1"]:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        role = env.agent_roles.get(aid, "?")
        alive = sim.is_alive if sim else False
        alt = float(sim.get_geodetic()[2]) if sim and alive else 0
        spd = float(np.linalg.norm(sim.get_velocity())) if sim and alive else 0
        pos = sim.get_position() if sim and alive else (0,0,0)
        print(f"  {aid} role={role} alive={alive} alt={alt:.0f}m spd={spd:.0f}m/s pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")

    # Distances
    print(f"\n--- Distances ---")
    print(f"  uav_direct_range_m: {env.uav_direct_observation_range_m}")
    print(f"  mav_observation_range_m: {env.mav_observation_range_m}")
    for red, rname in [(uav1,"red_1"),(uav2,"red_2")]:
        for blue, bname in [(b0,"blue_0"),(b1,"blue_1")]:
            d = _dist3d(red, blue)
            direct = d <= env.uav_direct_observation_range_m
            print(f"  {rname} -> {bname}: {d:.0f}m  direct_ok={direct}")

    mav_b0 = _dist3d(mav, b0); mav_b1 = _dist3d(mav, b1)
    print(f"  MAV -> blue_0: {mav_b0:.0f}m  shared_ok={mav_b0 <= env.mav_observation_range_m}")
    print(f"  MAV -> blue_1: {mav_b1:.0f}m  shared_ok={mav_b1 <= env.mav_observation_range_m}")

    # Obs cache
    print(f"\n--- _last_step_obs ---")
    obs = env._last_step_obs
    print(f"  populated: {bool(obs)}")
    if obs:
        print(f"  keys: {list(obs.keys())}")

    # Per-red-UAV track check
    rows = []
    for red, rid in [(uav1,"red_1"),(uav2,"red_2")]:
        agent_obs = obs.get(rid, {})
        print(f"\n--- {rid} obs ---")
        print(f"  present: {bool(agent_obs)}")
        if not agent_obs:
            continue
        keys = sorted(agent_obs.keys())
        print(f"  keys ({len(keys)}): {keys}")
        eam = agent_obs.get("enemy_alive_mask")
        eom = agent_obs.get("enemy_observed_mask")
        ets = agent_obs.get("enemy_track_source")
        egs = agent_obs.get("enemy_geo_states")
        print(f"  enemy_alive_mask:    {eam}")
        print(f"  enemy_observed_mask: {eom}")
        print(f"  enemy_track_source:  {ets}")
        print(f"  enemy_geo_states:    {np.array(egs).round(3) if egs is not None else 'None'}")

        for blue, bid in [(b0,"blue_0"),(b1,"blue_1")]:
            # Manual expected
            own_direct = _dist3d(red, blue) <= env.uav_direct_observation_range_m
            mav_shared_expected = (mav.is_alive and
                                   _dist3d(mav, blue) <= env.mav_observation_range_m)

            # Obs track source for this blue (index 0 or 1)
            b_idx = 0 if bid=="blue_0" else 1
            obs_direct = 0; obs_shared = 0
            if ets is not None and len(ets) > b_idx:
                ts_row = ets[b_idx]
                obs_direct = int(ts_row[0]) if len(ts_row) > 0 else 0
                obs_shared = int(ts_row[1]) if len(ts_row) > 1 else 0

            # _has_launch_track
            hlt = False; hlt_src = "none"
            try:
                hlt_raw = env._has_launch_track(rid, bid)
                hlt = hlt_raw[0] if isinstance(hlt_raw, tuple) else bool(hlt_raw)
                hlt_src = hlt_raw[1] if isinstance(hlt_raw, tuple) and len(hlt_raw)>1 else ("mav_shared" if (obs_shared and obs_direct==0) else ("direct" if obs_direct else "none"))
            except Exception as e:
                hlt_src = f"error:{e}"

            flags = []
            if mav_shared_expected and obs_shared == 0:
                flags.append("MISMATCH_MAV_SHARED_NOT_WRITTEN")
            if obs_shared == 1 and not hlt:
                flags.append("MISMATCH_HAS_LAUNCH_TRACK_IGNORES_OBS")
            if mav_shared_expected and obs_shared == 1 and hlt:
                flags.append("OK_SHARED_TRACK_WORKS")
            if own_direct and obs_direct == 1 and hlt:
                flags.append("OK_DIRECT_TRACK_WORKS")
            flag_str = " | ".join(flags) if flags else "NO_FLAG"

            print(f"  {rid}->{bid}: direct_exp={own_direct} shared_exp={mav_shared_expected} obs_d={obs_direct} obs_s={obs_shared} _has_track={hlt} flags=[{flag_str}]")
            rows.append({
                "step": step_num, "agent_id": rid, "target_id": bid,
                "own_direct_exp": int(own_direct), "mav_shared_exp": int(mav_shared_expected),
                "obs_track_direct": obs_direct, "obs_track_mav_shared": obs_shared,
                "has_launch_track": int(hlt), "flags": flag_str,
            })
    return rows

# Reset check
reset_rows = _check("AFTER RESET (step=0)", 0)

# 1 step
actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
obs, rewards, terminated, truncated, info = env.step(actions)
step1_rows = _check("AFTER 1 STEP", 1)

# Write CSVs
for fname, rows in [("static_track_reset.csv", reset_rows), ("static_track_step1.csv", step1_rows)]:
    with open(OUT / fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# Summary JSON
summary = {
    "mav_alive": bool(mav.is_alive),
    "mav_to_blue_0_m": round(_dist3d(mav, b0), 0),
    "mav_to_blue_1_m": round(_dist3d(mav, b1), 0),
    "uav_direct_range_m": env.uav_direct_observation_range_m,
    "mav_observation_range_m": env.mav_observation_range_m,
    "reset_any_shared_track": any("OK_SHARED_TRACK_WORKS" in r["flags"] for r in reset_rows),
    "reset_any_mismatch": any("MISMATCH" in r["flags"] for r in reset_rows),
    "step1_any_shared_track": any("OK_SHARED_TRACK_WORKS" in r["flags"] for r in step1_rows),
    "step1_any_mismatch": any("MISMATCH" in r["flags"] for r in step1_rows),
}
with open(OUT / "static_track_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Report
lines = ["# Static Track Diagnosis — tam_brma_scripted_reward_v1", ""]
lines.append(f"## Reset (step=0)")
for r in reset_rows:
    lines.append(f"- {r['agent_id']}->{r['target_id']}: direct_exp={r['own_direct_exp']} shared_exp={r['mav_shared_exp']} obs_d={r['obs_track_direct']} obs_s={r['obs_track_mav_shared']} has_track={r['has_launch_track']} [{r['flags']}]")
lines.append("")
lines.append(f"## After 1 step")
for r in step1_rows:
    lines.append(f"- {r['agent_id']}->{r['target_id']}: direct_exp={r['own_direct_exp']} shared_exp={r['mav_shared_exp']} obs_d={r['obs_track_direct']} obs_s={r['obs_track_mav_shared']} has_track={r['has_launch_track']} [{r['flags']}]")
lines.append("")
any_ok = any("OK" in r["flags"] for r in reset_rows + step1_rows)
any_mismatch = any("MISMATCH" in r["flags"] for r in reset_rows + step1_rows)
if any_ok:
    lines.append("## Conclusion: TRACK WORKS STATICALLY")
    lines.append("MAV shared track is correctly written to obs and read by _has_launch_track in static conditions.")
    lines.append("Previous audit track_ok=0 is either due to policy moving out of shared range during rollout, or audit metric definition.")
elif any_mismatch:
    lines.append("## Conclusion: MISMATCH FOUND")
    lines.append("See flags above. Check _build_mav_shared_geo_obs or _has_launch_track implementation.")
else:
    lines.append("## Conclusion: NO TRACK CONDITION MET")
    lines.append("In this initial geometry, neither direct nor MAV shared track conditions are satisfied at reset.")

with open(OUT / "static_track_report.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nOutputs: {OUT}")
print(f"  static_track_report.md")
print(f"  static_track_reset.csv")
print(f"  static_track_step1.csv")
print(f"  static_track_summary.json")
if hasattr(env, "close"): env.close()
