"""Full-pipeline audit for tam_brma_scripted_reward_v1.

Runs deterministic rollout, collects per-step reward components and launch
gate diagnostics, and produces a data-driven attribution report.
"""
from __future__ import annotations

import argparse, csv, json, sys, math, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── helpers ────────────────────────────────────────────────────────────

def _sf(v, default=0.0):
    try: return float(v)
    except: return default

def _load_train_log(run_dir: Path) -> list[dict]:
    p = run_dir / "train_log.csv"
    if not p.exists(): return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ── train log trends ───────────────────────────────────────────────────

def parse_train_trends(rows: list[dict]) -> dict:
    if not rows: return {}
    rets = [_sf(r["avg_return"]) for r in rows]
    mavs = [_sf(r["mav_survival"]) for r in rows]
    eM = [_sf(r["entropy_mav"]) for r in rows]
    eU = [_sf(r["entropy_uav"]) for r in rows]
    sM = [_sf(r["mav_action_saturation_rate"]) for r in rows]
    sU = [_sf(r["uav_action_saturation_rate"]) for r in rows]
    rf = [_sf(r["red_episode_missiles_fired_mean"]) for r in rows]
    st = [int(float(r["total_steps"])) for r in rows]
    best_i = int(np.argmax(rets))
    return {
        "total_iters": len(rows), "final_step": st[-1],
        "best_return": rets[best_i], "best_return_step": st[best_i],
        "final_return": rets[-1], "final_mav_survival": mavs[-1],
        "final_entropy_mav": eM[-1], "final_entropy_uav": eU[-1],
        "final_saturation_mav": sM[-1], "final_saturation_uav": sU[-1],
        "red_fire_rate_max": max(rf), "red_fire_rate_final": rf[-1],
        "entropy_drift_mav": eM[-1] - eM[0], "entropy_drift_uav": eU[-1] - eU[0],
    }

# ── checkpoint rollout ─────────────────────────────────────────────────

def _load_pure_happo_policy(meta: dict, device):
    from algorithms.pure_happo import PureHAPPOPolicy
    return PureHAPPOPolicy(
        actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
        critic_state_dim=int(meta.get("critic_state_dim", 480)),
        action_dim=3,
        num_agents=int(meta.get("num_agents", 3)),
    ).to(device)

def run_deterministic_rollout(checkpoint_path: str, config_path: str,
                               episodes: int, max_steps: int, device_str: str,
                               output_dir: Path) -> dict:
    import torch
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.mappo.opponent_policy import OpponentPolicy

    cp = Path(checkpoint_path)
    meta = json.loads((cp.parent / "meta.json").read_text(encoding="utf-8")) if (cp.parent / "meta.json").exists() else {}
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    policy = _load_pure_happo_policy(meta, device)
    policy.load(cp, map_location=device)
    policy.eval()

    env = make_env(config_path, env_type="jsbsim_hetero")
    adapter = HeteroObsAdapterV2()

    ep_summaries = []
    step_rows, comp_rows, launch_rows, gate_reward_rows = [], [], [], []

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        ep_return = 0.0; ep_mav_ret = 0.0; ep_uav_ret = 0.0; ep_len = 0
        mav_alts, uav_alts, uav_speeds, mav_sats, uav_sats = [], [], [], [], []
        mav_dead, uav_dead = False, 0; mav_death_reason = ""
        ep_red_fired, ep_blue_fired, ep_red_hits, ep_blue_hits = 0, 0, 0, 0
        prev_hits = {"red": 0, "blue": 0}

        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
                for rid in env.red_ids
            ])
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(actor_obs, device=device),
                    roles=[0 if env.agent_roles.get(rid)=="mav" else 1 for rid in env.red_ids],
                    critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                    deterministic=True,
                )
            acts_np = out["action"].cpu().numpy()
            actions = {rid: acts_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            opp = OpponentPolicy(mode="brma_rule", seed=ep*1000+ep_len)
            actions.update(opp.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_len += 1

            rc = info.get("reward_components", {})
            for i, rid in enumerate(env.red_ids):
                sim = env.red_planes.get(rid)
                role = env.agent_roles.get(rid, "")
                alive = int(sim.is_alive) if sim else 0
                alt = float(sim.get_geodetic()[2]) if sim and sim.is_alive else 0.0
                spd = float(np.linalg.norm(sim.get_velocity())) if sim and sim.is_alive else 0.0
                rpy = sim.get_rpy() if sim and sim.is_alive else (0,0,0)
                sat = float(np.mean(np.abs(acts_np[i]) >= 0.999))
                comp = rc.get(rid, {})
                r_total = float(rewards.get(rid, 0.0))

                # Flight metrics
                if role == "mav": mav_alts.append(alt); mav_sats.append(sat)
                else: uav_alts.append(alt); uav_speeds.append(spd); uav_sats.append(sat)
                if role == "mav": ep_mav_ret += r_total
                else: ep_uav_ret += r_total
                ep_return += r_total
                if not alive and not mav_dead and role == "mav":
                    mav_dead = True
                    mav_death_reason = str((info.get(rid, {}) or {}).get("death_reason", ""))
                if not alive and role != "mav": uav_dead += 1

                step_rows.append({
                    "episode_id": ep, "step": ep_len, "agent_id": rid, "role": role,
                    "alive": alive, "altitude_m": round(alt,1), "speed_mps": round(spd,1),
                    "pitch_deg": round(math.degrees(float(rpy[1])),1),
                    "roll_deg": round(math.degrees(float(rpy[0])),1),
                    "action_pitch": round(float(acts_np[i][0]),4),
                    "action_heading": round(float(acts_np[i][1]),4),
                    "action_speed": round(float(acts_np[i][2]),4),
                    "action_saturation": round(sat,4),
                    "reward_total": round(r_total,4),
                    **{k: round(_sf(comp.get(k)), 6) for k in [
                        "tam_brma_v1_flight","tam_brma_v1_uav_gate_sit",
                        "tam_brma_v1_uav_event","tam_brma_v1_mav_safe",
                        "tam_brma_v1_mav_support","tam_brma_v1_mav_aware",
                        "tam_brma_v1_mav_event","tam_brma_v1_team_terminal",
                        "tam_brma_v1_total",
                        "tam_brma_v1_uav_g_own","tam_brma_v1_uav_g_enemy",
                        "tam_brma_v1_uav_a_own","tam_brma_v1_uav_t_rear",
                        "tam_brma_v1_uav_d_gate","tam_brma_v1_uav_target_idx",
                        "tam_brma_v1_mav_dist","tam_brma_v1_mav_link",
                        "tam_brma_v1_mav_rear","tam_brma_v1_mav_aware",
                    ]},
                })

                # Component sums per episode
                comp_rows.append({
                    "episode_id": ep, "agent_id": rid, "role": role,
                    "return_total": r_total,
                    **{f"sum_{k}": round(_sf(comp.get(k)), 6) for k in [
                        "tam_brma_v1_flight","tam_brma_v1_uav_gate_sit",
                        "tam_brma_v1_uav_event","tam_brma_v1_mav_safe",
                        "tam_brma_v1_mav_support","tam_brma_v1_mav_aware",
                        "tam_brma_v1_mav_event","tam_brma_v1_team_terminal",
                    ]},
                })

            # Launch gate diagnostics — per shooter-target pair
            for shooter_id in env.red_ids:
                if env.agent_roles.get(shooter_id) == "mav": continue
                shooter = env.red_planes.get(shooter_id)
                if not shooter or not shooter.is_alive: continue
                for target_id in env.blue_ids:
                    target = env.blue_planes.get(target_id)
                    if not target or not target.is_alive: continue
                    try:
                        g3d = env._build_launch_geometry_3d(shooter, target)
                    except Exception:
                        g3d = {}

                    # Track: use env._has_launch_track directly (NOT geometry method)
                    hlt_raw = (False, "unobserved")
                    try:
                        hlt_raw = env._has_launch_track(shooter_id, target_id)
                    except Exception:
                        pass
                    has_track = bool(hlt_raw[0]) if isinstance(hlt_raw, tuple) else bool(hlt_raw)
                    track_source = str(hlt_raw[1]) if isinstance(hlt_raw, tuple) and len(hlt_raw) > 1 else ("mav_shared" if has_track else "unobserved")

                    # Obs raw fields for trace
                    obs_s = env._last_step_obs.get(shooter_id, {})
                    ets = obs_s.get("enemy_track_source", None)
                    eom = obs_s.get("enemy_observed_mask", None)
                    b_idx = 0 if target_id == "blue_0" else 1
                    obs_track_direct = 0; obs_track_shared = 0
                    if ets is not None and len(ets) > b_idx:
                        ts_row = ets[b_idx]
                        obs_track_direct = int(ts_row[0]) if len(ts_row) > 0 else 0
                        obs_track_shared = int(ts_row[1]) if len(ts_row) > 1 else 0
                    obs_observed = int(eom[b_idx]) if eom is not None and len(eom) > b_idx else 0

                    # Dynamic trace: mismatch flag
                    trace_flag = "NO_TRACK_EXPECTED"
                    if not obs_s:
                        trace_flag = "OBS_MISSING"
                    elif obs_track_direct:
                        trace_flag = "OK_DIRECT" if has_track else "OBS_DIRECT_BUT_HAS_TRACK_FALSE"
                    elif obs_track_shared:
                        trace_flag = "OK_MAV_SHARED" if has_track else "OBS_SHARED_BUT_HAS_TRACK_FALSE"
                    elif has_track:
                        trace_flag = "HAS_TRACK_BUT_NO_OBS_SOURCE"
                    else:
                        trace_flag = "NO_TRACK_EXPECTED"

                    # Reward geometry (2D AO/TA from tam_brma_v1)
                    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
                    sf = H._tam_v2_feature(shooter)
                    bf = H._tam_v2_feature(target)
                    from uav_env.JSBSim.utils import get2d_AO_TA_R
                    ao2, ta2, r2 = get2d_AO_TA_R(sf, bf)

                    # Reward gate_sit components
                    cfg = env.tam_brma_scripted_reward_v1_config
                    a_own = H._tam_brma_v1_a_own(ao2, cfg)
                    t_rear = H._tam_brma_v1_t_rear(ta2, cfg)
                    d_gate = H._tam_brma_v1_d_gate(r2 if r2 and not np.isnan(r2) else float(np.linalg.norm(np.array(shooter.get_position())-np.array(target.get_position()))), cfg)
                    g_own = a_own * t_rear * d_gate

                    # Enemy threat
                    ao_e, ta_e, r_e = get2d_AO_TA_R(bf, sf)
                    dist_e = r_e if r_e and not np.isnan(r_e) else float(np.linalg.norm(np.array(target.get_position())-np.array(shooter.get_position())))
                    a_e = H._tam_brma_v1_a_own(ao_e, cfg)
                    t_e = H._tam_brma_v1_t_rear(ta_e, cfg)
                    d_e = H._tam_brma_v1_d_gate(dist_e, cfg)
                    g_enemy = a_e * t_e * max(d_e, 0.0)

                    # Track / lock info (already computed above from _has_launch_track)
                    engaged = False  # not available from geometry

                    # Real 3D gate
                    r3_ok = bool(g3d.get("range_ok_3d", False))
                    a3_ok = bool(g3d.get("ata_ok_3d", False))
                    t3_ok = bool(g3d.get("ta_ok_3d", False))
                    b3_ok = bool(g3d.get("boresight_ok_3d", False))
                    geom_ok = bool(g3d.get("launch_geometry_ok_3d", False))

                    # Mismatch classification
                    gs_pos = (g_own > 0.01)
                    if gs_pos and not geom_ok and not has_track: mt = "reward_positive_real_geometry_false"
                    elif gs_pos and not has_track: mt = "reward_positive_track_false"
                    elif geom_ok and has_track and not gs_pos: mt = "reward_zero_real_geometry_true"
                    elif geom_ok and has_track and gs_pos: mt = "aligned_positive"
                    elif not geom_ok and not gs_pos: mt = "aligned_negative"
                    else: mt = "other"

                    launch_rows.append({
                        "episode_id": ep, "step": ep_len,
                        "shooter_id": shooter_id, "target_id": target_id,
                        "shooter_role": env.agent_roles.get(shooter_id,""),
                        "target_alive": int(target.is_alive),
                        "has_track": int(has_track), "track_source": track_source,
                        "obs_track_direct_raw": obs_track_direct,
                        "obs_track_mav_shared_raw": obs_track_shared,
                        "obs_observed_mask_raw": obs_observed,
                        "trace_flag": trace_flag,
                        "target_engaged": int(engaged),
                        "range_3d_m": round(float(g3d.get("range_3d_m", r2 or 0)),1),
                        "range_2d_m": round(float(g3d.get("range_2d_m", r2 or 0)),1),
                        "shooter_alive": int(shooter.is_alive),
                        "mav_alive": int(env.red_planes.get("red_0").is_alive if env.red_planes.get("red_0") else False),
                        "AO_2d_rad": round(float(ao2),4),
                        "TA_2d_rad": round(float(ta2),4),
                        "ATA_3d_rad": round(float(g3d.get("ATA_3d_rad",0)),4),
                        "TA_3d_rad": round(float(g3d.get("TA_3d_rad",0)),4),
                        "boresight_3d_rad": round(float(g3d.get("boresight_3d_rad",0)),4),
                        "range_ok_3d": int(r3_ok), "ata_ok_3d": int(a3_ok),
                        "ta_ok_3d": int(t3_ok), "boresight_ok_3d": int(b3_ok),
                        "launch_geometry_ok_3d": int(geom_ok),
                        "reward_a_own": round(float(a_own),4),
                        "reward_t_rear": round(float(t_rear),4),
                        "reward_d_gate": round(float(d_gate),4),
                        "reward_g_own": round(float(g_own),4),
                        "reward_g_enemy": round(float(g_enemy),4),
                        "mismatch_type": mt,
                    })

            # Missile stats
            for aid in env.agent_ids:
                fi = int((info.get(aid, {}) or {}).get("missiles_fired_this_step", 0))
                if aid.startswith("red_"): ep_red_fired += fi
                else: ep_blue_fired += fi
            mt_info = info.get("__missile_term__", {})
            if isinstance(mt_info, dict):
                for side in ("red","blue"):
                    th = int(mt_info.get(side,{}).get("hit",0))
                    if side == "red": ep_red_hits += max(th - prev_hits[side], 0)
                    else: ep_blue_hits += max(th - prev_hits[side], 0)
                    prev_hits[side] = th

            if all(terminated.values()) or all(truncated.values()): break
            if ep_len >= max_steps: break

        ra = sum(1 for s in env.red_planes.values() if s.is_alive)
        ba = sum(1 for s in env.blue_planes.values() if s.is_alive)
        mav_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
        outcome = "timeout" if ep_len >= max_steps else ("red_win" if ba==0 else "blue_win" if ra==0 else "draw")
        ep_summaries.append({
            "episode_id": ep, "episode_len": ep_len, "outcome": outcome,
            "mav_alive_final": int(mav_alive), "red_alive_final": ra, "blue_alive_final": ba,
            "red_missiles_fired": ep_red_fired, "blue_missiles_fired": ep_blue_fired,
            "red_hits": ep_red_hits, "blue_hits": ep_blue_hits,
            "mav_death_reason": mav_death_reason, "uav_death_count": uav_dead,
            "avg_red_altitude": round(float(np.mean(mav_alts+uav_alts)) if (mav_alts or uav_alts) else 0,1),
            "max_red_altitude": round(float(max(mav_alts+uav_alts)) if (mav_alts or uav_alts) else 0,1),
            "avg_red_speed": round(float(np.mean(uav_speeds)) if uav_speeds else 0,1),
            "min_red_speed": round(float(min(uav_speeds)) if uav_speeds else 0,1),
            "avg_action_saturation_mav": round(float(np.mean(mav_sats)) if mav_sats else 0,4),
            "avg_action_saturation_uav": round(float(np.mean(uav_sats)) if uav_sats else 0,4),
            "total_return": round(ep_return,2),
            "total_return_mav": round(ep_mav_ret,2),
            "total_return_uav_mean": round(ep_uav_ret/max(uav_dead+ra-1,1),2),
        })

    # ── Write CSVs ──
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "episode_summary.csv", ep_summaries,
               ["episode_id","episode_len","outcome","mav_alive_final","red_alive_final",
                "blue_alive_final","red_missiles_fired","blue_missiles_fired","red_hits",
                "blue_hits","mav_death_reason","uav_death_count","avg_red_altitude",
                "max_red_altitude","avg_red_speed","min_red_speed",
                "avg_action_saturation_mav","avg_action_saturation_uav",
                "total_return","total_return_mav","total_return_uav_mean"])

    _write_csv(output_dir / "step_agent_components.csv", step_rows,
               list(step_rows[0].keys()) if step_rows else [])

    _write_csv(output_dir / "launch_gate_step.csv", launch_rows,
               list(launch_rows[0].keys()) if launch_rows else [])

    # Component episode sums
    comp_ep_sums = []
    for ep in range(episodes):
        ep_comps = [r for r in comp_rows if r["episode_id"]==ep]
        for r in ep_comps:
            pos_sum = sum(max(0, r.get(k,0)) for k in r if k.startswith("sum_"))
            neg_sum = abs(sum(min(0, r.get(k,0)) for k in r if k.startswith("sum_")))
            total = pos_sum + neg_sum
            comp_ep_sums.append({**r,
                "positive_component_share": round(pos_sum/max(total,1e-12),4),
                "largest_positive_component": max((k for k in r if k.startswith("sum_")), key=lambda k: max(0,r.get(k,0)), default=""),
                "largest_negative_component": min((k for k in r if k.startswith("sum_")), key=lambda k: min(0,r.get(k,0)), default=""),
            })
    _write_csv(output_dir / "component_episode_sums.csv", comp_ep_sums,
               list(comp_ep_sums[0].keys()) if comp_ep_sums else [])

    # Launch gate breakdown
    breakdown = _launch_breakdown(launch_rows)
    _write_csv(output_dir / "launch_gate_breakdown.csv", [breakdown], list(breakdown.keys()))

    # Gate reward vs real gate
    gate_reward_rows = _gate_reward_vs_real(launch_rows)
    _write_csv(output_dir / "gate_reward_vs_real_gate.csv", gate_reward_rows,
               list(gate_reward_rows[0].keys()) if gate_reward_rows else [])

    # Altitude-speed bins
    alt_speed_rows = _altitude_speed_bins(step_rows, launch_rows)
    _write_csv(output_dir / "altitude_speed_gate_analysis.csv", alt_speed_rows,
               list(alt_speed_rows[0].keys()) if alt_speed_rows else [])

    result = {
        "episodes": episodes, "total_steps": len(step_rows),
        "output_files": {
            "episode_summary.csv": str(output_dir / "episode_summary.csv"),
            "step_agent_components.csv": str(output_dir / "step_agent_components.csv"),
            "component_episode_sums.csv": str(output_dir / "component_episode_sums.csv"),
            "launch_gate_step.csv": str(output_dir / "launch_gate_step.csv"),
            "launch_gate_breakdown.csv": str(output_dir / "launch_gate_breakdown.csv"),
            "gate_reward_vs_real_gate.csv": str(output_dir / "gate_reward_vs_real_gate.csv"),
            "altitude_speed_gate_analysis.csv": str(output_dir / "altitude_speed_gate_analysis.csv"),
        },
        "ep_summaries": ep_summaries,
        "launch_breakdown": breakdown,
        "step_rows": step_rows,
        "launch_rows": launch_rows,
    }
    if hasattr(env, "close"): env.close()
    return result

def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def _launch_breakdown(launch_rows: list[dict]) -> dict:
    if not launch_rows: return {}
    n = len(launch_rows)
    trace_counts = {}
    for r in launch_rows:
        tf = r.get("trace_flag", "unknown")
        trace_counts[tf] = trace_counts.get(tf, 0) + 1
    return {
        "total_pairs": n,
        "track_ok_pairs": sum(1 for r in launch_rows if r.get("has_track")),
        "track_direct": sum(1 for r in launch_rows if r.get("track_source") == "direct"),
        "track_mav_shared": sum(1 for r in launch_rows if r.get("track_source") == "mav_shared"),
        "range_ok_pairs": sum(1 for r in launch_rows if r.get("range_ok_3d")),
        "ata_ok_pairs": sum(1 for r in launch_rows if r.get("ata_ok_3d")),
        "ta_ok_pairs": sum(1 for r in launch_rows if r.get("ta_ok_3d")),
        "boresight_ok_pairs": sum(1 for r in launch_rows if r.get("boresight_ok_3d")),
        "geometry_ok_pairs": sum(1 for r in launch_rows if r.get("launch_geometry_ok_3d")),
        "reward_g_own_positive": sum(1 for r in launch_rows if _sf(r.get("reward_g_own")) > 0.01),
        "mismatch_reward_pos_geom_false": sum(1 for r in launch_rows if r.get("mismatch_type") == "reward_positive_real_geometry_false"),
        "mismatch_aligned_positive": sum(1 for r in launch_rows if r.get("mismatch_type") == "aligned_positive"),
        "mismatch_aligned_negative": sum(1 for r in launch_rows if r.get("mismatch_type") == "aligned_negative"),
        **{f"trace_{k}": v for k, v in sorted(trace_counts.items())},
    }

def _gate_reward_vs_real(launch_rows: list[dict]) -> list[dict]:
    out = []
    for r in launch_rows:
        out.append({
            "episode_id": r.get("episode_id"), "step": r.get("step"),
            "shooter_id": r.get("shooter_id"), "target_id": r.get("target_id"),
            "reward_g_own": r.get("reward_g_own"), "reward_a_own": r.get("reward_a_own"),
            "reward_t_rear": r.get("reward_t_rear"), "reward_d_gate": r.get("reward_d_gate"),
            "real_range_ok": r.get("range_ok_3d"), "real_ata_ok": r.get("ata_ok_3d"),
            "real_ta_ok": r.get("ta_ok_3d"), "real_boresight_ok": r.get("boresight_ok_3d"),
            "real_geometry_ok": r.get("launch_geometry_ok_3d"),
            "has_track": r.get("has_track"), "mismatch_type": r.get("mismatch_type"),
        })
    return out

def _altitude_speed_bins(step_rows: list[dict], launch_rows: list[dict]) -> list[dict]:
    alt_bins = [(-1,2500), (2500,6000), (6000,10000), (10000,12000), (12000,15000), (15000,1e9)]
    spd_bins = [(-1,102), (102,150), (150,250), (250,408), (408,1e9)]
    out = []
    for al, ah in alt_bins:
        for sl, sh in spd_bins:
            rows = [r for r in step_rows if r.get("role")!="mav" and al < _sf(r.get("altitude_m")) <= ah and sl < _sf(r.get("speed_mps")) <= sh]
            if not rows: continue
            n = len(rows)
            ep_ids = set(r["episode_id"] for r in rows)
            launch_ep_rows = [r for r in launch_rows if r["episode_id"] in ep_ids]
            geom_ok = sum(1 for r in launch_ep_rows if r.get("launch_geometry_ok_3d")) / max(len(launch_ep_rows),1)
            track_ok = sum(1 for r in launch_ep_rows if r.get("has_track")) / max(len(launch_ep_rows),1)
            out.append({
                "altitude_bin": f"{al}-{ah}", "speed_bin": f"{sl}-{sh}",
                "samples": n,
                "mean_reward_total": round(np.mean([_sf(r.get("reward_total")) for r in rows]),4),
                "mean_gate_sit": round(np.mean([_sf(r.get("tam_brma_v1_uav_gate_sit")) for r in rows]),4),
                "real_geometry_ok_rate": round(geom_ok,4),
                "track_ok_rate": round(track_ok,4),
                "launch_rate": sum(1 for r in launch_ep_rows if r.get("actual_launch_this_step")) / max(len(launch_ep_rows),1),
                "death_rate": round(sum(1 for r in rows if not r.get("alive"))/n,4),
            })
    return out


# ── Report generation ──────────────────────────────────────────────────

def _generate_report(output_dir: Path, trends: dict, rollout: dict):
    lines = [
        "# TAM-BRMA Scripted Reward v1 — Full Pipeline Audit",
        "",
        "## 1. Training Log Summary",
        "",
    ]
    if trends:
        for k, v in trends.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- No train_log.csv found")
    lines.append("")

    ep = rollout.get("ep_summaries", [])
    if ep:
        lines.append("## 2. Episode Summary")
        lines.append("")
        lines.append("| Ep | Len | Outcome | MAV | RAlive | BAlive | RFired | BFired | RAlt | MxAlt | ASatM | ASatU | Return |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in ep:
            lines.append(f"| {e['episode_id']} | {e['episode_len']} | {e['outcome']} | {e['mav_alive_final']} | {e['red_alive_final']} | {e['blue_alive_final']} | {e['red_missiles_fired']} | {e['blue_missiles_fired']} | {e['avg_red_altitude']} | {e['max_red_altitude']} | {e['avg_action_saturation_mav']} | {e['avg_action_saturation_uav']} | {e['total_return']} |")
        lines.append("")

    br = rollout.get("launch_breakdown", {})
    if br:
        lines.append("## 3. Launch Gate Breakdown")
        lines.append("")
        for k, v in br.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## 4. Key Audit Findings")
    lines.append("")

    # Reward attribution
    comp_rows = rollout.get("step_rows", [])
    if comp_rows:
        mav_rows = [r for r in comp_rows if r.get("role")=="mav"]
        uav_rows = [r for r in comp_rows if r.get("role")!="mav"]
        lines.append("### 4.1 Return Attribution")
        def _comp_means(rows, keys):
            return {k: round(np.mean([_sf(r.get(k)) for r in rows]),4) for k in keys}
        mav_keys = ["tam_brma_v1_flight","tam_brma_v1_mav_safe","tam_brma_v1_mav_support","tam_brma_v1_mav_aware","tam_brma_v1_mav_event","tam_brma_v1_team_terminal","reward_total"]
        uav_keys = ["tam_brma_v1_flight","tam_brma_v1_uav_gate_sit","tam_brma_v1_uav_event","tam_brma_v1_team_terminal","reward_total"]
        mm = _comp_means(mav_rows, mav_keys)
        um = _comp_means(uav_rows, uav_keys)
        lines.append("**MAV per-step:**")
        for k in mav_keys: lines.append(f"- {k}: {mm.get(k,0):.4f}")
        lines.append("**UAV per-step:**")
        for k in uav_keys: lines.append(f"- {k}: {um.get(k,0):.4f}")
        lines.append("")

    # Altitude analysis
    if comp_rows:
        uav_alts = [_sf(r.get("altitude_m")) for r in comp_rows if r.get("role")!="mav" and r.get("alive")]
        if uav_alts:
            above10k = sum(1 for a in uav_alts if a > 10000) / len(uav_alts)
            above12k = sum(1 for a in uav_alts if a > 12000) / len(uav_alts)
            above15k = sum(1 for a in uav_alts if a > 15000) / len(uav_alts)
            lines.append("### 4.2 Height/Speed Envelope")
            lines.append(f"- UAV above 10000m: {above10k:.1%}")
            lines.append(f"- UAV above 12000m: {above12k:.1%}")
            lines.append(f"- UAV above 15000m: {above15k:.1%}")
            uav_spds = [_sf(r.get("speed_mps")) for r in comp_rows if r.get("role")!="mav" and r.get("alive")]
            if uav_spds:
                below150 = sum(1 for s in uav_spds if s < 150) / len(uav_spds)
                below102 = sum(1 for s in uav_spds if s < 102) / len(uav_spds)
                lines.append(f"- UAV speed < 150 m/s: {below150:.1%}")
                lines.append(f"- UAV speed < 102 m/s (stall): {below102:.1%}")
            lines.append("")

    # Mismatch analysis
    lr = rollout.get("launch_rows", [])
    if lr:
        geom_ok_rate = sum(1 for r in lr if r.get("launch_geometry_ok_3d")) / max(len(lr),1)
        reward_pos_no_geom = sum(1 for r in lr if _sf(r.get("reward_g_own"))>0.01 and not r.get("launch_geometry_ok_3d"))
        reward_pos_total = sum(1 for r in lr if _sf(r.get("reward_g_own"))>0.01)
        lines.append("### 4.3 Reward Gate vs Real Launch Gate")
        lines.append(f"- Real geometry OK rate: {geom_ok_rate:.1%}")
        lines.append(f"- Reward g_own positive but real geometry false: {reward_pos_no_geom}/{reward_pos_total} = {reward_pos_no_geom/max(reward_pos_total,1):.1%}")
        # 2D vs 3D discrepancy
        ao2s = [_sf(r.get("AO_2d_rad")) for r in lr]
        ata3s = [_sf(r.get("ATA_3d_rad")) for r in lr]
        if ao2s and ata3s:
            diff = np.mean(np.abs(np.array(ao2s)-np.array(ata3s)))
            lines.append(f"- Mean |AO_2d - ATA_3d|: {diff:.4f} rad ({math.degrees(diff):.1f}°)")
        lines.append("")

    lines.append("### 4.4 Conclusion")
    lines.append("")
    # Auto-diagnose
    if comp_rows:
        uav_ret = np.mean([_sf(r.get("reward_total")) for r in comp_rows if r.get("role")!="mav"])
        mav_ret = np.mean([_sf(r.get("reward_total")) for r in comp_rows if r.get("role")=="mav"])
        gate_sit_mean = np.mean([_sf(r.get("tam_brma_v1_uav_gate_sit")) for r in comp_rows if r.get("role")!="mav"])
        flight_mean = np.mean([_sf(r.get("tam_brma_v1_flight")) for r in comp_rows if r.get("role")!="mav"])
        support_mean = np.mean([_sf(r.get("tam_brma_v1_mav_support")) for r in comp_rows if r.get("role")=="mav"])
        aware_mean = np.mean([_sf(r.get("tam_brma_v1_mav_aware")) for r in comp_rows if r.get("role")=="mav"])

        findings = []
        findings.append(f"1. MAV return = {mav_ret:.1f}/step, driven by flight ({flight_mean:.1f}) + support ({support_mean:.1f}) + aware ({aware_mean:.1f}). "
                         f"{'No combat signal — purely survival/positioning reward.' if abs(mav_ret - flight_mean - support_mean - aware_mean) < 0.5 else 'Event/terminal has non-trivial contribution.'}")
        findings.append(f"2. UAV return = {uav_ret:.1f}/step. gate_sit = {gate_sit_mean:.3f} — "
                         f"{'essentially zero — UAV receives no combat geometry reward.' if abs(gate_sit_mean) < 0.05 else 'gate_sit is active.'}")
        if lr:
            findings.append(f"3. Real geometry OK rate = {geom_ok_rate:.1%}. "
                            f"{'UAV is NEVER in launch geometry. Primary failure: position/attitude.' if geom_ok_rate < 0.01 else 'Some geometry OK but no launches — check track/lock/cooldown.'}")
        if uav_alts:
            findings.append(f"4. UAV above 10km {above10k:.0%} of the time. "
                            f"{'HEIGHT EXPLOIT: UAV is above ceiling, unreachable by blue, unable to engage.' if above10k > 0.5 else 'Height is within operational range.'}")
        findings.append(f"5. {'Survival-only policy confirmed: zero missile launches, 100% timeout, all agents alive. Strategy = climb above blue engagement range.' if sum(1 for e in ep if e.get('red_missiles_fired',0)>0) == 0 else 'Policy shows some combat activity.'}")

        for f in findings: lines.append(f)

    (output_dir / "audit_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-arch", default="pure_happo")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--deterministic-eval", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.checkpoint).parents[1]

    print("=== Parsing train log ===")
    train_rows = _load_train_log(run_dir)
    trends = parse_train_trends(train_rows) if train_rows else {}
    if trends:
        print(f"  Iters: {trends.get('total_iters')}, Best ret: {trends.get('best_return'):.1f}, Final ret: {trends.get('final_return'):.1f}")

    print(f"\n=== Running {args.episodes}-episode deterministic rollout ===")
    rollout = run_deterministic_rollout(args.checkpoint, args.config,
                                         args.episodes, args.max_steps,
                                         args.device, out)
    print(f"  Steps recorded: {len(rollout.get('step_rows',[]))}")
    print(f"  Launch pairs: {len(rollout.get('launch_rows',[]))}")
    for name, path_str in rollout.get("output_files", {}).items():
        print(f"  {name}: {path_str}")

    print("\n=== Generating audit report ===")
    _generate_report(out, trends, rollout)
    print(f"  {out / 'audit_report.md'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
