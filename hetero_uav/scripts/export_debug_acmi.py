"""Debug ACMI export with extra diagnostics. No training."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from uav_env import make_env
from algorithms.mappo.adapter_utils import (load_model_meta, make_obs_adapter, resolve_obs_adapter_version, validate_model_dims, make_mappo_model_for_adapter)
from algorithms.mappo.opponent_policy import OpponentPolicy

MODEL = "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"

def rd(r): return [math.degrees(float(x)) for x in r]

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--output-acmi", default="outputs/acmi/debug_alive_done_fix_3v2_episode0.acmi")
    p.add_argument("--output-summary", default="outputs/acmi/debug_alive_done_fix_3v2_episode0_summary.json")
    args = p.parse_args()

    meta = load_model_meta(args.model)
    version = resolve_obs_adapter_version(None, meta)
    adapter = make_obs_adapter(version)
    validate_model_dims(adapter, meta)
    arch = meta.get("actor_arch", "mlp")
    device = torch.device("cpu")
    model = make_mappo_model_for_adapter(adapter, device, actor_arch=arch)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    env = make_env(args.config, env_type="jsbsim_hetero")
    obs, info = env.reset(seed=0)
    dt = float(getattr(env, "env_dt", 0.2))

    def oid(aid):
        parts = aid.split("_")
        return (100 + int(parts[1])) if parts[0] == "red" else (200 + int(parts[1]))

    REF_LAT, REF_LON = 30.0, 120.0
    DPM = 111111.0

    def to_ll(north, east):
        lat = REF_LAT + north / DPM
        lon = REF_LON + east / (DPM * math.cos(math.radians(REF_LAT)))
        return lon, lat

    meta_air = {}
    for aid in env.red_ids + env.blue_ids:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        mname = getattr(sim, "model", "unknown")
        color = "Red" if aid.startswith("red") else "Blue"
        coal = "Allies" if aid.startswith("red") else "Enemies"
        meta_air[aid] = dict(obj_id=oid(aid), name=f"{aid}_{mname}", color=color, coalition=coal)

    lines = ["FileType=text/acmi/tacview", "FileVersion=2.2",
             "0,ReferenceTime=2026-01-01T00:00:00Z", "0,Title=hetero_uav_debug",
             "0,DataSource=hetero_uav_jsbsim"]
    recorded, prev_alive = set(), {}
    dead_removed = set()  # aircraft that died and were removed
    death_info = {}       # per-aircraft death step/time/position
    frame, ep_ret, done = 0, 0.0, False
    debug_red0 = []

    while not done:
        t = frame * dt; lines.append(f"#{t:.1f}")
        result = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [result["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, np.float32)) for rid in env.red_ids]
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(np.stack(aobs)), torch.as_tensor(result["critic_state"]).unsqueeze(0), deterministic=True)
        acts = {rid: action.cpu().numpy()[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
        opp = OpponentPolicy(mode="rule_nearest", seed=frame)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)
        done = all(terminated.values()) or all(truncated.values())
        for rid in env.red_ids: ep_ret += float(rewards.get(rid, 0.0))

        for aid in env.red_ids + env.blue_ids:
            sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
            if sim is None: continue
            oid_ = meta_air[aid]["obj_id"]
            alive_now = sim.is_alive

            # First appearance
            if oid_ not in recorded:
                pos = sim.get_position()
                lon, lat = to_ll(pos[0], pos[1]); alt = float(pos[2])
                r, p, y = sim.get_rpy()
                lines.append(f"{oid_},T={lon}|{lat}|{alt}|{math.degrees(r)}|{math.degrees(p)}|{math.degrees(y)},Type=Air+FixedWing,Name={meta_air[aid]['name']},Coalition={meta_air[aid]['coalition']},Color={meta_air[aid]['color']}")
                recorded.add(oid_)
                prev_alive[oid_] = alive_now
                continue

            # Death detection
            if prev_alive.get(oid_, True) and not alive_now and oid_ not in dead_removed:
                pos = sim.get_position()
                lines.append(f"0,Event=Destroyed|{oid_}|{aid} destroyed")
                death_info[aid] = dict(step=frame, time_s=frame * dt, lon=to_ll(pos[0], pos[1])[0], lat=to_ll(pos[0], pos[1])[1], altitude_m=float(pos[2]))
                dead_removed.add(oid_)

            # Only write T= if still alive
            if alive_now:
                pos = sim.get_position()
                lon, lat = to_ll(pos[0], pos[1]); alt = float(pos[2])
                r, p, y = sim.get_rpy()
                lines.append(f"{oid_},T={lon}|{lat}|{alt}|{math.degrees(r)}|{math.degrees(p)}|{math.degrees(y)}")
            # Dead aircraft: no T= line written (remains at last position in Tacview)

            prev_alive[oid_] = alive_now

        if frame < 100:
            sim = env.red_planes.get("red_0")
            rpy = rd(sim.get_rpy()) if sim and sim.is_alive else [0,0,0]
            pos = sim.get_position() if sim else [0,0,0]
            red0_raw = list(acts.get("red_0", [0,0,0]).tolist() if hasattr(acts.get("red_0"), "tolist") else acts.get("red_0", [0,0,0]))
            red0_trim = env._last_action_trim_applied.get("red_0", None)
            red0_eff = env._last_effective_actions.get("red_0", None)
            mw = info.get("red_0", {}).get("missile_warning", None) if isinstance(info, dict) else None
            if hasattr(mw, "tolist"): mw = mw.tolist()
            blu_info = {}
            for bid in env.blue_ids:
                bsim = env.blue_planes.get(bid)
                nr = -1.0
                if bsim and bsim.is_alive:
                    for rid in env.red_ids:
                        rs = env.red_planes.get(rid)
                        if rs and rs.is_alive:
                            d = float(np.linalg.norm(bsim.get_position() - rs.get_position()))
                            nr = d if nr < 0 else min(nr, d)
                blu_info[bid] = round(nr, 1) if nr > 0 else -1
            debug_red0.append(dict(
                step=frame, raw_action=red0_raw, action_trim=red0_trim, effective_action=red0_eff,
                roll_deg=rpy[0], pitch_deg=rpy[1], yaw_deg=rpy[2],
                altitude=float(pos[2]) if sim else 0, speed=float(np.linalg.norm(sim.get_velocity())) if sim and sim.is_alive else 0,
                alive=bool(sim and sim.is_alive),
                missile_warning=mw, nearest_blue_distance=blu_info))

        frame += 1
        if frame > 5000: break

    ra = sum(1 for s in env.red_planes.values() if s.is_alive)
    ba = sum(1 for s in env.blue_planes.values() if s.is_alive)
    ma = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)

    lines.append(f"0,Message=episode_end|red_alive={ra}|blue_alive={ba}|mav_alive={ma}|total_ret={ep_ret:.1f}")

    oa = Path(args.output_acmi); os_ = Path(args.output_summary)
    oa.parent.mkdir(parents=True, exist_ok=True); os_.parent.mkdir(parents=True, exist_ok=True)
    oa.write_text("\n".join(lines) + "\n", encoding="utf-8")
    red0_death = death_info.get("red_0", None)
    os_.write_text(json.dumps(dict(
        model=args.model, config=args.config, frames=frame, decision_dt=dt,
        simulated_time_s=frame*dt, red_alive_final=ra, blue_alive_final=ba, mav_alive=ma,
        total_return=float(ep_ret),
        red0_death_step=red0_death["step"] if red0_death else None,
        red0_death_altitude=red0_death["altitude_m"] if red0_death else None,
        red0_stopped_logging_after_death=True if red0_death else False,
        red0_first100=debug_red0,
        all_death_info=death_info,
        output_acmi=str(oa)), indent=2))
    print(f"output_acmi: {oa}"); print(f"output_summary: {os_}")
    print(f"frames: {frame} sim_t: {frame*dt:.0f}s red_alive: {ra} blue_alive: {ba} mav_alive: {ma}")

if __name__ == "__main__": main()
