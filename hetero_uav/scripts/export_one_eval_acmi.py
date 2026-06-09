"""Export one deterministic eval episode as Tacview ACMI. No training."""
from __future__ import annotations

import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from uav_env import make_env
from algorithms.mappo.adapter_utils import (
    load_model_meta, make_obs_adapter, resolve_obs_adapter_version,
    validate_model_dims, make_mappo_model_for_adapter,
)
from algorithms.mappo.opponent_policy import OpponentPolicy

DEFAULT_MODEL = 'outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt'
DEFAULT_CONFIG = 'uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml'
DEFAULT_OUTPUT = 'outputs/acmi/alive_done_fix_3v2_episode0.acmi'
DEFAULT_SUMMARY = 'outputs/acmi/alive_done_fix_3v2_episode0_summary.json'
REF_LAT, REF_LON = 30.0, 120.0
METERS_PER_DEG = 111111.0

def _lon_off(e): return e / (METERS_PER_DEG * math.cos(math.radians(REF_LAT)))
def _lat_off(n): return n / METERS_PER_DEG
def _obj_id(aid):
    parts = aid.split('_')
    return (100 + int(parts[1])) if parts[0] == 'red' else (200 + int(parts[1]))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--output-acmi', default=DEFAULT_OUTPUT)
    p.add_argument('--output-summary', default=DEFAULT_SUMMARY)
    p.add_argument('--opponent-policy', default='rule_nearest')
    p.add_argument('--obs-adapter-version', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--deterministic', action='store_true', default=True)
    args = p.parse_args()

    meta = load_model_meta(args.model)
    version = resolve_obs_adapter_version(args.obs_adapter_version, meta)
    adapter = make_obs_adapter(version)
    validate_model_dims(adapter, meta)
    actor_arch = meta.get('actor_arch', 'mlp')
    device = torch.device('cpu')
    model = make_mappo_model_for_adapter(adapter, device, actor_arch=actor_arch)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    env = make_env(args.config, env_type='jsbsim_hetero')
    obs, info = env.reset(seed=args.seed)
    dt = float(getattr(env, 'env_dt', 0.2))

    meta_air = {}
    for aid in env.red_ids + env.blue_ids:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        mname = getattr(sim, 'model', 'unknown')
        color = 'Red' if aid.startswith('red') else 'Blue'
        coalition = 'Allies' if aid.startswith('red') else 'Enemies'
        meta_air[aid] = dict(obj_id=_obj_id(aid), name=f'{aid}_{mname}', color=color, coalition=coalition)

    lines = ['FileType=text/acmi/tacview', 'FileVersion=2.2',
             '0,ReferenceTime=2026-01-01T00:00:00Z', '0,Title=hetero_uav_alive_done_fix_3v2',
             '0,DataSource=hetero_uav_jsbsim']
    recorded, prev_alive = set(), {}
    frame, ep_ret, done = 0, 0.0, False

    while not done:
        t = frame * dt; lines.append(f'#{t:.1f}')
        result = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [result['actor_obs'].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32)) for rid in env.red_ids]
        aobs_np = np.stack(aobs)
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(aobs_np), torch.as_tensor(result['critic_state']).unsqueeze(0), deterministic=args.deterministic)
        acts = {rid: action.cpu().numpy()[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
        opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + frame)
        acts.update(opponent.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)
        done = all(terminated.values()) or all(truncated.values())
        for rid in env.red_ids: ep_ret += float(rewards.get(rid, 0.0))

        for aid in env.red_ids + env.blue_ids:
            sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
            if sim is None: continue
            oid = meta_air[aid]['obj_id']; pos = sim.get_position()
            lon = REF_LON + _lon_off(pos[1]); lat = REF_LAT + _lat_off(pos[0])
            alt = float(pos[2]); r, p, y = sim.get_rpy()
            rd, pd, yd = math.degrees(r), math.degrees(p), math.degrees(y)
            if oid not in recorded:
                lines.append(f'{oid},T={lon}|{lat}|{alt}|{rd}|{pd}|{yd},Type=Air+FixedWing,Name={meta_air[aid]["name"]},Coalition={meta_air[aid]["coalition"]},Color={meta_air[aid]["color"]}')
                recorded.add(oid)
            else:
                if prev_alive.get(oid, True) and not sim.is_alive:
                    lines.append(f'0,Event=Destroyed|{oid}|{aid} destroyed')
                lines.append(f'{oid},T={lon}|{lat}|{alt}|{rd}|{pd}|{yd}')
            prev_alive[oid] = sim.is_alive
        frame += 1
        if frame > 5000: break

    ra = sum(1 for s in env.red_planes.values() if s.is_alive)
    ba = sum(1 for s in env.blue_planes.values() if s.is_alive)
    ma = bool(env.red_planes.get('red_0') and env.red_planes['red_0'].is_alive)
    lines.append(f'0,Message=episode_end|red_alive={ra}|blue_alive={ba}|mav_alive={ma}|total_ret={ep_ret:.1f}')

    oa, os_ = Path(args.output_acmi), Path(args.output_summary)
    oa.parent.mkdir(parents=True, exist_ok=True); os_.parent.mkdir(parents=True, exist_ok=True)
    oa.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    os_.write_text(json.dumps(dict(model=args.model, config=args.config, seed=args.seed, frames=frame, decision_dt=dt, simulated_time_s=frame*dt, red_alive_final=ra, blue_alive_final=ba, mav_alive=ma, total_return=float(ep_ret), output_acmi=str(oa)), indent=2))

    print(f'output_acmi: {oa}'); print(f'output_summary: {os_}'); print(f'frames: {frame} sim_t: {frame*dt:.0f}s red_alive: {ra} blue_alive: {ba} mav_alive: {ma}')
if __name__ == '__main__': main()
