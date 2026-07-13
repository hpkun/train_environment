from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path: __import__('sys').path.insert(0,str(ROOT))
from scripts.eval_policy_launch_diagnostics import _build_policy,_load_meta,_policy_actions
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.envs.paper_calibrated_v4 import V4_COMPONENT_FIELDS

def collect(config,label,episodes,max_steps,seed,checkpoint=None,device='cpu'):
    policy=None; adapter=HeteroObsAdapterV2()
    if checkpoint:
        cp=Path(checkpoint); meta=_load_meta(cp); policy=_build_policy(meta,torch.device(device)); policy.load(cp,map_location=device); policy.eval()
    rows=[]
    for ep in range(episodes):
        env=make_env(config,suppress_jsbsim_output=True); opp=OpponentPolicy(mode='tam_greedy_rule',seed=seed+ep+91)
        obs,info=env.reset(seed=seed+ep); sums={k:0.0 for k in V4_COMPONENT_FIELDS}; launches=hits=0
        try:
            for step in range(max_steps):
                if policy is None: red={r:env.action_space[r].sample() for r in env.red_ids}
                else:
                    acts,_=_policy_actions(policy,adapter,env,obs,info,torch.device(device)); red={r:acts[i] for i,r in enumerate(env.red_ids)}
                actions={**red,**opp.act(obs,env.blue_ids,env=env)}; obs,reward,term,trunc,info=env.step(actions)
                for c in info.get('reward_components',{}).values():
                    for k in V4_COMPONENT_FIELDS: sums[k]+=float(c.get(k,0.0) or 0.0)
                launches+=sum(int(info.get(r,{}).get('missiles_fired_this_step',0)) for r in env.red_ids)
                hits+=sum(1 for x in info.get('__launch_quality_done__',[]) or [] if str(x.get('shooter_id','')).startswith('red') and x.get('raw_termination_reason')=='hit')
                if all(term.values()) or all(trunc.values()): break
            ra=sum(env.red_planes[r].is_alive for r in env.red_ids); ba=sum(env.blue_planes[b].is_alive for b in env.blue_ids); mav=int(env.red_planes[env.red_ids[0]].is_alive)
            rows.append({'source':label,'episode':ep,'episode_length':step+1,'red_alive':ra,'blue_alive':ba,'mav_alive':mav,
                         'red_win':int(ba==0 and ra>0),'timeout':int(step+1>=max_steps),'red_launch':launches,'red_hit':hits,**sums})
        finally: env.close()
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',action='append',default=[])
    ap.add_argument('--episodes',type=int,default=3); ap.add_argument('--max-steps',type=int,default=200); ap.add_argument('--seed',type=int,default=700); ap.add_argument('--device',default='cpu'); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=collect(a.config,'random',a.episodes,a.max_steps,a.seed,device=a.device)
    for cp in a.checkpoint: rows+=collect(a.config,Path(cp).parents[1].name,a.episodes,a.max_steps,a.seed,cp,a.device)
    df=pd.DataFrame(rows); raw=[k for k in V4_COMPONENT_FIELDS if k.endswith('_raw')]
    records=[]
    for source,g in df.groupby('source'):
        for k in raw:
            x=g[k].astype(float); records.append({'source':source,'component':k,'mean':x.mean(),'std':x.std(ddof=0),'p10':x.quantile(.1),'p25':x.quantile(.25),'p50':x.quantile(.5),'p75':x.quantile(.75),'p90':x.quantile(.9),'p95':x.quantile(.95),'abs_mean':x.abs().mean(),'nonzero_ratio':(x.abs()>1e-12).mean(),'saturation_ratio':(x.abs()>=.999).mean()})
    pd.DataFrame(records).to_csv(out/'v4_raw_component_distribution.csv',index=False)
    df.groupby(['source','red_win','timeout'],dropna=False)[raw+['red_alive','blue_alive','mav_alive','red_launch','red_hit']].mean().reset_index().to_csv(out/'v4_outcome_conditioned_components.csv',index=False)
    magnitude=lambda k:max(float(df[k].abs().mean()),1e-6)
    base={'uav_dense_scale':.375/magnitude('v4_uav_dense_raw'),'uav_event_scale':1.0,'terminal_scale':1.0,'flight_scale':.02/magnitude('v4_flight_raw')}
    candidates=[]
    for name,weights in [('safety_dominant',(1.0,.5,.75)),('balanced',(.75,.75,1.0)),('event_dominant',(.5,.5,1.5))]:
        c={**base,'candidate':name,'mav_safety_scale':weights[0]/magnitude('v4_mav_safety_raw'),'mav_support_scale':weights[1]/magnitude('v4_mav_support_raw'),'mav_event_scale':weights[2],'mav_death_event_scale':1.0,'mav_team_kill_credit_scale':.5,'mav_team_kill_credit_cap':1.0}; candidates.append(c)
    pd.DataFrame(candidates).to_csv(out/'v4_scale_candidates.csv',index=False)
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    for c in candidates:
        x=json.loads(json.dumps(cfg)); x['brma_tam_paper_calibrated_v4']['calibration_status']='data_calibrated_candidate'; x['brma_tam_paper_calibrated_v4']['scales'].update({k:v for k,v in c.items() if k!='candidate'}); (out/f"v4_{c['candidate']}.yaml").write_text(yaml.safe_dump(x,sort_keys=False),encoding='utf-8')
    (out/'v4_calibration_report.md').write_text('# V4 calibration report\n\nData are short deterministic/random rollouts, not performance claims. Top-level scales are empirical and are not paper coefficients.\n\n'+pd.DataFrame(candidates).to_markdown(index=False),encoding='utf-8')
    df.to_csv(out/'v4_episode_raw_components.csv',index=False); print(out)
if __name__=='__main__': main()
