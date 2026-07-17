"""Finite 100-step interface check for both formal paper scenarios."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_env import TAMPaperCombatEnv

def main():
    results={}
    for mode in ("paper_nominal_1v1","paper_nominal_2v2"):
        env=TAMPaperCombatEnv(mode,"all",max_steps=100);obs,info=env.reset(seed=1);steps=0
        shapes={aid:{k:list(v.shape) for k,v in item.items()} for aid,item in obs.items()}
        for _ in range(100):
            actions={aid:env.action_space[aid].sample() for aid in env.controlled_ids}
            obs,rewards,terminated,truncated,info=env.step(actions);steps+=1
            assert all(np.isfinite(v).all() for item in obs.values() for v in item.values())
            assert all(np.isfinite(v) for v in rewards.values())
            if terminated or truncated:break
        results[mode]={"steps":steps,"action_nvec":[40,40,40,40],"observation_shapes":shapes,
                       "flattened_dimensions":{aid:int(env.flatten_observation(item).size) for aid,item in obs.items()},
                       "invalid":info["invalid_episode"]}
        env.close()
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
