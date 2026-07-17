"""Evaluate a recurrent checkpoint on nominal or zero-shot perturbations."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
if __package__ in (None,""): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.evaluation import evaluate_policy
from aircombat_env_v1.recurrent_ppo import RecurrentActor
from aircombat_env_v1.seeds import *
from aircombat_env_v1.training import write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--level",choices=("nominal","low","medium","high"),required=True); p.add_argument("--output"); p.add_argument("--device",default="cpu"); a=p.parse_args()
 actor=RecurrentActor().to(a.device); payload=torch.load(a.checkpoint,map_location=a.device,weights_only=False); actor.load_state_dict(payload["actor"]); actor.eval()
 mapping={"nominal":("paper_nominal_1v1",NOMINAL_VALIDATION_SEEDS),"low":("generalization_low",GENERALIZATION_LOW_SEEDS),"medium":("generalization_medium",GENERALIZATION_MEDIUM_SEEDS),"high":("generalization_high",GENERALIZATION_HIGH_SEEDS)}
 scenario,seeds=mapping[a.level]; result=evaluate_policy("recurrent_ppo",scenario=scenario,seeds=seeds,actor=actor,device=a.device)
 if a.output: write_json(a.output,result)
 print(json.dumps(result,indent=2))
if __name__=="__main__": main()
