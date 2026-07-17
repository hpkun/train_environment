"""Pre-training 20-episode nominal missile/rule gate."""
from __future__ import annotations
import csv,json,sys
from datetime import datetime
from pathlib import Path
if __package__ in (None,""): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.evaluation import evaluate_policy

def run_baselines(episodes=20,seed=1):
    rows=[evaluate_policy(p,episodes,"paper_nominal_1v1","paper_greedy",seed)
          for p in ("zero_no_fire","random","pursuit_fire_rule")]
    lookup={x["policy"]:x for x in rows}; zero=lookup["zero_no_fire"]; random=lookup["random"]; pursuit=lookup["pursuit_fire_rule"]
    passed=bool(zero["red_hits"]==0 and random["red_hits"]<16 and pursuit["red_hits"]>=16
        and all(x["numerical_invalid"]==0 and x["opponent_failures"]==0 for x in rows))
    return {"passed":passed,"gate":{"zero_no_fire_red_kills":0,"random_not_stable_red_kills_lt":16,
        "pursuit_fire_rule_red_kills_at_least":16,"invalid":0,"opponent_failure":0},"rows":rows}

def main():
    result=run_baselines(); out=Path("aircombat_env_v1/outputs")/("missile_rule_gate_"+datetime.now().strftime("%Y%m%d_%H%M%S")); out.mkdir(parents=True,exist_ok=True)
    (out/"baselines.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (out/"baselines.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(result["rows"][0])); w.writeheader(); w.writerows(result["rows"])
    print(json.dumps({**result,"output_dir":str(out.resolve())},indent=2)); raise SystemExit(0 if result["passed"] else 1)
if __name__=="__main__": main()
