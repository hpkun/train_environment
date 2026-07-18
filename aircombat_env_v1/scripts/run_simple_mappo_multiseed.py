"""Run and summarize the fixed three-seed SimpleTAMCombatEnv MAPPO experiment."""
from __future__ import annotations
import argparse,csv,json,math,statistics,subprocess,sys,time
from pathlib import Path

if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model,load_checkpoint,resolve_device

SCENARIO="simple_paper_1v1"
SEEDS=(1,2,3)
CHECKPOINTS=("initial","best","latest")
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT=ROOT/"aircombat_env_v1"/"outputs"/"simple_mappo_1v1_multiseed_100k"

def read_csv(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def number(row,key,default=0.):
    try:return float(row[key])
    except (KeyError,TypeError,ValueError):return default

def value_range(rows,key):
    values=[number(row,key,float("nan")) for row in rows]
    values=[value for value in values if math.isfinite(value)]
    return [min(values),max(values)] if values else [None,None]

def logs_are_finite(*logs):
    for rows in logs:
        for row in rows:
            for value in row.values():
                try:
                    if not math.isfinite(float(value)):return False
                except (TypeError,ValueError):pass
    return True

def evaluate_checkpoints(seed_dir,device):
    results={}
    for name in CHECKPOINTS:
        model=load_checkpoint(seed_dir/f"{name}.pt",SCENARIO,device)
        deterministic=evaluate_model(model,SCENARIO,1,device,True,10001,True)
        deterministic["winner"]=deterministic.pop("rows")[0]["winner"]
        stochastic=evaluate_model(model,SCENARIO,20,device,False,20001)
        results[name]={"deterministic":deterministic,"stochastic":stochastic}
    return results

def checkpoint_step(path):
    import torch
    return int(torch.load(path,map_location="cpu",weights_only=False)["env_steps"])

def collect_seed(seed,seed_dir,training_seconds,evaluations,training_ok=True):
    train_rows=read_csv(seed_dir/"train_log.csv");eval_rows=read_csv(seed_dir/"eval_log.csv")
    last_train=train_rows[-1] if train_rows else {};initial_eval=eval_rows[0] if eval_rows else {};last_eval=eval_rows[-1] if eval_rows else {}
    max_eval=max(eval_rows,key=lambda row:number(row,"mean_return",-float("inf"))) if eval_rows else {}
    finite=logs_are_finite(train_rows,eval_rows)
    result={
      "seed":seed,"completed":bool(training_ok and train_rows and number(last_train,"env_steps")>=100000),"finite":finite,
      "training_seconds":training_seconds,"env_steps":int(number(last_train,"env_steps")),"updates":len(train_rows),
      "episodes_completed":int(number(last_train,"episodes_completed")),
      "eval_log_initial_return":number(initial_eval,"mean_return"),"eval_log_max_return":number(max_eval,"mean_return"),
      "eval_log_max_step":int(number(max_eval,"env_steps")),"eval_log_last_return":number(last_eval,"mean_return"),
      "recent20_train_mean_return":number(last_train,"mean_episode_return"),"recent20_train_red_win_rate":number(last_train,"red_win_rate"),
      "actor_loss_range":value_range(train_rows,"actor_loss"),"critic_loss_range":value_range(train_rows,"critic_loss"),
      "entropy_range":value_range(train_rows,"entropy"),"approx_kl_range":value_range(train_rows,"approx_kl"),
      "clip_fraction_range":value_range(train_rows,"clip_fraction"),"action_mean_abs_range":value_range(train_rows,"action_mean_abs"),
      "action_saturation_rate_range":value_range(train_rows,"action_saturation_rate"),
      "best_checkpoint_env_steps":checkpoint_step(seed_dir/"best.pt"),"evaluations":evaluations,
    }
    for name in CHECKPOINTS:
        det=evaluations[name]["deterministic"];sto=evaluations[name]["stochastic"]
        result[f"{name}_deterministic_return"]=det["mean_return"]
        result[f"{name}_deterministic_red_win"]=det["red_win_rate"]
        result[f"{name}_stochastic_mean_return"]=sto["mean_return"]
        result[f"{name}_stochastic_return_std"]=sto["return_std"]
        result[f"{name}_stochastic_red_win_rate"]=sto["red_win_rate"]
    result["best_improvement"]=result["best_deterministic_return"]-result["initial_deterministic_return"]
    result["latest_improvement"]=result["latest_deterministic_return"]-result["initial_deterministic_return"]
    summaries=[evaluation[mode] for evaluation in evaluations.values() for mode in ("deterministic","stochastic")]
    result["numerical_invalid_episodes"]=int(sum(x["numerical_invalid_episodes"] for x in summaries)+number(last_train,"numerical_invalid_episodes"))
    result["crashes"]=int(sum(x["crashes"] for x in summaries));result["boundary_deaths"]=int(sum(x["boundary_deaths"] for x in summaries))
    result["flight_envelope_violation_episodes"]=int(sum(x["flight_envelope_violation_episodes"] for x in summaries))
    return result

def mean_std(values):
    values=[float(value) for value in values]
    return {"mean":statistics.fmean(values),"std":statistics.pstdev(values)} if values else {"mean":None,"std":None}

def classify(rows):
    complete=[row for row in rows if row.get("completed")]
    stable=len(complete)==3 and all(row.get("finite") and row.get("numerical_invalid_episodes")==0 for row in complete)
    best=[row["best_improvement"] for row in complete];latest=[row["latest_improvement"] for row in complete]
    learning=len(best)==3 and sum(value>0 for value in best)>=2 and statistics.fmean(best)>0
    retention=len(latest)==3 and sum(value>0 for value in latest)>=2 and statistics.median(latest)>0
    return {"A_stable":stable,"B_learning_signal":learning,"C_retention":retention,
      "best_improved_seed_count":sum(value>0 for value in best),"latest_improved_seed_count":sum(value>0 for value in latest)}

def aggregate(rows):
    complete=[row for row in rows if row.get("completed")]
    keys=("initial_deterministic_return","best_deterministic_return","latest_deterministic_return","best_improvement","latest_improvement",
          "initial_stochastic_mean_return","best_stochastic_mean_return","latest_stochastic_mean_return",
          "initial_stochastic_red_win_rate","best_stochastic_red_win_rate","latest_stochastic_red_win_rate")
    result={key:mean_std([row[key] for row in complete]) for key in keys}
    result.update({"completed_seed_count":len(complete),"total_training_seconds":sum(row.get("training_seconds",0) for row in complete),
      "total_numerical_invalid_episodes":sum(row.get("numerical_invalid_episodes",0) for row in complete),
      "total_crashes":sum(row.get("crashes",0) for row in complete),"total_boundary_deaths":sum(row.get("boundary_deaths",0) for row in complete),
      "total_flight_envelope_violation_episodes":sum(row.get("flight_envelope_violation_episodes",0) for row in complete)})
    return result

def add_required_aliases(row):
    aliases={"best_return_improvement":"best_improvement","latest_return_improvement":"latest_improvement",
      "initial_stochastic_return_mean":"initial_stochastic_mean_return","best_stochastic_return_mean":"best_stochastic_mean_return",
      "latest_stochastic_return_mean":"latest_stochastic_mean_return","crash_count":"crashes","boundary_death_count":"boundary_deaths",
      "envelope_violation_count":"flight_envelope_violation_episodes"}
    for target,source in aliases.items():
        if source in row:row[target]=row[source]
    return row

def write_summary(output,rows):
    output.mkdir(parents=True,exist_ok=True);rows=[add_required_aliases(dict(row)) for row in rows]
    summary={"scenario":SCENARIO,"expected_seeds":list(SEEDS),"seeds":rows,"aggregate":aggregate(rows),"criteria":classify(rows)}
    (output/"multiseed_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    flat_keys=("seed","completed","finite","training_seconds","env_steps","updates","episodes_completed","initial_deterministic_return","best_deterministic_return","latest_deterministic_return","best_checkpoint_env_steps","best_return_improvement","latest_return_improvement","initial_stochastic_return_mean","best_stochastic_return_mean","latest_stochastic_return_mean","initial_stochastic_return_std","best_stochastic_return_std","latest_stochastic_return_std","initial_stochastic_red_win_rate","best_stochastic_red_win_rate","latest_stochastic_red_win_rate","numerical_invalid_episodes","crash_count","boundary_death_count","envelope_violation_count")
    with (output/"multiseed_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=flat_keys);writer.writeheader();writer.writerows([{key:row.get(key) for key in flat_keys} for row in rows])
    lines=["# Simple MAPPO 1v1 multi-seed report","", "| Seed | Complete | Initial det. | Best det. | Latest det. | Best step | Best gain | Latest gain |", "|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:lines.append(f"| {row['seed']} | {row.get('completed',False)} | {row.get('initial_deterministic_return','')} | {row.get('best_deterministic_return','')} | {row.get('latest_deterministic_return','')} | {row.get('best_checkpoint_env_steps','')} | {row.get('best_improvement','')} | {row.get('latest_improvement','')} |")
    lines.extend(["","## Aggregate","",f"- A stable: {summary['criteria']['A_stable']}",f"- B learning signal: {summary['criteria']['B_learning_signal']}",f"- C retention: {summary['criteria']['C_retention']}","", "```json",json.dumps(summary["aggregate"],indent=2),"```",""])
    (output/"multiseed_report.md").write_text("\n".join(lines),encoding="utf-8")
    return summary

def summarize_root(output,expected_seeds=SEEDS):
    rows=[]
    for seed in expected_seeds:
        path=Path(output)/f"seed_{seed}"/"seed_summary.json"
        if path.exists():rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:rows.append({"seed":seed,"completed":False,"finite":False,"failure":"missing seed_summary.json"})
    return write_summary(Path(output),rows)

def train_seed(seed,seed_dir):
    command=[sys.executable,"-u",str(ROOT/"aircombat_env_v1"/"scripts"/"train_simple_mappo.py"),"--scenario",SCENARIO,"--total-env-steps","100000","--rollout-length","256","--seed",str(seed),"--device","auto","--output-dir",str(seed_dir),"--eval-interval","10000","--eval-episodes","1","--deterministic-eval","--actor-lr","0.0003","--critic-lr","0.0003","--entropy-coef","0.01","--ppo-epochs","4"]
    started=time.perf_counter()
    with (seed_dir/"training_stdout.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(f"[seed {seed}] {line}",end="",flush=True);log.write(line);log.flush()
        return_code=process.wait()
    return return_code,time.perf_counter()-started

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    output=args.output_root.resolve();output.mkdir(parents=True,exist_ok=True);device=resolve_device("auto")
    for seed in SEEDS:
        seed_dir=output/f"seed_{seed}";seed_dir.mkdir(parents=True,exist_ok=True)
        try:
            return_code,seconds=train_seed(seed,seed_dir)
            if return_code:raise RuntimeError(f"training exited with code {return_code}")
            evaluations=evaluate_checkpoints(seed_dir,device);row=collect_seed(seed,seed_dir,seconds,evaluations)
        except Exception as exc:
            row={"seed":seed,"completed":False,"finite":False,"failure":f"{type(exc).__name__}: {exc}"}
            print(f"[seed {seed}] FAILED: {row['failure']}",flush=True)
        (seed_dir/"seed_summary.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    summary=summarize_root(output);print(json.dumps(summary["criteria"],indent=2),flush=True)

if __name__=="__main__":main()
