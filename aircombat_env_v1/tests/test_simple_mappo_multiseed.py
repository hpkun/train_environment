import json,subprocess,sys
from pathlib import Path
import pytest
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model
from aircombat_env_v1.scripts.run_simple_mappo_multiseed import classify,mean_std,summarize_root
from aircombat_env_v1.simple_mappo import SharedMAPPOActorCritic
import torch

def evaluate(model,deterministic,seed):
    return evaluate_model(model,"simple_paper_1v1",1,torch.device("cpu"),deterministic,seed,True)

def test_stochastic_evaluation_repeats_with_same_base_seed():
    model=SharedMAPPOActorCritic(61,61)
    first=evaluate(model,False,20001);second=evaluate(model,False,20001)
    assert first["rows"][0]["action_trace"]==second["rows"][0]["action_trace"] and first["mean_return"]==second["mean_return"]

def test_different_stochastic_seeds_produce_different_action_trajectories():
    model=SharedMAPPOActorCritic(61,61)
    assert evaluate(model,False,20001)["rows"][0]["action_trace"]!=evaluate(model,False,20002)["rows"][0]["action_trace"]

def test_deterministic_evaluation_is_seed_independent():
    model=SharedMAPPOActorCritic(61,61)
    first=evaluate(model,True,10001);second=evaluate(model,True,10099)
    assert first["rows"][0]["action_trace"]==second["rows"][0]["action_trace"] and first["mean_return"]==second["mean_return"]

def test_training_saves_initial_checkpoint_at_zero_steps(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts"/"train_simple_mappo.py";output=tmp_path/"train"
    subprocess.run([sys.executable,str(script),"--scenario","simple_paper_1v1","--total-env-steps","0","--eval-episodes","1","--output-dir",str(output)],check=True,capture_output=True,text=True)
    checkpoint=torch.load(output/"initial.pt",map_location="cpu",weights_only=False)
    assert checkpoint["env_steps"]==0 and checkpoint["training_args"]["scenario"]=="simple_paper_1v1" and "model_state_dict" in checkpoint

def seed_row(seed,initial=-10.,best=-5.,latest=-6.,completed=True):
    row={"seed":seed,"completed":completed,"finite":True,"training_seconds":1.,"numerical_invalid_episodes":0,"crashes":0,"boundary_deaths":0,"flight_envelope_violation_episodes":0,
      "initial_deterministic_return":initial,"best_deterministic_return":best,"latest_deterministic_return":latest,
      "best_improvement":best-initial,"latest_improvement":latest-initial}
    for checkpoint,value in (("initial",initial),("best",best),("latest",latest)):
        row[f"{checkpoint}_stochastic_mean_return"]=value;row[f"{checkpoint}_stochastic_red_win_rate"]=1.
    return row

def make_fake_root(path,rows):
    for row in rows:
        seed_dir=path/f"seed_{row['seed']}";seed_dir.mkdir(parents=True);(seed_dir/"seed_summary.json").write_text(json.dumps(row),encoding="utf-8")

def test_summary_reads_minimal_fake_directory_and_writes_outputs(tmp_path):
    make_fake_root(tmp_path,[seed_row(1),seed_row(2),seed_row(3)])
    summary=summarize_root(tmp_path)
    assert summary["aggregate"]["completed_seed_count"]==3
    assert (tmp_path/"multiseed_summary.json").exists() and (tmp_path/"multiseed_summary.csv").exists() and (tmp_path/"multiseed_report.md").exists()

def test_mean_std_uses_population_standard_deviation():
    result=mean_std([1,2,3]);assert result["mean"]==2 and result["std"]==pytest.approx((2/3)**.5)

def test_missing_seed_is_reported_instead_of_silently_ignored(tmp_path):
    make_fake_root(tmp_path,[seed_row(1),seed_row(2)])
    summary=summarize_root(tmp_path)
    assert summary["aggregate"]["completed_seed_count"]==2 and summary["seeds"][2]["failure"]=="missing seed_summary.json"

def test_abc_criteria_apply_exact_cross_seed_thresholds():
    passing=[seed_row(1),seed_row(2),seed_row(3,latest=-11.)]
    assert classify(passing)=={"A_stable":True,"B_learning_signal":True,"C_retention":True,"best_improved_seed_count":3,"latest_improved_seed_count":2}
    one_seed_only=[seed_row(1),seed_row(2,best=-12.,latest=-12.),seed_row(3,best=-12.,latest=-12.)]
    assert not classify(one_seed_only)["B_learning_signal"] and not classify(one_seed_only)["C_retention"]
