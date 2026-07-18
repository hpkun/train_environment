import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model
from aircombat_env_v1.scripts.run_simple_mappo_multiseed import build_parser,classify,mean_std,summarize_root
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

def test_evaluation_restores_numpy_rng_state():
    model=SharedMAPPOActorCritic(61,61);np.random.seed(123);state=np.random.get_state();evaluate(model,False,20001);actual=np.random.random(5)
    np.random.set_state(state);assert np.array_equal(actual,np.random.random(5))

def test_evaluation_restores_torch_cpu_rng_state():
    model=SharedMAPPOActorCritic(61,61);torch.manual_seed(123);state=torch.random.get_rng_state();evaluate(model,False,20001);actual=torch.rand(5)
    torch.random.set_rng_state(state);assert torch.equal(actual,torch.rand(5))

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_evaluation_restores_cuda_rng_states():
    model=SharedMAPPOActorCritic(61,61);torch.cuda.manual_seed_all(123);states=[state.clone() for state in torch.cuda.get_rng_state_all()]
    evaluate(model,False,20001);assert all(torch.equal(a,b) for a,b in zip(states,torch.cuda.get_rng_state_all()))

def test_training_saves_initial_checkpoint_at_zero_steps(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts"/"train_simple_mappo.py";output=tmp_path/"train"
    subprocess.run([sys.executable,str(script),"--scenario","simple_paper_1v1","--total-env-steps","0","--eval-episodes","1","--output-dir",str(output)],check=True,capture_output=True,text=True)
    checkpoint=torch.load(output/"initial.pt",map_location="cpu",weights_only=False)
    assert checkpoint["env_steps"]==0 and checkpoint["training_args"]["scenario"]=="simple_paper_1v1" and "model_state_dict" in checkpoint

def seed_row(seed,initial=-10.,best=-5.,latest=-6.,completed=True,stochastic_best=None):
    stochastic_best=best if stochastic_best is None else stochastic_best
    row={"seed":seed,"completed":completed,"finite":True,"training_seconds":1.,"checkpoint_numerical_invalid_episodes":0,"checkpoint_crashes":0,"checkpoint_boundary_deaths":0,"checkpoint_flight_envelope_violation_episodes":0,"recent_train_numerical_invalid_episodes":0,
      "initial_deterministic_return":initial,"best_deterministic_return":best,"latest_deterministic_return":latest,
      "best_improvement":best-initial,"latest_improvement":latest-initial,"best_stochastic_improvement":stochastic_best-initial,"latest_stochastic_improvement":latest-initial}
    for checkpoint,value in (("initial",initial),("best",stochastic_best),("latest",latest)):
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

def test_mean_std_uses_sample_standard_deviation():
    result=mean_std([1,2,3]);assert result["mean"]==2 and result["std"]==1

def test_single_seed_standard_deviation_is_zero():
    assert mean_std([7])=={"mean":7.,"std":0.}

def test_parameterized_script_accepts_simple_paper_2v2():
    args=build_parser().parse_args(["--scenario","simple_paper_2v2","--total-env-steps","12","--seeds","3","4","--output-dir","out"])
    assert args.scenario=="simple_paper_2v2" and args.total_env_steps==12 and args.seeds==[3,4] and args.output_dir==Path("out")

def test_2v2_summary_reads_minimal_fake_directory(tmp_path):
    make_fake_root(tmp_path,[seed_row(1)])
    summary=summarize_root(tmp_path,(1,),"simple_paper_2v2")
    assert summary["scenario"]=="simple_paper_2v2" and summary["aggregate"]["completed_seed_count"]==1

def test_missing_seed_is_reported_instead_of_silently_ignored(tmp_path):
    make_fake_root(tmp_path,[seed_row(1),seed_row(2)])
    summary=summarize_root(tmp_path)
    assert summary["aggregate"]["completed_seed_count"]==2 and summary["seeds"][2]["failure"]=="missing seed_summary.json"

def test_abcd_criteria_apply_exact_cross_seed_thresholds():
    passing=[seed_row(1),seed_row(2),seed_row(3,latest=-11.)]
    assert classify(passing)=={"A_stable":True,"B_learning_signal":True,"C_stochastic_robustness":True,"D_latest_retention":True,"best_improved_seed_count":3,"best_stochastic_improved_seed_count":3,"latest_improved_seed_count":2}
    one_seed_only=[seed_row(1),seed_row(2,best=-12.,latest=-12.,stochastic_best=-12.),seed_row(3,best=-12.,latest=-12.,stochastic_best=-12.)]
    result=classify(one_seed_only);assert not result["B_learning_signal"] and not result["C_stochastic_robustness"] and not result["D_latest_retention"]
