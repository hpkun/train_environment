import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from aircombat_env_v1.scripts import eval_simple_mappo as eval_module
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model
from aircombat_env_v1.scripts.run_simple_mappo_multiseed import build_parser,classify,experiment_reward_metadata,mean_std,summarize_root
from aircombat_env_v1.scripts.train_simple_mappo import EPISODE_COMPONENT_FIELDS,EVAL_FIELDS,build_environment_contract,build_episode_component_row,build_parser as build_train_parser,save_checkpoint
from aircombat_env_v1.simple_hetero_reward import reward_contract_metadata
from aircombat_env_v1.simple_mappo import SharedMAPPOActorCritic
import torch

class _OneStepModel:
    def act(self,obs,state,deterministic):
        return torch.zeros((obs.shape[0],3)),None,None,None

class _OneStepEnv:
    def __init__(self,scenario,controlled_team="red",hetero_perception_mode="paper_fused",hetero_reward_mode="paper_table1_v2"):
        self.scenario=scenario;self.hetero_perception_mode=hetero_perception_mode;self.hetero_reward_mode=hetero_reward_mode
    def close(self):
        pass

class _OneStepAdapter:
    num_agents=1;agent_ids=["red_mav_0"]
    def __init__(self,env):
        self.env=env
    def reset(self,seed=None):
        return np.zeros((1,1),np.float32),np.zeros(1,np.float32),{}
    def step(self,actions):
        info={"winner":"red","termination_reason":"timeout","alive_red":np.int64(2),"alive_blue":np.int64(1),
          "missiles_fired":np.int64(3),"missile_hits":np.int64(2),"red_crashes":np.int64(1),"blue_crashes":np.int64(0),
          "boundary_deaths":np.int64(1),"numerical_invalid":np.bool_(False),"flight_envelope_violation":np.bool_(True),
          "mav_alive":np.bool_(True),"red_uav_alive":np.int64(1),"red_team_failed_by_mav_loss":np.bool_(False),
          "red_team_failed_by_uav_loss":np.bool_(True),"red_missile_kills":np.int64(2),"blue_missile_kills":np.int64(1)}
        info.update({"relay_only_track_count":np.int64(4),"mav_support_active":np.bool_(True),
          "target_selection_source":{"red_uav_0":"mav_shared","red_uav_1":"direct"},
          "visible_enemy_ids_by_agent":{"red_uav_0":["blue_0","blue_1"],"red_uav_1":["blue_0"]},
          "direct_enemy_ids_by_agent":{"red_uav_0":[],"red_uav_1":["blue_0"]},
          "shared_enemy_ids_by_agent":{"red_uav_0":["blue_0","blue_1"],"red_uav_1":["blue_0"]},
          "red_uav_launches_using_shared_track":np.int64(1),"red_uav_launches_using_direct_track":np.int64(2),
          "reward_components":{"red_mav_0":{"r_safety":np.float64(.1),"r_support":np.float64(.2),"r_event":np.float64(100),
            "r_event_death":np.float64(0),"r_event_team_contribution":np.float64(100),"r_support_awareness":np.float64(.3),
            "r_support_position":np.float64(.4),"total_dense":np.float64(.3)}}})
        return np.zeros((1,1),np.float32),np.zeros(1,np.float32),np.zeros(1,np.float32),True,np.zeros(1,np.float32),info

def _one_step_evaluation(monkeypatch,scenario):
    monkeypatch.setattr(eval_module,"SimpleTAMCombatEnv",_OneStepEnv)
    monkeypatch.setattr(eval_module,"SimpleMAPPOAdapter",_OneStepAdapter)
    return eval_module.evaluate_model(_OneStepModel(),scenario,1,torch.device("cpu"),True,1,True)

def test_evaluation_result_and_numpy_episode_fields_are_json_serializable(monkeypatch):
    result=_one_step_evaluation(monkeypatch,"simple_paper_3v2_hetero")
    json.dumps(result)
    for key in ("missile_launches","missile_hits","crashes","boundary_deaths","numerical_invalid_episodes","flight_envelope_violation_episodes"):
        assert type(result[key]) is int
    row=result["rows"][0]
    for key in ("length","red_alive","blue_alive","missile_launches","missile_hits","crashes","boundary_deaths","numerical_invalid"):
        assert type(row[key]) is int
    assert type(row["timeout"]) is bool and type(row["envelope"]) is bool

def test_3v2_role_metrics_exist_with_native_types(monkeypatch):
    result=_one_step_evaluation(monkeypatch,"simple_paper_3v2_hetero")
    for key in ("mav_survival_rate","mean_red_uav_alive","mav_loss_rate","red_uav_team_loss_rate"):
        assert type(result[key]) is float
    assert type(result["red_missile_kills"]) is int and type(result["blue_missile_kills"]) is int
    for key in ("mean_relay_only_tracks_per_step","fraction_steps_with_mav_support",
      "fraction_uav_target_selections_from_shared_tracks","mean_visible_enemies_per_red_uav",
      "mean_direct_enemies_per_red_uav","mean_shared_enemies_per_red_uav"):
        assert type(result[key]) is float
    assert result["mean_relay_only_tracks_per_step"]==4. and result["fraction_steps_with_mav_support"]==1.
    assert result["red_uav_launches_using_shared_track"]==1 and result["red_uav_launches_using_direct_track"]==2
    for key in ("mean_mav_return","mean_red_uav_return","mean_mav_safety_return","mean_mav_support_return","mean_mav_event_return",
      "mean_mav_death_penalty","mean_mav_team_contribution","mean_mav_awareness_return","mean_mav_position_return","mean_mav_dense_return"):
        assert type(result[key]) is float
    assert result["hetero_perception_mode"]=="paper_fused" and result["hetero_reward_mode"]=="paper_table1_v2"
    assert result["reward_contract_schema_version"]=="1" and result["reward_contract_version"]=="paper_table1_mav_reward_v2"
    assert result["checkpoint_reward_contract_known"] is False
    for key in ("checkpoint_trained_perception_mode","checkpoint_trained_reward_mode","checkpoint_trained_reward_contract_version"):
        assert result[key] is None

def test_heterogeneous_checkpoint_shape_is_valid_in_both_perception_modes(tmp_path):
    model=SharedMAPPOActorCritic(81,243);path=tmp_path/"checkpoint.pt"
    torch.save({"scenario":"simple_paper_3v2_hetero","obs_dim":81,"state_dim":243,"action_dim":3,
      "agent_ids":["red_uav_0","red_uav_1","red_mav_0"],"model_state_dict":model.state_dict()},path)
    for mode in ("paper_fused","uav_only_ablation"):
        loaded=eval_module.load_checkpoint(path,"simple_paper_3v2_hetero",torch.device("cpu"),mode)
        assert loaded.obs_dim==81 and loaded.state_dim==243 and loaded.checkpoint_reward_contract_known is False

def _checkpoint_contract(mode="legacy_v1"):
    meta=reward_contract_metadata(mode)
    return {"environment_contract_schema_version":"1","scenario":"simple_paper_3v2_hetero","obs_dim":81,"state_dim":243,"action_dim":3,
      "agent_ids":["red_uav_0","red_uav_1","red_mav_0"],"hetero_perception_mode":"paper_fused","hetero_reward_mode":mode,
      "reward_contract_schema_version":meta["reward_contract_schema_version"],"reward_contract_version":meta["reward_contract_version"],"reward_config":meta["reward_config"]}

def _write_checkpoint(path,contract):
    model=SharedMAPPOActorCritic(81,243)
    torch.save({"scenario":"simple_paper_3v2_hetero","obs_dim":81,"state_dim":243,"action_dim":3,
      "agent_ids":["red_uav_0","red_uav_1","red_mav_0"],"model_state_dict":model.state_dict(),
      "environment_contract":contract},path)
    return path

def test_checkpoint_reward_contract_mismatch_rejected_and_override_allowed(tmp_path):
    path=_write_checkpoint(tmp_path/"checkpoint.pt",_checkpoint_contract("legacy_v1"))
    with pytest.raises(ValueError,match="does not match checkpoint"):
        eval_module.load_checkpoint(path,"simple_paper_3v2_hetero",torch.device("cpu"),hetero_reward_mode="paper_table1_v2")
    loaded=eval_module.load_checkpoint(path,"simple_paper_3v2_hetero",torch.device("cpu"),hetero_reward_mode="paper_table1_v2",allow_reward_mode_override=True)
    assert loaded.checkpoint_reward_contract_known is True and loaded.checkpoint_trained_reward_mode=="legacy_v1"
    assert loaded.checkpoint_trained_reward_contract_version=="legacy_mav_reward_v1"
    loaded=eval_module.load_checkpoint(path,"simple_paper_3v2_hetero",torch.device("cpu"),hetero_perception_mode="uav_only_ablation",hetero_reward_mode="legacy_v1")
    assert loaded.checkpoint_trained_perception_mode=="paper_fused"

def test_checkpoint_rejects_inconsistent_version_and_config(tmp_path):
    contract=_checkpoint_contract();contract["reward_contract_version"]="wrong"
    with pytest.raises(ValueError,match="version are inconsistent"):
        eval_module.load_checkpoint(_write_checkpoint(tmp_path/"version.pt",contract),"simple_paper_3v2_hetero",torch.device("cpu"),hetero_reward_mode="legacy_v1")
    contract=_checkpoint_contract();contract["reward_config"]={"wrong":True}
    with pytest.raises(ValueError,match="reward_config"):
        eval_module.load_checkpoint(_write_checkpoint(tmp_path/"config.pt",contract),"simple_paper_3v2_hetero",torch.device("cpu"),hetero_reward_mode="legacy_v1")

def test_old_checkpoint_contract_remains_unknown_and_loadable(tmp_path):
    old={"scenario":"simple_paper_3v2_hetero","obs_dim":81,"state_dim":243,"action_dim":3,
      "agent_ids":["red_uav_0","red_uav_1","red_mav_0"],"hetero_reward_mode":"legacy_v1"}
    loaded=eval_module.load_checkpoint(_write_checkpoint(tmp_path/"old.pt",old),"simple_paper_3v2_hetero",torch.device("cpu"),hetero_reward_mode="paper_table1_v2")
    assert loaded.checkpoint_reward_contract_known is False and loaded.checkpoint_trained_reward_mode=="legacy_v1"

@pytest.mark.parametrize("scenario",["simple_paper_1v1","simple_paper_2v2"])
def test_non_heterogeneous_evaluation_does_not_add_role_metrics(monkeypatch,scenario):
    result=_one_step_evaluation(monkeypatch,scenario)
    assert "mav_survival_rate" not in result and "mean_red_uav_alive" not in result

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
    assert checkpoint["environment_contract"]["hetero_reward_mode"] is None
    assert checkpoint["environment_contract"]["reward_contract_version"] is None
    assert checkpoint["environment_contract"]["reward_config"] is None
    with (output/"episode_reward_components.csv").open(encoding="utf-8") as handle:
        assert tuple(handle.readline().strip().split(","))==EPISODE_COMPONENT_FIELDS

def test_training_parser_accepts_heterogeneous_contract_modes():
    args=build_train_parser().parse_args(["--scenario","simple_paper_3v2_hetero","--output-dir","out","--hetero-perception-mode","uav_only_ablation","--hetero-reward-mode","legacy_v1"])
    assert args.hetero_perception_mode=="uav_only_ablation" and args.hetero_reward_mode=="legacy_v1"

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
    assert summary["hetero_reward_mode"] is None and summary["reward_contract_version"] is None
    assert (tmp_path/"multiseed_summary.json").exists() and (tmp_path/"multiseed_summary.csv").exists() and (tmp_path/"multiseed_report.md").exists()

def test_mean_std_uses_sample_standard_deviation():
    result=mean_std([1,2,3]);assert result["mean"]==2 and result["std"]==1

def test_single_seed_standard_deviation_is_zero():
    assert mean_std([7])=={"mean":7.,"std":0.}

def test_parameterized_script_accepts_simple_paper_2v2():
    args=build_parser().parse_args(["--scenario","simple_paper_2v2","--total-env-steps","12","--seeds","3","4","--output-dir","out"])
    assert args.scenario=="simple_paper_2v2" and args.total_env_steps==12 and args.seeds==[3,4] and args.output_dir==Path("out")

def test_multiseed_parser_accepts_heterogeneous_contract_modes():
    args=build_parser().parse_args(["--scenario","simple_paper_3v2_hetero","--hetero-perception-mode","uav_only_ablation","--hetero-reward-mode","legacy_v1"])
    assert args.hetero_perception_mode=="uav_only_ablation" and args.hetero_reward_mode=="legacy_v1"

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

def test_environment_contract_and_multiseed_metadata_are_mode_specific():
    adapter=type("Adapter",(),{"obs_dim":81,"state_dim":243,"action_dim":3,"agent_ids":["a","b","c"]})()
    args=type("Args",(),{"scenario":"simple_paper_3v2_hetero","hetero_perception_mode":"paper_fused","hetero_reward_mode":"legacy_v1"})()
    contract=build_environment_contract(args,adapter)
    assert contract["reward_contract_version"]=="legacy_mav_reward_v1" and contract["reward_config"]["mav_danger_distance_m"]==5000
    assert experiment_reward_metadata("simple_paper_3v2_hetero","paper_table1_v2")["reward_contract_version"]=="paper_table1_mav_reward_v2"
    for scenario in ("simple_paper_1v1","simple_paper_2v2"):
        meta=experiment_reward_metadata(scenario,"paper_table1_v2")
        assert meta["hetero_reward_mode"] is None and meta["reward_contract_version"] is None and meta["reward_config"] is None

def test_heterogeneous_multiseed_summary_uses_requested_reward_contract(tmp_path):
    make_fake_root(tmp_path,[seed_row(1)])
    summary=summarize_root(tmp_path,(1,),"simple_paper_3v2_hetero","paper_fused","legacy_v1")
    assert summary["reward_contract_schema_version"]=="1"
    assert summary["reward_contract_version"]=="legacy_mav_reward_v1"
    assert summary["reward_config"]==reward_contract_metadata("legacy_v1")["reward_config"]

@pytest.mark.parametrize("mode,version",[("legacy_v1","legacy_mav_reward_v1"),("paper_table1_v2","paper_table1_mav_reward_v2")])
def test_saved_heterogeneous_checkpoint_uses_mode_contract(tmp_path,mode,version):
    adapter=type("Adapter",(),{"obs_dim":81,"state_dim":243,"action_dim":3,"agent_ids":["red_uav_0","red_uav_1","red_mav_0"]})()
    args=type("Args",(),{"scenario":"simple_paper_3v2_hetero","hetero_perception_mode":"paper_fused","hetero_reward_mode":mode})()
    path=tmp_path/f"{mode}.pt";save_checkpoint(path,SharedMAPPOActorCritic(81,243),adapter,args,0)
    contract=torch.load(path,map_location="cpu",weights_only=False)["environment_contract"]
    assert contract["reward_contract_version"]==version and contract["reward_config"]==reward_contract_metadata(mode)["reward_config"]

def test_episode_component_rows_are_isolated_csv_serializable_and_header_complete():
    import csv,io
    info={"winner":"red","termination_reason":"blue_eliminated","red_missile_kills":1,"blue_missile_kills":0,
      "mav_alive":True,"red_uav_alive":2,"numerical_invalid":0,"red_crashes":0,"blue_crashes":0,"boundary_deaths":0}
    first=build_episode_component_row(1,10,10,info,5.,{"red_mav_0":1.,"red_uav_0":2.,"red_uav_1":3.},{"mav_safety_sum":7.,"relay_only_track_steps":4.},"simple_paper_3v2_hetero")
    second=build_episode_component_row(2,20,10,info,-1.,{"red_mav_0":-2.,"red_uav_0":0.,"red_uav_1":0.},{"mav_support_sum":9.},"simple_paper_3v2_hetero")
    assert first["episode"]==1 and second["episode"]==2
    assert first["mav_return"]==1 and second["mav_return"]==-2
    assert first["mav_safety_sum"]==7 and second["mav_safety_sum"]==0
    assert first["relay_only_track_steps"]==4 and second["relay_only_track_steps"]==0
    assert second["mav_support_sum"]==9 and set(first)==set(second)==set(EPISODE_COMPONENT_FIELDS)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=EPISODE_COMPONENT_FIELDS);writer.writeheader();writer.writerows([first,second])
    assert len(list(csv.DictReader(io.StringIO(stream.getvalue()))))==2

def test_eval_fields_include_heterogeneous_periodic_diagnostics():
    for key in ("hetero_perception_mode","hetero_reward_mode","reward_contract_version","mav_survival_rate","mean_red_uav_alive",
      "mean_mav_return","mean_red_uav_return","mean_mav_safety_return","mean_mav_support_return","mean_mav_event_return",
      "mean_mav_death_penalty","mean_mav_team_contribution","mean_mav_awareness_return","mean_mav_position_return",
      "mean_mav_dense_return","mean_relay_only_tracks_per_step","fraction_steps_with_mav_support"):
        assert key in EVAL_FIELDS
